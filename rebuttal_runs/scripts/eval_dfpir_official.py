#!/usr/bin/env python3
"""Evaluate the official DFPIR 5D checkpoint on paired images.

Protocol follows scripts/benchmark_inference.py:
  official state_dict, text_code = randn(1,512), pad 16, tile 1024 (no overlap).
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark/DFPIR"))
sys.path.insert(0, str(ROOT / "rebuttal_runs/scripts"))
from net.model import ChannelShuffle_skip_textguaid  # noqa: E402
from eval_checkpoint import psnr, ssim, lpips_fn  # noqa: E402


def load_rgb(p):
    return np.array(Image.open(p).convert("RGB"), dtype=np.uint8)


def to_tensor(a):
    return torch.from_numpy(a).permute(2, 0, 1).float().div(255.0)


def pad16(x):
    _, _, h, w = x.shape
    ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, h, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--controller", default="DFPIR-official")
    ap.add_argument("--scenario", default="e4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--text-seed", type=int, default=42)
    ap.add_argument("--text-prompt", default="", help="official CLIP text prompt; empty = random text_code")
    ap.add_argument("--gpu-id", type=int, default=None)
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=None)
    args = ap.parse_args()
    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda")
    model = ChannelShuffle_skip_textguaid(device=device).to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    sd = ck.get("state_dict", ck.get("params", ck.get("model", ck)))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()
    if args.text_prompt:
        import clip
        clip_model, _ = clip.load("ViT-B/32", device=device)
        for p in clip_model.parameters():
            p.requires_grad = False
        text_code = clip_model.encode_text(clip.tokenize([args.text_prompt]).to(device)).to(dtype=torch.float32)
        print(f"[official] CLIP prompt: {args.text_prompt!r}", flush=True)
    else:
        torch.manual_seed(args.text_seed)
        text_code = torch.randn(1, 512, device=device)
    print(f"[official] missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    pairs = []
    d = Path(args.pairs_dir)
    for c in sorted(d.glob("clean_*.png")):
        idx = c.stem.split("_")[1]
        lq = d / f"degraded_{idx}.png"
        if lq.exists():
            pairs.append((lq, c))
    if args.num_gpus > 1:
        shard = args.shard_id if args.shard_id is not None else (args.gpu_id or 0)
        pairs = [p for i, p in enumerate(pairs) if i % args.num_gpus == shard]
    out = open(args.out, "a")
    ck_hash = hashlib.sha256(open(args.checkpoint, "rb").read()).hexdigest()
    for lq_p, c_p in pairs:
        lq = to_tensor(load_rgb(lq_p)).unsqueeze(0).cuda()
        clean = to_tensor(load_rgb(c_p)).cuda()
        with torch.no_grad():
            if args.tile > 0 and max(lq.shape[2], lq.shape[3]) > args.tile:
                _, _, H, W = lq.shape
                T = args.tile
                pred = torch.zeros(1, 3, H, W, device=device)
                for y in range(0, H, T):
                    for x in range(0, W, T):
                        y2, x2 = min(y + T, H), min(x + T, W)
                        tile = lq[:, :, y:y2, x:x2]
                        th, tw = tile.shape[2], tile.shape[3]
                        if th < T or tw < T:
                            tile = F.pad(tile, (0, T - tw, 0, T - th))
                        tile, _, _ = pad16(tile)
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            o = model(tile, text_code.expand(tile.shape[0], -1))
                        pred[:, :, y:y2, x:x2] = o[:, :, :th, :tw].float()
            else:
                x, h, w = pad16(lq)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    o = model(x, text_code.expand(x.shape[0], -1))
                pred = o[:, :, :h, :w].float()
            pred = pred.clamp(0, 1)
        rec = {"controller": args.controller, "scenario": args.scenario,
               "dataset": "E4", "image_id": c_p.stem,
               "checkpoint": args.checkpoint, "checkpoint_sha256": ck_hash,
               "text_code_seed": args.text_seed, "text_prompt": args.text_prompt or None, "tile": args.tile,
               "psnr_rgb": psnr(pred[0], clean), "ssim_rgb": ssim(pred[0], clean),
               "lpips_vgg": float(lpips_fn()(pred * 2 - 1, clean.unsqueeze(0) * 2 - 1).item())}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
        print(f"[official-eval] {args.controller} {c_p.stem} PSNR={rec['psnr_rgb']:.2f}", flush=True)
    out.close()


if __name__ == "__main__":
    main()
