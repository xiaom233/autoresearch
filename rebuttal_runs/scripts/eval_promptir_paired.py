#!/usr/bin/env python3
"""Evaluate official PromptIR or DCPT (5D) weights on paired SIDD crops."""
import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark/PromptIR"))
sys.path.insert(0, str(ROOT / "rebuttal_runs/scripts"))
from net.model import PromptIR  # noqa: E402
from eval_paired import psnr, ssim, lpips_fn  # noqa: E402


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--variant", choices=["promptir", "dcpt"], required=True)
    ap.add_argument("--controller", default=None)
    ap.add_argument("--scenario", default="sidd")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-id", type=int, required=True)
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=None)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda")
    net = PromptIR(decoder=True, dim=48, num_blocks=[4, 6, 6, 8], heads=[1, 2, 4, 8],
                   ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias",
                   num_refinement_blocks=4).to(device).eval()
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if args.variant == "dcpt":
        net.load_state_dict(ck["params"], strict=False)
    else:
        state = ck["state_dict"]
        state = {k.replace("net.", ""): v for k, v in state.items()}
        net.load_state_dict(state, strict=False)
    ck_hash = sha_file(args.checkpoint)
    controller = args.controller or ("DCPT-5D" if args.variant == "dcpt" else "PromptIR")

    d = Path(args.pairs_dir)
    pairs = []
    for c in sorted(d.glob("clean_*.png")):
        idx = c.stem.split("_")[1]
        lq = d / f"degraded_{idx}.png"
        if lq.exists():
            pairs.append((lq, c))
    if args.num_gpus > 1:
        shard = args.shard_id if args.shard_id is not None else args.gpu_id
        pairs = [p for i, p in enumerate(pairs) if i % args.num_gpus == shard]
    out = open(args.out, "a")
    for lq_p, c_p in pairs:
        lq = np.array(Image.open(lq_p).convert("RGB"), dtype=np.uint8)
        clean = np.array(Image.open(c_p).convert("RGB"), dtype=np.uint8)
        x = torch.from_numpy(lq).permute(2, 0, 1).float().div(255).unsqueeze(0).cuda()
        c = torch.from_numpy(clean).permute(2, 0, 1).float().div(255).cuda()
        h, w = x.shape[2], x.shape[3]
        ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                y = net(x)
            if isinstance(y, (tuple, list)):
                y = y[0]
        pred = y[:, :, :h, :w].float().clamp(0, 1)[0]
        rec = {"controller": controller, "scenario": args.scenario,
               "dataset": "E4", "image_id": c_p.stem,
               "checkpoint": args.checkpoint, "checkpoint_sha256": ck_hash,
               "variant": args.variant, "tile": 0,
               "psnr_rgb": psnr(pred, c), "ssim_rgb": ssim(pred, c),
               "lpips_vgg": float(lpips_fn()(pred.unsqueeze(0) * 2 - 1,
                                             c.unsqueeze(0) * 2 - 1).item())}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        print(f"[{controller}] {c_p.stem} PSNR={rec['psnr_rgb']:.2f}", flush=True)
    out.close()


if __name__ == "__main__":
    main()
