#!/usr/bin/env python3
"""Evaluate RestoreNet or DFPIR on real paired images (clean_i/degraded_i).

No synthetic GT parameters are applied. Supports full-image or tiled inference.
Run with the dfpir env (has torch, lpips, einops, DFPIR deps).
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rebuttal_runs" / "scripts"))
sys.path.insert(0, str(ROOT / "resource" /
                   "Degradation-Aware-Feature-Perturbation-for-All-in-One-Image-Restoration-Modified-Version-"))

from PIL import Image  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def load_rgb(p):
    return np.array(Image.open(p).convert("RGB"), dtype=np.uint8)


def to_tensor(a):
    return torch.from_numpy(a).permute(2, 0, 1).float().div(255.0)


def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    return 100.0 if mse.item() == 0 else float(10 * np.log10(1.0 / mse.item()))


_SSIM = {}
def ssim(pred, target):
    c = pred.shape[0]
    k = _SSIM.get(c)
    if k is None or k.device != pred.device:
        coords = torch.arange(11, dtype=torch.float32, device=pred.device) - 5.0
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2)); g = g / g.sum()
        k = (g[:, None] * g[None, :]).view(1, 1, 11, 11).repeat(c, 1, 1, 1)
        _SSIM[c] = k
    p, t = pred.unsqueeze(0), target.unsqueeze(0)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx = F.conv2d(p, k, groups=c, padding=5); my = F.conv2d(t, k, groups=c, padding=5)
    sx = F.conv2d(p * p, k, groups=c, padding=5) - mx * mx
    sy = F.conv2d(t * t, k, groups=c, padding=5) - my * my
    sxy = F.conv2d(p * t, k, groups=c, padding=5) - mx * my
    m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sx + sy + c2))
    return float(m.mean().item())


_LP = None
def lpips_fn():
    global _LP
    if _LP is None:
        import lpips
        _LP = lpips.LPIPS(net="vgg").cuda().eval()
    return _LP


@torch.no_grad()
def tiled_restorenet(model, x, tile, overlap):
    _, _, h, w = x.shape
    out = torch.zeros_like(x); cnt = torch.zeros(1, 1, h, w, device=x.device)
    step = tile - overlap
    for y in range(0, h, step):
        for xx in range(0, w, step):
            y2, x2 = min(y + tile, h), min(xx + tile, w)
            y1, x1 = max(0, y2 - tile), max(0, x2 - tile)
            patch = x[:, :, y1:y2, x1:x2]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(patch)
            out[:, :, y1:y2, x1:x2] += pred.float()
            cnt[:, :, y1:y2, x1:x2] += 1
            if x2 >= w: break
        if y2 >= h: break
    return (out / cnt).clamp(0, 1)


def load_restorenet(ckpt, meta):
    from model import RestoreNet
    m = meta.get("model", meta)
    kw = dict(in_ch=3, embed_dim=m.get("embed_dim", 64), depths=tuple(m.get("depths", (2, 2, 2, 2))),
              num_heads=tuple(m.get("num_heads", (4, 4, 4, 4))), window_size=m.get("window_size", 8),
              mlp_ratio=m.get("mlp_ratio", 2), activation=m.get("activation", "gelu"),
              head_dim=m.get("head_dim", 16), attention_type=m.get("attention_type", "swin"),
              use_gcm=bool(m.get("gcm", False)), use_csn=bool(m.get("csn", False)),
              use_color_pre=bool(m.get("color_pre", False)), use_color_mlp=bool(m.get("color_mlp", False)),
              use_pcp=bool(m.get("pcp", False)), use_freqmod=bool(m.get("freqmod", False)),
              use_channel_curve=bool(m.get("channel_curve", False)),
              use_dual_branch=bool(m.get("dual_branch", False)),
              use_simple_gate=bool(m.get("simple_gate", False)), use_sca=bool(m.get("sca", False)),
              use_gdfn=bool(m.get("gdfn", False)), use_fpro=bool(m.get("fpro", False)))
    model = RestoreNet(**kw).cuda()
    ck = torch.load(ckpt, map_location="cuda", weights_only=True)
    model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    model.eval()
    return model


def load_dfpir(ckpt):
    from train_dfpir_budget import DFPIRBlind
    from net.model import ChannelShuffle_skip_textguaid
    base = ChannelShuffle_skip_textguaid(dim=48, num_blocks=[4, 6, 6, 8], heads=[1, 2, 4, 8],
                                         ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias")
    model = DFPIRBlind(base).cuda()
    ck = torch.load(ckpt, map_location="cuda", weights_only=True)
    model.load_state_dict(ck["model"]); model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--model-kind", choices=["restorenet", "dfpir"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--meta", default=None, help="JSON with EXP_META model dict")
    ap.add_argument("--controller", default="method")
    ap.add_argument("--scenario", default="e4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--gpu-id", type=int, default=None)
    ap.add_argument("--shard-id", type=int, default=None)
    ap.add_argument("--num-gpus", type=int, default=1)
    args = ap.parse_args()
    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)
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
    meta = json.load(open(args.meta)) if args.meta else {}
    if args.model_kind == "restorenet":
        model = load_restorenet(args.checkpoint, meta)
    else:
        model = load_dfpir(args.checkpoint)
    out = open(args.out, "a")
    for lq_p, c_p in pairs:
        lq = to_tensor(load_rgb(lq_p)).unsqueeze(0).cuda()
        clean = to_tensor(load_rgb(c_p)).cuda()
        with torch.no_grad():
            if args.model_kind == "restorenet":
                if args.tile > 0:
                    pred = tiled_restorenet(model, lq, args.tile, args.overlap)[0]
                else:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        pred = model(lq)[0].float().clamp(0, 1)
            else:
                from train_dfpir_budget import tiled_infer
                pred = tiled_infer(model, lq, tile=args.tile or 512, overlap=args.overlap)[0]
        rec = {"controller": args.controller, "scenario": args.scenario,
               "dataset": "E4", "image_id": c_p.stem,
               "psnr_rgb": psnr(pred, clean), "ssim_rgb": ssim(pred, clean),
               "lpips_vgg": float(lpips_fn()(pred.unsqueeze(0) * 2 - 1,
                                              clean.unsqueeze(0) * 2 - 1).item())}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
        print(f"[e4-eval] {args.controller} {c_p.stem} PSNR={rec['psnr_rgb']:.2f}", flush=True)
    out.close()


if __name__ == "__main__":
    main()
