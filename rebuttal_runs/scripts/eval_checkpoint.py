#!/usr/bin/env python3
"""Explicit checkpoint evaluator for rebuttal experiments.

Modes:
  support : evaluate every checkpoint in the A0 manifest on its 5 support pairs.
  test    : evaluate explicitly listed checkpoints on DIV2K/LSDIR with a
            deterministic, hash-seeded LQ realization (same for every method).

No fallback logic: a missing checkpoint or dataset is a hard error.
Per-image metrics and LQ hashes are written as JSONL.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model import RestoreNet  # noqa: E402
from x_distortion import add_distortion  # noqa: E402

DEVICE = torch.device("cuda")
AMP = torch.amp.autocast("cuda", dtype=torch.bfloat16)

_RGB_TO_Y = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)
_SSIM_KERNEL = {}
_LPIPS = None


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rgb(path):
    img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    return img


def to_tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)


def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse.item() == 0:
        return 100.0
    return float(20 * math.log10(1.0) - 10 * math.log10(mse.item()))


def _kernel(channels, device):
    key = (channels, str(device))
    if key not in _SSIM_KERNEL:
        coords = torch.arange(11, dtype=torch.float32, device=device) - 5.0
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        g = g / g.sum()
        k = (g[:, None] * g[None, :]).view(1, 1, 11, 11).repeat(channels, 1, 1, 1)
        _SSIM_KERNEL[key] = k
    return _SSIM_KERNEL[key]


def ssim(pred, target):
    c = pred.shape[0]
    k = _kernel(c, pred.device)
    p, t = pred.unsqueeze(0), target.unsqueeze(0)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx = F.conv2d(p, k, groups=c, padding=5)
    my = F.conv2d(t, k, groups=c, padding=5)
    mx2, my2 = mx ** 2, my ** 2
    sx2 = F.conv2d(p ** 2, k, groups=c, padding=5) - mx2
    sy2 = F.conv2d(t ** 2, k, groups=c, padding=5) - my2
    sxy = F.conv2d(p * t, k, groups=c, padding=5) - mx * my
    m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx2 + my2 + c1) * (sx2 + sy2 + c2))
    return float(m.mean().item())


def lpips_fn():
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="vgg").to(DEVICE).eval()
    return _LPIPS


def pad_to_window(x, ws):
    _, _, h, w = x.shape
    ph, pw = (ws - h % ws) % ws, (ws - w % ws) % ws
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, h, w


def build_model(meta):
    m = meta["model"]
    kw = dict(
        in_ch=3,
        embed_dim=m.get("embed_dim", 64),
        depths=tuple(m.get("depths", (2, 2, 2, 2))),
        num_heads=tuple(m.get("num_heads", (4, 4, 4, 4))),
        window_size=m.get("window_size", 8),
        mlp_ratio=m.get("mlp_ratio", 2),
        activation=m.get("activation", "gelu"),
        head_dim=m.get("head_dim", 16),
        attention_type=m.get("attention_type", "swin"),
        use_gcm=bool(m.get("gcm", False)),
        use_csn=bool(m.get("csn", False)),
        use_color_pre=bool(m.get("color_pre", False)),
        use_color_mlp=bool(m.get("color_mlp", False)),
        use_pcp=bool(m.get("pcp", False)),
        use_freqmod=bool(m.get("freqmod", False)),
        use_channel_curve=bool(m.get("channel_curve", False)),
        use_dual_branch=bool(m.get("dual_branch", False)),
        use_simple_gate=bool(m.get("simple_gate", False)),
        use_sca=bool(m.get("sca", False)),
        use_gdfn=bool(m.get("gdfn", False)),
        use_fpro=bool(m.get("fpro", False)),
    )
    model = RestoreNet(**kw).to(DEVICE)
    return model


def load_model(meta, ckpt_path):
    model = build_model(meta)
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=True)
    model.eval()
    return model


@torch.no_grad()
def infer(model, img_t):
    ws = model.window_size
    x, h, w = pad_to_window(img_t.unsqueeze(0).to(DEVICE), ws)
    with AMP:
        y = model(x)
    return y[:, :, :h, :w].float().clamp(0, 1)


def eval_pair(model, degraded_path, clean_path, want_lpips=True):
    d_np = load_rgb(degraded_path)
    c_np = load_rgb(clean_path)
    d_t, c_t = to_tensor(d_np), to_tensor(c_np)
    p = infer(model, d_t)[0]
    c_t = c_t.to(DEVICE)
    out = {"psnr_rgb": psnr(p, c_t), "ssim_rgb": ssim(p, c_t)}
    if want_lpips:
        with torch.no_grad():
            out["lpips_vgg"] = float(lpips_fn()(p.unsqueeze(0) * 2 - 1,
                                                c_t.unsqueeze(0) * 2 - 1).item())
    return out


def cmd_support(args):
    rows = [json.loads(l) for l in open(args.manifest)]
    out = open(args.out, "a")
    for idx, row in enumerate(rows):
        if args.gpu_id is not None and idx % args.num_gpus != args.gpu_id:
            continue
        if row["status"] != "ok":
            continue
        for ck in row["checkpoints"]:
            model = load_model(row["exp_meta"], ROOT / ck["path"])
            per = []
            for pair in row["support_pairs"]:
                m = eval_pair(model, ROOT / pair["lq"], ROOT / pair["clean"], want_lpips=False)
                m.update({"image_id": pair["image_id"], "lq_sha256": pair["lq_sha256"],
                          "clean_sha256": pair["clean_sha256"]})
                per.append(m)
            psnrs = [x["psnr_rgb"] for x in per]
            ssims = [x["ssim_rgb"] for x in per]
            rec = {
                "scenario_id": row["scenario_id"], "round": row["round"],
                "checkpoint": ck["path"], "checkpoint_sha256": ck["sha256"],
                "checkpoint_step": ck["step"], "params_path": row["params_path"],
                "params_sha256": row["params_sha256"],
                "support_psnr_mean": float(np.mean(psnrs)),
                "support_psnr_std": float(np.std(psnrs)),
                "support_psnr_min": float(np.min(psnrs)),
                "support_ssim_mean": float(np.mean(ssims)),
                "support": per,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[support] {row['scenario_id']} {row['round']} step{ck['step']} "
                  f"PSNR={rec['support_psnr_mean']:.3f}", flush=True)
            del model
            torch.cuda.empty_cache()
    out.close()


def stable_seed(*parts):
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:16], 16) % (2 ** 32)


def make_lq(clean_np, bid, image_id, pipeline, seed_salt):
    seed = stable_seed(bid, image_id, "test-lq-v1", seed_salt)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    img = clean_np.copy()
    for step in pipeline:
        img = add_distortion(img, severity=int(step["severity"]),
                             distortion_name=step["function"])
    return img


def cmd_test(args):
    rows = [json.loads(l) for l in open(args.selection)]
    if args.gpu_id is not None:
        rows = [r for i, r in enumerate(rows) if i % args.num_gpus == args.gpu_id]
    datasets = {
        "DIV2K": sorted(glob.glob(str(ROOT / "datasets/DIV2K/DIV2K_valid_HR/*.png"))),
        "LSDIR": sorted(glob.glob(str(ROOT / "datasets/LSDIR/val1/HR/val/*.png"))),
    }
    out = open(args.out, "a")
    for row in rows:
        bid = row["scenario_id"]
        gt_path = ROOT / "exp41" / "degradation_gt" / f"{bid}_params.json"
        if not gt_path.exists():
            raise FileNotFoundError(gt_path)
        pipeline = json.load(open(gt_path))["pipeline"]
        model = load_model(row["exp_meta"], ROOT / row["checkpoint"])
        for ds_name in args.datasets.split(","):
            for img_path in datasets[ds_name]:
                image_id = Path(img_path).stem
                clean_np = load_rgb(img_path)
                lq_np = make_lq(clean_np, bid, f"{ds_name}/{image_id}", pipeline, args.seed_salt)
                d_t, c_t = to_tensor(lq_np), to_tensor(clean_np)
                p = infer(model, d_t)[0]
                c_t = c_t.to(DEVICE)
                rec = {
                    "run_id": row["run_id"], "scenario_id": bid,
                    "controller": row.get("controller"), "round": row.get("round"),
                    "checkpoint": row["checkpoint"], "checkpoint_sha256": row["checkpoint_sha256"],
                    "checkpoint_step": row.get("checkpoint_step"),
                    "dataset": ds_name, "image_id": image_id,
                    "image_hash": sha256_file(img_path),
                    "lq_sha256": sha256_bytes(lq_np.tobytes()),
                    "metric_version": ("psnr_rgb_v1|ssim_gauss11x11_s1.5|lpips_vgg"
                                       if not args.no_lpips else "psnr_rgb_v1|ssim_gauss11x11_s1.5"),
                    "psnr_rgb": psnr(p, c_t),
                }
                if not args.psnr_only:
                    rec["ssim_rgb"] = ssim(p, c_t)
                if not args.psnr_only and not args.no_lpips:
                    rec["lpips_vgg"] = float(lpips_fn()(p.unsqueeze(0) * 2 - 1,
                                                        c_t.unsqueeze(0) * 2 - 1).item())
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
            print(f"[test] {bid} {ds_name} {row['checkpoint_step']} done", flush=True)
        del model
        torch.cuda.empty_cache()
    out.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("support")
    s.add_argument("--manifest", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--gpu-id", type=int, default=None)
    s.add_argument("--num-gpus", type=int, default=1)
    s.set_defaults(fn=cmd_support)

    t = sub.add_parser("test")
    t.add_argument("--selection", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--datasets", default="DIV2K,LSDIR")
    t.add_argument("--seed-salt", default="v1")
    t.add_argument("--no-lpips", action="store_true",
                   help="skip LPIPS (much faster; PSNR/SSIM only)")
    t.add_argument("--psnr-only", action="store_true",
                   help="compute only PSNR (fastest)")
    t.add_argument("--gpu-id", type=int, default=None)
    t.add_argument("--num-gpus", type=int, default=1)
    t.set_defaults(fn=cmd_test)

    args = ap.parse_args()
    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)
    args.fn(args)


if __name__ == "__main__":
    main()
