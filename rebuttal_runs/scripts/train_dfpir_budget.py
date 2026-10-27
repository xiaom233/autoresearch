#!/usr/bin/env python3
"""E2: DFPIR fine-tune with the AutoIR three-round GPU-time budget.

Fixes vs scripts/train_dfpir_ft.py:
  - validation uses the 5 support pairs (not DIV2K/LSDIR test)
  - pure training time is measured separately from validation
  - snapshots at 1/3, 2/3, T of the budget, each with optimizer state
  - best and last are always saved
  - micro-iterations, optimizer steps, effective samples and LR are logged
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def rel(p):
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)


sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "resource" /
                   "Degradation-Aware-Feature-Perturbation-for-All-in-One-Image-Restoration-Modified-Version-"))

from net.model import ChannelShuffle_skip_textguaid  # noqa: E402
from prepare import make_dataloader_restoration  # noqa: E402

CKPT_PATH = ROOT / "resource" / "Degradation-Aware-Feature-Perturbation-for-All-in-One-Image-Restoration-Modified-Version-" / "dfpir_blind/checkpoints/dfpir_blind_step301920.pt"


class DFPIRBlind(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.text_embed = nn.Parameter(torch.randn(1, 512) * 0.02)

    def _pad16(self, x):
        _, _, h, w = x.shape
        ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        return x

    def forward(self, x):
        _, _, h, w = x.shape
        x = self._pad16(x)
        out = self.base(x, self.text_embed.expand(x.shape[0], -1))
        return out[:, :, :h, :w]


def psnr(pred, target):
    mse = F.mse_loss(pred.float(), target.float())
    return 100.0 if mse.item() == 0 else float(10 * math.log10(1.0 / mse.item()))


@torch.no_grad()
def tiled_infer(model, img, tile=512, overlap=64):
    _, _, h, w = img.shape
    out = torch.zeros_like(img)
    wsum = torch.zeros(1, 1, h, w, device=img.device)
    step = tile - overlap
    for y in range(0, h, step):
        for x in range(0, w, step):
            y2, x2 = min(y + tile, h), min(x + tile, w)
            y1, x1 = max(0, y2 - tile), max(0, x2 - tile)
            patch = img[:, :, y1:y2, x1:x2]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(patch)
            pred = pred.float()
            ph, pw = y2 - y1, x2 - x1
            out[:, :, y1:y2, x1:x2] += pred
            wsum[:, :, y1:y2, x1:x2] += 1
            if x2 >= w:
                break
        if y2 >= h:
            break
    return (out / wsum).clamp(0, 1)


def load_support(support_dir):
    pairs = []
    d = Path(support_dir)
    for i in range(5):
        c = d / f"clean_{i}.png"
        lq = d / f"degraded_{i}.png"
        if not (c.exists() and lq.exists()):
            raise FileNotFoundError(f"missing support pair {i} in {support_dir}")
        c_np = np.array(Image.open(c).convert("RGB"), dtype=np.uint8)
        l_np = np.array(Image.open(lq).convert("RGB"), dtype=np.uint8)
        pairs.append((torch.from_numpy(l_np).permute(2, 0, 1).float().div(255).cuda(),
                      torch.from_numpy(c_np).permute(2, 0, 1).float().div(255).cuda()))
    return pairs


@torch.no_grad()
def support_eval(model, pairs):
    model.eval()
    vals = []
    for lq, clean in pairs:
        pred = tiled_infer(model, lq.unsqueeze(0))[0]
        vals.append(psnr(pred, clean))
    model.train()
    return float(np.mean(vals)), vals


def save_ckpt(path, model, optimizer, meta):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), **meta}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--support-dir", required=True)
    ap.add_argument("--budget-seconds", type=float, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--accum-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-optimizer-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--scratch", action="store_true", help="random init; do not load blinded checkpoint")
    args = ap.parse_args()

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    amp = torch.amp.autocast("cuda", dtype=torch.bfloat16)

    base = ChannelShuffle_skip_textguaid(
        dim=48, num_blocks=[4, 6, 6, 8], heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias")
    model = DFPIRBlind(base).to(device)
    if not args.scratch:
        ck = torch.load(CKPT_PATH, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
    model.train()

    loader = make_dataloader_restoration(
        params_path=str(ROOT / args.params),
        shards_url=str(ROOT / "datasets/DIV2K/DIV2K_train_HR_wds/train-*.tar"),
        batch_size=args.batch_size, num_workers=args.num_workers, endless=True)
    pairs = load_support(args.support_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    meta = {
        "scenario": args.scenario, "params": args.params,
        "budget_seconds": args.budget_seconds, "micro_batch": args.batch_size,
        "accum_steps": args.accum_steps, "effective_batch": args.batch_size * args.accum_steps,
        "lr": args.lr, "seed": args.seed, "init": "scratch" if args.scratch else "blinded",
    }
    print(json.dumps(meta, ensure_ascii=False), flush=True)

    micro = 0
    opt_step = 0
    pure_train = 0.0
    validation_time = 0.0
    eff_samples = 0
    best = {"psnr": -1, "opt_step": None, "path": None}
    last_path = None
    snapshots = {1.0 / 3, 2.0 / 3, 1.0}
    done_snapshots = set()

    it = iter(loader)
    while pure_train < args.budget_seconds:
        # Time from before the DataLoader fetch, matching train.py's
        # training_seconds (fetch + forward + backward + optimizer).
        t0 = time.time()
        x_deg, x_clean, _epoch = next(it)
        x_deg = x_deg.to(device, non_blocking=True)
        x_clean = x_clean.to(device, non_blocking=True)
        with amp:
            pred = model(x_deg)
            loss = criterion(pred, x_clean)
        (loss / args.accum_steps).backward()
        micro += 1

        if micro % args.accum_steps == 0:
            if opt_step < args.warmup_optimizer_steps:
                lr = args.lr * max(opt_step + 1, 1) / max(args.warmup_optimizer_steps, 1)
            else:
                progress = min(pure_train / args.budget_seconds, 1.0)
                lr = 0.5 * args.lr * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1
            eff_samples += args.batch_size * args.accum_steps

        torch.cuda.synchronize()
        pure_train += time.time() - t0

        if micro % 100 == 0:
            print(f"micro {micro:06d} opt {opt_step:06d} loss {loss.item():.6f} "
                  f"lr {optimizer.param_groups[0]['lr']:.2e} "
                  f"pure_train {pure_train:.0f}/{args.budget_seconds:.0f}s", flush=True)

        ratio = pure_train / args.budget_seconds
        for frac in sorted(snapshots - done_snapshots):
            if ratio >= frac:
                tv0 = time.time()
                p, per = support_eval(model, pairs)
                validation_time += time.time() - tv0
                step_tag = f"T{int(frac*3)}of3"
                path = out / f"{args.scenario}_dfpir_{step_tag}_opt{opt_step}.pt"
                save_ckpt(path, model, optimizer, {
                    "scenario": args.scenario, "pure_train_seconds": pure_train,
                    "opt_step": opt_step, "micro_iter": micro,
                    "support_psnr": p, "snapshot_fraction": frac,
                    "effective_samples": eff_samples, "lr": optimizer.param_groups[0]["lr"],
                })
                done_snapshots.add(frac)
                print(f"[snapshot {step_tag}] support_psnr={p:.3f} opt={opt_step} "
                      f"path={rel(path)}", flush=True)
                if p > best["psnr"]:
                    best = {"psnr": p, "opt_step": opt_step, "path": rel(path)}
                last_path = rel(path)
                model.train()

    # Always save last, even if the final snapshot was skipped.
    last_path = rel(out / f"{args.scenario}_dfpir_last_opt{opt_step}.pt")
    save_ckpt(ROOT / last_path, model, optimizer, {
        "scenario": args.scenario, "pure_train_seconds": pure_train,
        "opt_step": opt_step, "micro_iter": micro, "effective_samples": eff_samples,
        "lr": optimizer.param_groups[0]["lr"], "kind": "last",
    })
    p_last, per_last = support_eval(model, pairs)
    if p_last > best["psnr"]:
        best = {"psnr": p_last, "opt_step": opt_step, "path": last_path}
    summary = {
        **meta, "micro_iterations": micro, "optimizer_steps": opt_step,
        "effective_samples": eff_samples, "pure_train_seconds": round(pure_train, 1),
        "validation_seconds": round(validation_time, 1),
        "final_lr": optimizer.param_groups[0]["lr"],
        "last_checkpoint": last_path, "last_support_psnr": p_last,
        "best_checkpoint": best["path"], "best_support_psnr": best["psnr"],
        "best_opt_step": best["opt_step"],
        "snapshots": sorted(f"T{int(f*3)}of3" for f in done_snapshots),
        "support_per_image_psnr": per_last,
    }
    (out / f"{args.scenario}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
