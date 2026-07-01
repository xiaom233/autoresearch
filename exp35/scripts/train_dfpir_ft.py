#!/usr/bin/env python3
"""DFPIR (31M) fine-tune on specific degradation, fixed time budget."""

import os, sys, time, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.join(THIS_DIR, "..", "..")
sys.path.insert(0, PROJ_DIR)
sys.path.insert(0, os.path.join(PROJ_DIR, "resource",
    "Degradation-Aware-Feature-Perturbation-for-All-in-One-Image-Restoration-Modified-Version-"))

from net.model import ChannelShuffle_skip_textguaid
from prepare import make_dataloader_restoration
from train import ValDataset, evaluate_all

CKPT_PATH = os.path.join(PROJ_DIR, "resource",
    "Degradation-Aware-Feature-Perturbation-for-All-in-One-Image-Restoration-Modified-Version-",
    "dfpir_blind/checkpoints/dfpir_blind_step301920.pt")

VAL_DIRS = [
    "datasets/Set5/GTmod4", "datasets/Set14/GTmod4",
    "datasets/B100/GTmod4", "datasets/Urban100/GTmod4",
    "datasets/Manga109/GTmod4", "datasets/DIV2K/DIV2K_valid_HR",
]


class DFPIRBlind(nn.Module):
    """DFPIR with learned text embedding (no CLIP needed)."""
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.text_embed = nn.Parameter(torch.randn(1, 512) * 0.02)

    def _pad16(self, x):
        _, _, H, W = x.shape
        ph, pw = (16 - H % 16) % 16, (16 - W % 16) % 16
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        return x

    def forward(self, x):
        _, _, H, W = x.shape
        x = self._pad16(x)
        b = x.shape[0]
        out = self.base(x, self.text_embed.expand(b, -1))
        return out[:, :, :H, :W]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True, help="degradation params.json")
    parser.add_argument("--time-budget", type=float, default=5400,
                        help="training time budget in seconds (default 1.5h)")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size (DFPIR 31M: 4/GPU due to VRAM)")
    parser.add_argument("--accum-steps", type=int, default=4, help="gradient accumulation steps (effective_batch = batch_size × accum_steps = 16)")
    parser.add_argument("--lr", type=float, default=2e-4, help="LR (DFPIR original: 2e-4)")
    parser.add_argument("--warmup-steps", type=int, default=500, help="warmup optimizer steps (not iterations)")
    parser.add_argument("--ckpt-prefix", required=True, help="output checkpoint prefix")
    parser.add_argument("--val-interval", type=int, default=900,
                        help="validate every N seconds (default 15min)")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    amp = torch.amp.autocast("cuda", dtype=torch.bfloat16)

    # Build model
    base = ChannelShuffle_skip_textguaid(
        dim=48, num_blocks=[4, 6, 6, 8], heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias",
    )
    model = DFPIRBlind(base).to(device)

    # Load pretrained blind checkpoint
    ck = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ck["model"])
    print(f"Loaded DFPIR blind checkpoint ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    # Dataloader (same as RestoreNet training)
    train_loader = make_dataloader_restoration(
        params_path=args.params,
        shards_url="datasets/DIV2K/DIV2K_train_HR_wds/train-*.tar",
        batch_size=args.batch_size,
        num_workers=4, endless=True,
    )

    # Validation sets
    val_sets = [(d.split("/")[-2], ValDataset([d], count=0, params_path=args.params))
                for d in VAL_DIRS if os.path.isdir(d)]

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(args.time_budget / 0.8))  # ~steps estimate
    criterion = nn.L1Loss()

    os.makedirs(os.path.dirname(args.ckpt_prefix), exist_ok=True)

    print(f"Time budget: {args.time_budget:.0f}s ({args.time_budget/3600:.1f}h)")
    eff_batch = args.batch_size * args.accum_steps
    print(f"Batch: {args.batch_size} × accum {args.accum_steps} = effective {eff_batch} (RestoreNet: 16)")
    print(f"Training...")

    start_time = time.time()
    last_val_time = start_time
    step = 0
    best_psnr = 0.0
    best_ckpt = None

    while True:
        for x_deg, x_clean, _epoch in train_loader:
            elapsed = time.time() - start_time
            if elapsed >= args.time_budget:
                break

            x_deg = x_deg.to(device)
            x_clean = x_clean.to(device)

            with amp:
                pred = model(x_deg)
                loss = criterion(pred, x_clean)
            loss = loss / args.accum_steps
            loss.backward()

            # Gradient accumulation: step every accum_steps iterations
            if (step + 1) % args.accum_steps == 0:
                # Warmup (counted in optimizer steps, not iterations)
                opt_step = step // args.accum_steps
                if opt_step <= args.warmup_steps:
                    warmup_lr = args.lr * max(opt_step, 1) / max(args.warmup_steps, 1)
                    for pg in optimizer.param_groups:
                        pg['lr'] = warmup_lr

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1
            if step % 100 == 0:
                dt = elapsed / max(step, 1) * 1000
                opt_step = step // args.accum_steps
                print(f"iter {step:05d} (opt {opt_step:05d}) | loss: {loss.item()*args.accum_steps:.6f} | "
                      f"lr: {optimizer.param_groups[0]['lr']:.2e} | dt: {dt:.0f}ms | elapsed: {elapsed/60:.1f}min",
                      flush=True)

            # Periodic validation
            if time.time() - last_val_time >= args.val_interval:
                model.eval()
                _, ov = evaluate_all(model, val_sets, device, amp)
                model.train()
                psnr = ov["psnr_rgb"]
                last_val_time = time.time()
                print(f"[VAL iter {step:05d}] PSNR={psnr:.2f} dB (elapsed {elapsed/60:.1f}min)")

                if psnr > best_psnr:
                    best_psnr = psnr
                    best_ckpt = os.path.join(
                        os.path.dirname(args.ckpt_prefix),
                        f"{os.path.basename(args.ckpt_prefix)}_iter{step:05d}_psnr{psnr:.2f}.pt"
                    )
                    torch.save({"model": model.state_dict(), "iter": step, "psnr": psnr}, best_ckpt)

        if elapsed >= args.time_budget:
            break

    total_time = time.time() - start_time
    opt_steps = step // args.accum_steps
    print(f"\nDone: {step} iters ({opt_steps} optimizer steps) in {total_time/60:.1f}min, "
          f"best PSNR={best_psnr:.2f}")

    # Always save final checkpoint if no best was saved
    if best_ckpt is None:
        final_ckpt = os.path.join(
            os.path.dirname(args.ckpt_prefix),
            f"{os.path.basename(args.ckpt_prefix)}_iter{step:05d}_final.pt"
        )
        torch.save({"model": model.state_dict(), "iter": step}, final_ckpt)
        best_ckpt = final_ckpt
        print(f"Final ckpt: {best_ckpt}")
    else:
        print(f"Best ckpt: {best_ckpt}")


if __name__ == "__main__":
    main()
