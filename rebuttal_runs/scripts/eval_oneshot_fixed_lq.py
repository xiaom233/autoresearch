#!/usr/bin/env python3
"""Evaluate one-shot checkpoints on the materialized fixed-LQ images.

Only the 6 fixed-LQ scenarios in the 23-scenario common set are needed
(blind_0020 is excluded because the rule R0 training was NaN).
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rebuttal_runs/scripts"))
from eval_checkpoint import load_model, infer, psnr, ssim, to_tensor  # noqa: E402

FIX = ROOT / "rebuttal_runs/E1E2_fixed_lq"
DEFAULT = "blind_0003,blind_0005,blind_0017,blind_0018,blind_0022,blind_0023"


def load_rgb(path):
    import numpy as np
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenarios", default=DEFAULT)
    ap.add_argument("--datasets", default="DIV2K")
    ap.add_argument("--gpu-id", type=int, required=True)
    ap.add_argument("--num-gpus", type=int, required=True)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda")

    wanted = set(args.scenarios.split(","))
    rows = [json.loads(l) for l in open(args.selection) if l.strip()]
    rows = [r for i, r in enumerate(rows)
            if r["scenario_id"] in wanted and i % args.num_gpus == args.gpu_id]
    out = open(args.out, "a")
    for row in rows:
        sid = row["scenario_id"]
        index = json.load(open(FIX / sid / "index.json"))
        wanted_ds = set(args.datasets.split(","))
        pairs = [p for p in index["pairs"] if p["dataset"] in wanted_ds]
        model = load_model(row["exp_meta"], ROOT / row["checkpoint"])
        model.eval()
        for p in pairs:
            clean_np = load_rgb(ROOT / p["clean_path"])
            lq_np = load_rgb(ROOT / p["lq_path"])
            d_t = to_tensor(lq_np)
            c_t = to_tensor(clean_np).to(device)
            pred = infer(model, d_t)[0]
            rec = {
                "run_id": row["run_id"], "scenario_id": sid,
                "controller": row.get("controller"), "round": row.get("round"),
                "checkpoint": row["checkpoint"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "checkpoint_step": row.get("checkpoint_step"),
                "dataset": p["dataset"], "image_id": p["image_id"],
                "image_hash": p["image_sha256"], "lq_sha256": p["lq_sha256"],
                "metric_version": "psnr_rgb_v1|ssim_gauss11x11_s1.5|fixed_lq_v1",
                "psnr_rgb": psnr(pred, c_t), "ssim_rgb": ssim(pred, c_t),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
        print(f"[fixed] {sid} {row['controller']} {row['checkpoint_step']} done", flush=True)
        del model
        torch.cuda.empty_cache()
    out.close()


if __name__ == "__main__":
    main()
