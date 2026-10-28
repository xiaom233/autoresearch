#!/usr/bin/env python3
"""Evaluate DFPIR checkpoints on materialized fixed-LQ images (DIV2K)."""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rebuttal_runs/scripts"))
from eval_dfpir import load_dfpir  # noqa: E402
from eval_checkpoint import to_tensor, psnr, ssim  # noqa: E402
from train_dfpir_budget import tiled_infer  # noqa: E402

FIX = ROOT / "rebuttal_runs/E1E2_fixed_lq"
DEFAULT = "blind_0003,blind_0005,blind_0017,blind_0018,blind_0020,blind_0022,blind_0023"


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
        model = load_dfpir(ROOT / row["checkpoint"])
        model.eval()
        for p in pairs:
            clean = load_rgb(ROOT / p["clean_path"])
            lq = load_rgb(ROOT / p["lq_path"])
            d_t = to_tensor(lq).unsqueeze(0).to(device)
            c_t = to_tensor(clean).to(device)
            pred = tiled_infer(model, d_t)[0]
            rec = {
                "run_id": row["run_id"], "scenario_id": sid,
                "controller": row.get("controller"), "round": row.get("round"),
                "checkpoint": row["checkpoint"],
                "checkpoint_sha256": row.get("checkpoint_sha256"),
                "checkpoint_step": row.get("checkpoint_step"),
                "dataset": p["dataset"], "image_id": p["image_id"],
                "image_hash": p["image_sha256"], "lq_sha256": p["lq_sha256"],
                "metric_version": "psnr_rgb_v1|ssim_gauss11x11_s1.5|fixed_lq_v1|dfpir_tile512_ov64",
                "psnr_rgb": psnr(pred, c_t), "ssim_rgb": ssim(pred, c_t),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
        print(f"[dfpir-fixed] {sid} opt{row.get('checkpoint_step')} done", flush=True)
        del model
        torch.cuda.empty_cache()
    out.close()


if __name__ == "__main__":
    main()
