#!/usr/bin/env python3
"""Re-evaluate E6-A full24 selections on the materialized fixed-LQ images.

Only the 7 nondeterministic scenarios (impulse/glass private RNG) need this:
their per-process LQ pixels differ, so the main pass used different LQ
realizations for rep1/rep2 and across best/last/final. This script evaluates
every selected checkpoint on one shared LQ realization from E1E2_fixed_lq.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rebuttal_runs/scripts"))
from eval_checkpoint import load_model, load_rgb, to_tensor, infer, psnr, ssim  # noqa: E402

FIX = ROOT / "rebuttal_runs/E1E2_fixed_lq"
SCENARIOS = {"blind_0003", "blind_0005", "blind_0017", "blind_0018",
             "blind_0020", "blind_0022", "blind_0023"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-id", type=int, required=True)
    ap.add_argument("--num-gpus", type=int, default=8)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu_id)

    rows = [json.loads(l) for l in open(args.selection)]
    rows = [r for r in rows if r["scenario_id"] in SCENARIOS]
    rows = [r for i, r in enumerate(rows) if i % args.num_gpus == args.gpu_id]
    out = open(args.out, "a")
    for r in rows:
        idx = json.load(open(FIX / r["scenario_id"] / "index.json"))
        model = load_model(r["exp_meta"], ROOT / r["checkpoint"])
        for p in idx["pairs"]:
            clean = load_rgb(ROOT / p["clean_path"])
            lq = load_rgb(ROOT / p["lq_path"])
            d_t, c_t = to_tensor(lq), to_tensor(clean)
            pred = infer(model, d_t)[0]
            c_t = c_t.to(pred.device)
            rec = {
                "run_id": r["run_id"], "scenario_id": r["scenario_id"],
                "dataset": p["dataset"], "image_id": p["image_id"],
                "checkpoint": r["checkpoint"],
                "checkpoint_sha256": r["checkpoint_sha256"],
                "checkpoint_step": r.get("checkpoint_step"),
                "selected_round": r.get("selected_round"),
                "lq_sha256": p["lq_sha256"],
                "metric_version": "psnr_rgb_v1|ssim_gauss11x11_s1.5|fixed_lq_E1E2",
                "psnr_rgb": psnr(pred, c_t), "ssim_rgb": ssim(pred, c_t),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        print(f"[fixed-lq] {r['run_id']} {r['checkpoint_step']} done", flush=True)
        del model
        torch.cuda.empty_cache()
    out.close()


if __name__ == "__main__":
    main()
