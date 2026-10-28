#!/usr/bin/env python3
"""Merge support metrics and select round/checkpoint by support PSNR.

Pre-registered rule:
  - per round: highest mean support PSNR; tie -> higher step
  - per scenario: highest selected-checkpoint support PSNR;
    tie (<=0.05 dB) -> later round
No test metric is used.
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A0 = ROOT / "rebuttal_runs" / "A0"


def main():
    manifest = {}
    for line in open(A0 / "exp41_selection_manifest.jsonl"):
        row = json.loads(line)
        manifest[(row["scenario_id"], row["round"])] = row

    metrics = defaultdict(list)
    for p in glob.glob(str(A0 / "support_metrics" / "support_gpu*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            metrics[(r["scenario_id"], r["round"])].append(r)

    scenarios = sorted({k[0] for k in manifest})
    out = []
    last_step_rows = []
    for sid in scenarios:
        rounds = {}
        for r in ("R0", "R1", "R2"):
            ms = metrics.get((sid, r), [])
            if not ms:
                rounds[r] = None
                continue
            best = sorted(ms, key=lambda x: (x["support_psnr_mean"], x["checkpoint_step"]))[-1]
            last = sorted(ms, key=lambda x: x["checkpoint_step"])[-1]
            mrow = manifest[(sid, r)]
            rounds[r] = {
                "selected": {
                    "round": r, "checkpoint": best["checkpoint"],
                    "checkpoint_sha256": best["checkpoint_sha256"],
                    "step": best["checkpoint_step"],
                    "support_psnr_mean": best["support_psnr_mean"],
                    "support_ssim_mean": best["support_ssim_mean"],
                    "params_path": mrow["params_path"],
                    "params_sha256": mrow["params_sha256"],
                    "exp_meta": mrow["exp_meta"],
                    "training_seconds": mrow["training_seconds"],
                },
                "last_step": {
                    "round": r, "checkpoint": last["checkpoint"],
                    "step": last["checkpoint_step"],
                    "support_psnr_mean": last["support_psnr_mean"],
                    "params_path": mrow["params_path"],
                    "exp_meta": mrow["exp_meta"],
                },
                "n_checkpoints": len(ms),
            }
        avail = [(r, rounds[r]["selected"]["support_psnr_mean"]) for r in ("R0", "R1", "R2")
                 if rounds[r] is not None]
        top = max(v for _, v in avail)
        cands = [r for r, v in avail if v >= top - 0.05]
        chosen = cands[-1]  # later round on ties
        sel = rounds[chosen]["selected"]
        t_s = sum((manifest[(sid, r)]["training_seconds"] or 0) for r in ("R0", "R1", "R2"))
        out.append({
            "scenario_id": sid,
            "selected_round": chosen,
            "selected_checkpoint": sel["checkpoint"],
            "selected_checkpoint_sha256": sel["checkpoint_sha256"],
            "selected_step": sel["step"],
            "support_psnr_mean": sel["support_psnr_mean"],
            "support_ssim_mean": sel["support_ssim_mean"],
            "params_path": sel["params_path"],
            "params_sha256": sel["params_sha256"],
            "exp_meta": sel["exp_meta"],
            "per_round": rounds,
            "T_s_gpu_seconds": t_s,
            "T_s_gpu_hours": round(t_s / 3600, 3),
            "n_unique_trainings": sum(1 for r in ("R0", "R1", "R2")
                                      if manifest[(sid, r)]["training_seconds"] is not None),
        })
        last_step_rows.append({
            "scenario_id": sid,
            "last_round": "R2" if rounds["R2"] else ("R1" if rounds["R1"] else "R0"),
            "last_step_support_psnr": (rounds["R2"] or rounds["R1"] or rounds["R0"])["last_step"]["support_psnr_mean"],
            "selected_support_psnr": sel["support_psnr_mean"],
        })

    with open(A0 / "support_selection.jsonl", "w") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "n_scenarios": len(out),
        "selected_round_counts": {r: sum(1 for x in out if x["selected_round"] == r)
                                  for r in ("R0", "R1", "R2")},
        "support_psnr_mean_by_selected_round": {
            r: round(sum(x["support_psnr_mean"] for x in out if x["selected_round"] == r) /
                     max(1, sum(1 for x in out if x["selected_round"] == r)), 4)
            for r in ("R0", "R1", "R2")
        },
        "mean_selected_support_psnr": round(sum(x["support_psnr_mean"] for x in out) / len(out), 4),
        "mean_last_step_support_psnr": round(sum(x["last_step_support_psnr"] for x in last_step_rows) / len(last_step_rows), 4),
        "selection_basis": "support PSNR only; no test metric",
        "total_T_s_gpu_hours": round(sum(x["T_s_gpu_hours"] for x in out), 2),
        "mean_T_s_gpu_hours": round(sum(x["T_s_gpu_hours"] for x in out) / len(out), 3),
        "predicted_step_distribution": {},
    }
    for x in out:
        n = len(x["params_path"] and json.load(open(ROOT / x["params_path"])).get("pipeline", []))
        summary["predicted_step_distribution"][str(n)] = summary["predicted_step_distribution"].get(str(n), 0) + 1
    (A0 / "support_selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
