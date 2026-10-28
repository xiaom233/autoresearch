#!/usr/bin/env python3
"""Select E1 rule-controller round/checkpoint by support PSNR and emit test selection."""
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E1 = ROOT / "rebuttal_runs" / "E1"


def main():
    manifest = {}
    for line in open(E1 / "manifest.jsonl"):
        r = json.loads(line)
        manifest[(r["scenario_id"], r["round"])] = r

    metrics = defaultdict(list)
    for p in glob.glob(str(E1 / "support_metrics" / "support_gpu*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            metrics[(r["scenario_id"], r["round"])].append(r)

    rows = []
    for sid in sorted({k[0] for k in manifest}):
        avail = {}
        rounds = sorted({k[1] for k in manifest if k[0] == sid})
        for rnd in rounds:
            ms = [x for x in metrics.get((sid, rnd), [])
                  if math.isfinite(x.get("support_psnr_mean", float("nan")))]
            if not ms:
                continue
            best = sorted(ms, key=lambda x: (x["support_psnr_mean"], x["checkpoint_step"]))[-1]
            avail[rnd] = best
        if not avail:
            continue
        top = max(v["support_psnr_mean"] for v in avail.values())
        cands = [r for r, v in avail.items() if v["support_psnr_mean"] >= top - 0.05]

        def rkey(r):
            fixed = 1 if r.endswith("_fixed") else 0
            num = int(r[1]) if len(r) > 1 and r[1].isdigit() else 0
            return (fixed, num, avail[r]["support_psnr_mean"])
        chosen = max(cands, key=rkey)
        m = manifest[(sid, chosen)]
        b = avail[chosen]
        rows.append({
            "scenario_id": sid, "selected_round": chosen,
            "checkpoint": b["checkpoint"], "checkpoint_sha256": b["checkpoint_sha256"],
            "checkpoint_step": b["checkpoint_step"],
            "support_psnr_mean": b["support_psnr_mean"],
            "params_path": m["params_path"], "params_sha256": m["params_sha256"],
            "exp_meta": m["exp_meta"],
            "per_round_support": {r: v["support_psnr_mean"] for r, v in avail.items()},
        })
    with open(E1 / "selection.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(E1 / "test_selection.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({
                "run_id": f"E1_rule_{r['scenario_id']}_{r['selected_round']}",
                "scenario_id": r["scenario_id"], "controller": "rule",
                "round": r["selected_round"], "checkpoint": r["checkpoint"],
                "checkpoint_sha256": r["checkpoint_sha256"],
                "checkpoint_step": r["checkpoint_step"], "exp_meta": r["exp_meta"],
            }, ensure_ascii=False) + "\n")
    all_rounds = sorted({x["selected_round"] for x in rows})
    summary = {
        "n_scenarios": len(rows),
        "selected_round_counts": {r: sum(1 for x in rows if x["selected_round"] == r)
                                  for r in all_rounds},
        "mean_support_psnr": sum(x["support_psnr_mean"] for x in rows) / max(1, len(rows)),
        "selection_basis": "support PSNR only; fixed-mapping rounds preferred on ties",
    }
    (E1 / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
