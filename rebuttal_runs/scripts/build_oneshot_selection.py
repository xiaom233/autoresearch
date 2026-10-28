#!/usr/bin/env python3
"""Build one-shot (no-reflection) selections for the E1 rule and AutoIR R0 arms.

Rule arm: best R0 checkpoint (R0_fixed preferred when the noise-mapping fix exists).
LLM arm: AutoIR's per-round R0 selection from A0/support_selection.jsonl.
No new training; both arms reuse existing checkpoints.
"""
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E1 = ROOT / "rebuttal_runs/E1"
A0 = ROOT / "rebuttal_runs/A0"
OUT = ROOT / "rebuttal_runs/E1_oneshot"
A0_OUT = ROOT / "rebuttal_runs/A0_oneshot"


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    A0_OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_metrics").mkdir(exist_ok=True)
    (A0_OUT / "test_metrics").mkdir(exist_ok=True)

    manifest = {(r["scenario_id"], r["round"]): r for r in load_jsonl(E1 / "manifest.jsonl")}
    metrics = defaultdict(list)
    for p in glob.glob(str(E1 / "support_metrics/support_gpu*.jsonl")):
        for r in load_jsonl(p):
            metrics[(r["scenario_id"], r["round"])].append(r)

    rule_rows = []
    for sid in sorted({k[0] for k in manifest}):
        rnd = "R0_fixed" if (sid, "R0_fixed") in manifest else "R0"
        rows = [r for r in metrics.get((sid, rnd), [])
                if math.isfinite(r.get("support_psnr_mean", float("nan")))]
        if not rows:
            print(f"[warn] no support metrics for {sid} {rnd}")
            continue
        best = sorted(rows, key=lambda x: (x["support_psnr_mean"], x["checkpoint_step"]))[-1]
        m = manifest[(sid, rnd)]
        rule_rows.append({
            "run_id": f"E1oneshot_{sid}_{rnd}",
            "scenario_id": sid,
            "controller": "rule-oneshot",
            "round": rnd,
            "checkpoint": best["checkpoint"],
            "checkpoint_sha256": best["checkpoint_sha256"],
            "checkpoint_step": best["checkpoint_step"],
            "params_path": m["params_path"],
            "params_sha256": m["params_sha256"],
            "support_psnr_mean": best["support_psnr_mean"],
            "exp_meta": m["exp_meta"],
        })
    (OUT / "selection.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rule_rows), encoding="utf-8")

    a0_rows = []
    for r in load_jsonl(A0 / "support_selection.jsonl"):
        sel = r.get("per_round", {}).get("R0", {}).get("selected")
        if not sel:
            print(f"[warn] no A0 R0 selection for {r['scenario_id']}")
            continue
        a0_rows.append({
            "run_id": f"A0oneshot_{r['scenario_id']}_R0",
            "scenario_id": r["scenario_id"],
            "controller": "AutoIR-oneshot",
            "round": "R0",
            "checkpoint": sel["checkpoint"],
            "checkpoint_sha256": sel["checkpoint_sha256"],
            "checkpoint_step": sel["step"],
            "params_path": sel["params_path"],
            "params_sha256": sel["params_sha256"],
            "support_psnr_mean": sel["support_psnr_mean"],
            "exp_meta": sel["exp_meta"],
        })
    (A0_OUT / "selection.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in a0_rows), encoding="utf-8")

    combined = rule_rows + a0_rows
    (OUT / "selection_combined.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in combined), encoding="utf-8")
    print(f"wrote {len(rule_rows)} rule-oneshot, {len(a0_rows)} AutoIR-oneshot, "
          f"{len(combined)} combined")


if __name__ == "__main__":
    main()
