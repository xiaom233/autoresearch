#!/usr/bin/env python3
"""Build the DFPIR test-evaluation selection from E2 summaries."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E2 = ROOT / "rebuttal_runs" / "E2"
BLIND = ROOT / "resource" / "Degradation-Aware-Feature-Perturbation-for-All-in-One-Image-Restoration-Modified-Version-" / "dfpir_blind/checkpoints/dfpir_blind_step301920.pt"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-snapshots", action="store_true",
                    help="also evaluate T/3 and 2T/3 snapshots")
    args = ap.parse_args()
    rows = []
    blind_rel = str(BLIND.relative_to(ROOT))
    for sid in [f"blind_{i:04d}" for i in range(1, 25)]:
        summ = E2 / sid / f"{sid}_summary.json"
        if not summ.exists():
            continue
        d = json.load(open(summ))
        # best, last, and the three budget snapshots (dedupe by path)
        paths = []
        for key in ("best_checkpoint", "last_checkpoint"):
            p = d.get(key)
            if p and p not in paths:
                paths.append(p)
        if args.include_snapshots:
            for p in sorted((E2 / sid).glob("*.pt")):
                rel = str(p.relative_to(ROOT))
                if rel not in paths:
                    paths.append(rel)
        for p in paths:
            if p == d.get("best_checkpoint"):
                ctrl = "DFPIR-ft-best"
            elif p == d.get("last_checkpoint"):
                ctrl = "DFPIR-ft-last"
            elif "T1of3" in Path(p).stem:
                ctrl = "DFPIR-ft-T1"
            elif "T2of3" in Path(p).stem:
                ctrl = "DFPIR-ft-T2"
            elif "T3of3" in Path(p).stem:
                ctrl = "DFPIR-ft-T3"
            else:
                ctrl = "DFPIR-ft-other"
            rows.append({
                "run_id": f"E2_{sid}_{Path(p).stem}",
                "scenario_id": sid, "controller": ctrl,
                "round": "E2", "checkpoint": p,
                "checkpoint_sha256": sha(ROOT / p),
                "checkpoint_step": Path(p).stem,
            })
        rows.append({
            "run_id": f"E2_{sid}_original_blind",
            "scenario_id": sid, "controller": "DFPIR-original",
            "round": "zero-shot", "checkpoint": blind_rel,
            "checkpoint_sha256": sha(BLIND),
            "checkpoint_step": 301920,
        })
    out = E2 / "test_selection.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
