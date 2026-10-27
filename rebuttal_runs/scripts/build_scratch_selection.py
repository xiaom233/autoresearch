#!/usr/bin/env python3
"""Build DFPIR-scratch best/last selection rows from E2_scratch summaries."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "rebuttal_runs/E2_scratch"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    rows = []
    for d in sorted(BASE.glob("blind_*")):
        sf = d / f"{d.name}_summary.json"
        if not sf.exists():
            continue
        s = json.load(open(sf))
        for kind, ckkey, pskey in [("best", "best_checkpoint", "best_support_psnr"),
                                   ("last", "last_checkpoint", "last_support_psnr")]:
            ck = ROOT / s[ckkey]
            if not ck.exists():
                continue
            rows.append({
                "run_id": f"E2scratch_{d.name}_{kind}",
                "scenario_id": d.name,
                "controller": f"DFPIR-scratch-{kind}",
                "round": kind,
                "checkpoint": s[ckkey],
                "checkpoint_sha256": sha_file(ck),
                "checkpoint_step": s.get("best_opt_step" if kind == "best" else "optimizer_steps"),
                "support_psnr": s.get(pskey),
                "init": "scratch",
            })
    out = BASE / "selection.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
