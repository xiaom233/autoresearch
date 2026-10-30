#!/usr/bin/env python3
"""Materialize E6-A full-24 challenge copies: 24 scenarios x 2 independent runs.

Only the 5 support pairs (clean_i / degraded_i + clean/degraded aliases) are
copied. No GT params, no prior predictions. Both reps are hardlinked to the
same exp41 source files, so sha256 equality across reps and source is exact.
"""
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "exp41/challenges/phase4"
OUT = ROOT / "rebuttal_runs/E6A_full24"
SCENARIOS = [f"blind_{i:04d}" for i in range(1, 25)]
REPS = ["rep1", "rep2"]
FILE_NAMES = (["clean.png", "degraded.png"]
              + [f"clean_{i}.png" for i in range(5)]
              + [f"degraded_{i}.png" for i in range(5)])


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    challenges = OUT / "challenges"
    if challenges.exists() and any(challenges.iterdir()):
        sys.exit(f"refusing to overwrite non-empty {challenges}")
    challenges.mkdir(parents=True, exist_ok=True)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SRC.relative_to(ROOT)),
        "scenarios": SCENARIOS,
        "reps": REPS,
        "note": "support images only; no GT params, no prior predictions",
        "runs": {},
    }

    for sid in SCENARIOS:
        src_dir = SRC / sid
        missing = [n for n in FILE_NAMES if not (src_dir / n).exists()]
        if missing:
            sys.exit(f"{sid}: missing source files {missing}")
        src_hashes = {n: sha256(src_dir / n) for n in FILE_NAMES}
        for rep in REPS:
            dst_dir = challenges / f"{sid}_{rep}"
            dst_dir.mkdir(parents=True, exist_ok=False)
            for n in FILE_NAMES:
                link_or_copy(src_dir / n, dst_dir / n)
            dst_hashes = {n: sha256(dst_dir / n) for n in FILE_NAMES}
            if dst_hashes != src_hashes:
                sys.exit(f"{sid}_{rep}: copied hashes differ from source")
            pairs = [
                {"index": i,
                 "clean_sha256": src_hashes[f"clean_{i}.png"],
                 "degraded_sha256": src_hashes[f"degraded_{i}.png"]}
                for i in range(5)
            ]
            manifest = {
                "scenario_id": sid,
                "rep": rep,
                "pairs": pairs,
                "aliases": {
                    "clean.png": src_hashes["clean.png"],
                    "degraded.png": src_hashes["degraded.png"],
                },
            }
            (dst_dir / "support_manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            report["runs"][f"{sid}_{rep}"] = {
                "pairs": pairs,
                "aliases": manifest["aliases"],
            }
            print(f"[ok] {sid}_{rep}", flush=True)

    # cross-rep equality check
    for sid in SCENARIOS:
        a = report["runs"][f"{sid}_rep1"]
        b = report["runs"][f"{sid}_rep2"]
        if a != b:
            sys.exit(f"{sid}: rep1/rep2 hashes differ")
    report["cross_rep_check"] = "PASS: all 24 scenarios rep1 == rep2"

    (OUT / "setup_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {len(report['runs'])} challenge dirs -> {challenges}")
    print(report["cross_rep_check"])


if __name__ == "__main__":
    main()
