#!/usr/bin/env python3
"""Materialize E6-C support-size ablation: 24 scenarios x {size1,size3}.

Nested design: for each scenario, take E6-B setA's 5 support images, apply a
deterministic permutation pi(sid), then size1={pi[0]}, size3={pi[0..2]}.
The same image always gets the same LQ realization (key = image_id, not size),
so 1 < 3 < 5 share image identity and degradation.
"""
import hashlib
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rebuttal_runs/scripts"))
from eval_checkpoint import load_rgb, make_lq  # noqa: E402

OUT = ROOT / "rebuttal_runs/E6C_support_size"
E6B = ROOT / "rebuttal_runs/E6B_full24"
SCENARIOS = [f"blind_{i:04d}" for i in range(1, 25)]
SIZES = {"size1": 1, "size3": 3}
SEED = 20261030
LQ_SALT = "e6c-size-v1"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def stable_seed(*parts):
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:16], 16) % (2 ** 32)


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

    plan = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "base": "rebuttal_runs/E6B_full24 setA (5 random DIV2K_train images)",
        "nested": "size1 = pi[0]; size3 = pi[0..2]; size5 = setA = pi[0..4]",
        "permutation": "random.Random(stable_seed(SEED, sid, 'size_order')).shuffle(setA pairs)",
        "lq_salt": LQ_SALT,
        "lq_note": "make_lq key = image_id (size-independent), so identical images share identical LQ",
        "scenarios": {},
    }
    for sid in SCENARIOS:
        base_dir = E6B / "challenges" / f"{sid}_setA"
        man = json.load(open(base_dir / "support_manifest.json"))
        pairs5 = sorted(man["pairs"], key=lambda p: p["index"])
        assert len(pairs5) == 5, (sid, len(pairs5))
        pipe = json.load(open(ROOT / f"exp41/degradation_gt/{sid}_params.json"))["pipeline"]
        perm = pairs5[:]
        random.Random(stable_seed(SEED, sid, "size_order")).shuffle(perm)
        rec = {"pipeline": pipe,
               "permutation": [p["image_id"] for p in perm], "sizes": {}}
        for size_name, k in SIZES.items():
            picked = perm[:k]
            d = challenges / f"{sid}_{size_name}"
            d.mkdir(parents=True, exist_ok=False)
            pairs = []
            for i, src_rec in enumerate(picked):
                src = ROOT / src_rec["source"]
                clean = load_rgb(src)
                lq = make_lq(clean, sid, src.stem, pipe, LQ_SALT)
                cpath = d / f"clean_{i}.png"
                lpath = d / f"degraded_{i}.png"
                link_or_copy(src, cpath)
                Image.fromarray(lq).save(lpath, compress_level=1)
                pairs.append({
                    "index": i, "image_id": src.stem,
                    "source": str(src.relative_to(ROOT)),
                    "clean_sha256": sha(cpath), "degraded_sha256": sha(lpath),
                })
            link_or_copy(d / "clean_0.png", d / "clean.png")
            link_or_copy(d / "degraded_0.png", d / "degraded.png")
            manifest = {
                "scenario_id": sid, "size": size_name, "k": k,
                "sampling_seed": stable_seed(SEED, sid, "size_order"),
                "permutation": [p["image_id"] for p in perm],
                "pairs": pairs,
                "aliases": {"clean.png": pairs[0]["clean_sha256"],
                            "degraded.png": pairs[0]["degraded_sha256"]},
            }
            (d / "support_manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            rec["sizes"][size_name] = pairs
            print(f"[ok] {sid}_{size_name}: {[p['image_id'] for p in picked]}", flush=True)
        plan["scenarios"][sid] = rec
    (OUT / "support_plan_gt.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    os.chmod(OUT / "support_plan_gt.json", 0o600)
    print(f"[done] 48 challenge dirs (24 x size1/size3)")


if __name__ == "__main__":
    main()
