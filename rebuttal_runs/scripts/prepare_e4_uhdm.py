#!/usr/bin/env python3
"""Extract UHDM test pairs and create 5 support + a fixed test subset."""
import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path

SUFFIX = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def stem_key(p):
    n = p.stem
    for s in ("_moire", "-moire", "_gt", "-gt", "_clean", "-clean", "_input", "-input"):
        if n.endswith(s):
            n = n[: -len(s)]
    return n


def classify(dirs):
    clean, deg = [], []
    for d in dirs:
        n = d.name.lower()
        if any(x in n for x in ("gt", "clean", "target", "clear", "sharp")):
            clean.append(d)
        elif any(x in n for x in ("moire", "input", "degraded", "screen", "lq")):
            deg.append(d)
    return clean, deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--n-test", type=int, default=50)
    args = ap.parse_args()
    out = Path(args.out_root)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if not any(raw.iterdir()):
        with tarfile.open(args.tar) as t:
            t.extractall(raw)
    images = [p for p in raw.rglob("*") if p.suffix.lower() in SUFFIX]
    print("images", len(images))
    dirs = sorted({p.parent for p in images})
    clean_dirs, deg_dirs = classify(dirs)
    print("clean dirs", [str(d) for d in clean_dirs])
    print("degraded dirs", [str(d) for d in deg_dirs])

    if clean_dirs and deg_dirs:
        clean_map = {}
        for d in clean_dirs:
            for p in d.rglob("*"):
                if p.suffix.lower() in SUFFIX:
                    clean_map.setdefault(stem_key(p), p)
        pairs = []
        for d in deg_dirs:
            for p in d.rglob("*"):
                if p.suffix.lower() in SUFFIX:
                    c = clean_map.get(stem_key(p))
                    if c:
                        pairs.append((p, c))
    else:
        # fallback: pair by stem across all directories
        by = {}
        for p in images:
            by.setdefault(stem_key(p), []).append(p)
        pairs = []
        for k, ps in by.items():
            if len(ps) >= 2:
                gt = [p for p in ps if "gt" in p.stem.lower()]
                mo = [p for p in ps if "moire" in p.stem.lower()]
                if gt and mo:
                    pairs.append((mo[0], gt[0]))  # (degraded, clean)
                else:
                    pairs.append((ps[0], ps[1]))
    pairs = sorted({(str(a), str(b)) for a, b in pairs})
    print("pairs", len(pairs))
    if len(pairs) < 6:
        raise RuntimeError("not enough paired images; inspect raw directory")
    support = pairs[:5]
    rest = pairs[5:]
    if len(rest) <= args.n_test:
        test = rest
    else:
        step = len(rest) / args.n_test
        test = [rest[int(i * step)] for i in range(args.n_test)]

    def save(sel, sub):
        d = out / sub
        d.mkdir(parents=True, exist_ok=True)
        meta = []
        for i, (lq, clean) in enumerate(sel):
            lq_p, c_p = Path(lq), Path(clean)
            (d / f"degraded_{i}.png").symlink_to(lq_p.resolve())
            (d / f"clean_{i}.png").symlink_to(c_p.resolve())
            meta.append({"index": i, "degraded_source": str(lq_p), "clean_source": str(c_p),
                         "degraded_sha256": sha256(lq_p), "clean_sha256": sha256(c_p)})
        return meta

    report = {"source": args.tar, "n_pairs_total": len(pairs),
              "support": save(support, "support"), "test": save(test, "test"),
              "test_selection": f"fixed every-{len(rest)/max(1,args.n_test):.3f}-th of remaining pairs"}
    (out / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({"support": len(support), "test": len(test)}, indent=2))


if __name__ == "__main__":
    main()
