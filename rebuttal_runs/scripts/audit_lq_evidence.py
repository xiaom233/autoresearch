#!/usr/bin/env python3
"""Reproduce the LQ-consistency audit evidence."""
import glob
import hashlib
import io
import json
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rebuttal_runs/scripts"))
sys.path.insert(0, str(ROOT))
from eval_checkpoint import load_rgb, make_lq  # noqa: E402

SCENARIOS = ["blind_0003", "blind_0005", "blind_0017", "blind_0018",
             "blind_0020", "blind_0022", "blind_0023"]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


def load_metric_rows(pattern):
    out = []
    for p in glob.glob(str(ROOT / pattern)):
        out += [json.loads(l) for l in open(p) if l.strip()]
    return out


def tar_image(method, sid, name):
    for f in sorted(glob.glob(str(ROOT / f"benchmark_results/{method}/DIV2K_gpu*.tar"))):
        try:
            with tarfile.open(f) as t:
                for m in t.getmembers():
                    if m.name.endswith(f"{sid}/{name}.degraded.jpg"):
                        return np.array(Image.open(io.BytesIO(
                            t.extractfile(m).read())).convert("RGB"))
        except Exception:
            pass
    return None


def main():
    ev = {}
    # 1) support/test overlap from the audited manifest
    sup = json.load(open(ROOT / "rebuttal_runs/A0/support_test_overlap.json"))
    ev["support_test_overlap"] = {
        "div2k_test_images": sup["div2k_test_images"],
        "total_support_draws": sup["total_support_draws"],
        "unique_div2k_test_images_used_as_support": sup["unique_div2k_test_images_used_as_support"],
        "per_scenario_after_excluding_support": 95,
        "note": "all 120 support draws matched DIV2K_valid_HR; 74 unique images",
    }

    # 2) repeatability of make_lq for the 7 suspected scenarios
    clean = load_rgb(ROOT / "datasets/DIV2K/DIV2K_valid_HR/0801.png")
    det = {}
    for sid in SCENARIOS:
        pipe = json.load(open(ROOT / f"exp41/degradation_gt/{sid}_params.json"))["pipeline"]
        a = make_lq(clean, sid, "DIV2K/0801", pipe, "v1")
        b = make_lq(clean, sid, "DIV2K/0801", pipe, "v1")
        det[sid] = {
            "pipeline": pipe,
            "hash_a": sha(a.tobytes()), "hash_b": sha(b.tobytes()),
            "identical": sha(a.tobytes()) == sha(b.tobytes()),
            "lq_vs_lq_psnr": psnr(a, b),
            "lq_vs_clean_psnr_a": psnr(a, clean),
        }
    ev["make_lq_repeatability"] = det

    # 3) LQ hash agreement across controllers' existing metrics
    def km(rows):
        return {(r["scenario_id"], r["dataset"], r["image_id"]): r["lq_sha256"] for r in rows}
    a0 = km(load_metric_rows("rebuttal_runs/A0/test_metrics/*.jsonl"))
    e1 = km(load_metric_rows("rebuttal_runs/E1/test_metrics/*.jsonl"))
    e2 = km(load_metric_rows("rebuttal_runs/E2/test_metrics_labeled.jsonl"))
    agree = {}
    for name, X, Y in [("AutoIR_vs_rule", a0, e1), ("AutoIR_vs_DFPIR", a0, e2),
                       ("rule_vs_DFPIR", e1, e2)]:
        common = set(X) & set(Y)
        bad = [k for k in common if X[k] != Y[k]]
        agree[name] = {"common": len(common), "mismatch": len(bad),
                       "scenarios": dict(Counter(k[0] for k in bad))}
    ev["controller_lq_agreement"] = agree

    # 4) baseline benchmark LQ vs unified evaluator / other baseline
    base = {}
    for sid, fn in [("blind_0006", "0801"), ("blind_0001", "0801"),
                    ("blind_0005", "0801"), ("blind_0003", "0801")]:
        pipe = json.load(open(ROOT / f"exp41/degradation_gt/{sid}_params.json"))["pipeline"]
        unified = make_lq(load_rgb(ROOT / f"datasets/DIV2K/DIV2K_valid_HR/{fn}.png"),
                          sid, f"DIV2K/{fn}", pipe, "v1")
        air = tar_image("airnet", sid, fn)
        pro = tar_image("promptir", sid, fn)
        rec = {"pipeline": pipe}
        if air is not None:
            rec["airnet_vs_unified_psnr"] = psnr(air, unified)
        if air is not None and pro is not None:
            rec["airnet_vs_promptir_psnr"] = psnr(air, pro)
        base[sid] = rec
    ev["baseline_lq_consistency"] = base

    out = ROOT / "rebuttal_runs/AUDIT_lq_evidence.json"
    out.write_text(json.dumps(ev, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(ev, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
