#!/usr/bin/env python3
"""E4: run the existing rule controller on real support pairs.

No new operators, no threshold tuning. The only difference from exp41 is that
the 5 support pairs are real images (SIDD/UHDM) rather than synthetic ones.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rebuttal_runs" / "scripts"))
import rule_controller as rc  # noqa: E402


def load_pairs(d):
    pairs = []
    for i in range(5):
        c = d / f"clean_{i}.png"
        lq = d / f"degraded_{i}.png"
        if not (c.exists() and lq.exists()):
            raise FileNotFoundError(f"missing pair {i} in {d}")
        pairs.append((c, lq,
                      np.array(Image.open(c).convert("RGB"), dtype=np.uint8),
                      np.array(Image.open(lq).convert("RGB"), dtype=np.uint8)))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--support-dir", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()
    sup = Path(args.support_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    analyses, steps, errors = [], [], []
    for i in range(5):
        c = sup / f"clean_{i}.png"
        lq = sup / f"degraded_{i}.png"
        try:
            r = rc.subprocess.run([rc.PY, str(rc.RFA), "--target", str(lq), "--clean", str(c)],
                                  capture_output=True, text=True, timeout=args.timeout,
                                  env=rc._cpu_env())
            if r.returncode != 0:
                raise RuntimeError((r.stderr or r.stdout)[-500:])
            d = rc.parse_rfa(r.stdout)
            st = rc.dc_from_one(d)
            analyses.append(d); steps.append(st)
            (out / f"full_analysis_{i}.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
            print(f"[e4-rule] pair{i}: {st}", flush=True)
        except Exception as e:
            errors.append({"pair": i, "error": str(e)})
            print(f"[e4-rule] pair{i}: ERROR {e}", flush=True)

    agg_pipe, agg = rc.aggregate(steps) if steps else ([], {})
    consensus = bool(agg_pipe)
    fallback_used = False
    fallback_reason = None
    if not agg_pipe and steps:
        # No category reached 3/5 consensus. Keep the first pair's evidence
        # as an explicitly labelled fallback, not as a confident prediction.
        fallback_used = True
        fallback_reason = "no category reached 3/5 consensus; single-pair fallback"
        agg_pipe = [(s["func"], s["sev"], s["cat"]) for s in steps[0]][:3]
        agg = {s["cat"]: {"func": s["func"], "sev": s["sev"], "votes": 1, "n": len(steps)}
               for s in steps[0][:3]}
    pipeline = [(f, s) for f, s, _ in agg_pipe]
    rs_id, arch = rc.rs_for(agg_pipe, agg, analyses) if analyses else ("default_swin_sg", {
        "attention_type": "swin", "embed_dim": 64, "window_size": 8, "use_simple_gate": 1})

    score = None
    if pipeline:
        pairs = load_pairs(sup)
        import tempfile
        score = rc.score_pipeline(args.scenario, pipeline, pairs, Path(tempfile.mkdtemp(prefix="e4sim_")))

    verdict = ("POOR" if fallback_used else
               "LIKELY" if score and rc.threshold_good(score) else
               "POOR" if not pipeline else "UNCERTAIN")
    params = {"pipeline": [{"function": f, "severity": int(s)} for f, s in pipeline],
              "analysis": {"source": "existing rule controller on real support pairs",
                           "verdict": verdict, "score": score, "rs_rule": rs_id,
                           "arch": arch, "n_valid_pairs": len(steps),
                           "consensus": consensus, "fallback_used": fallback_used,
                           "fallback_reason": fallback_reason,
                           "errors": errors, "note": "no new operators or thresholds"}}
    (out / "predicted_params_rule.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    (out / "e4_rule_decision.json").write_text(json.dumps({
        "scenario": args.scenario, "support_dir": str(sup),
        "aggregated": agg, "pipeline": pipeline, "score": score,
        "verdict": verdict, "rs_rule": rs_id, "arch": arch,
        "consensus": consensus, "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "errors": errors, "n_valid_pairs": len(steps),
        "arch_flags": rc.arch_flags(arch),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"scenario": args.scenario, "pipeline": pipeline, "score": score,
                      "verdict": verdict, "rs_rule": rs_id}, ensure_ascii=False))


if __name__ == "__main__":
    main()
