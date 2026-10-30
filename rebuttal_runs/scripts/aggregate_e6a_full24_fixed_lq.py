#!/usr/bin/env python3
"""Compare E6-A full24 original on-the-fly LQ vs one shared fixed LQ (7 scenarios)."""
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rebuttal_runs/E6A_full24"
SCEN = ["blind_0003", "blind_0005", "blind_0017", "blind_0018",
        "blind_0020", "blind_0022", "blind_0023"]
KINDS = {"best": "test_metrics_gpu*.jsonl",
         "last": "test_metrics_last_gpu*.jsonl",
         "final": "test_metrics_final_gpu*.jsonl"}
FIXED = {"best": "test_metrics_fixed_best_gpu*.jsonl",
         "last": "test_metrics_fixed_last_gpu*.jsonl",
         "final": "test_metrics_fixed_final_gpu*.jsonl"}


def load(pattern):
    by = defaultdict(lambda: defaultdict(list))
    for p in glob.glob(str(OUT / pattern)):
        for l in open(p):
            if not l.strip():
                continue
            r = json.loads(l)
            rep = int(re.search(r"_rep(\d+)$", r["run_id"]).group(1))
            by[(r["scenario_id"], rep)][r["dataset"]].append(r["psnr_rgb"])
    return by


def _stats(v):
    return {"mean": float(np.mean(v)) if v else None,
            "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
            "range": float(max(v) - min(v)) if len(v) > 1 else None}


def _pipeline_sets(kind):
    if kind == "final":
        rows = [json.loads(l) for l in open(OUT / "selection_final.jsonl")]
    elif kind == "last":
        rows = [json.loads(l) for l in open(OUT / "selection_last.jsonl")]
    else:
        rows = [json.loads(l) for l in open(OUT / "selection.jsonl")]
    import re as _re
    out = {}
    for r in rows:
        rep = int(_re.search(r"_rep(\d+)$", r["run_id"]).group(1))
        out[(r["scenario_id"], rep)] = tuple(
            (x[0], int(x[1])) for x in r["predicted_pipeline"])
    return out


def _gt_steps(sid):
    p = json.load(open(ROOT / f"exp41/degradation_gt/{sid}_params.json"))
    return len(p["pipeline"])


def corrected_macro(kind, orig, fix):
    pipes = _pipeline_sets(kind)
    rows = {}
    for sid in [f"blind_{i:04d}" for i in range(1, 25)]:
        src = fix if sid in SCEN else orig
        v = {ds: [float(np.mean(src[(sid, r)][ds])) for r in (1, 2)
                  if src[(sid, r)][ds]] for ds in ("DIV2K", "LSDIR")}
        rows[sid] = {ds: _stats(v[ds]) for ds in ("DIV2K", "LSDIR")}
        rows[sid]["gt_steps"] = _gt_steps(sid)
        rows[sid]["pipeline_equal"] = pipes[(sid, 1)] == pipes[(sid, 2)]
    macro = {}
    for ds in ("DIV2K", "LSDIR"):
        means = [rows[s][ds]["mean"] for s in rows if rows[s][ds]["mean"] is not None]
        sds = [rows[s][ds]["sd"] for s in rows if rows[s][ds]["sd"] is not None]
        ranges = [rows[s][ds]["range"] for s in rows if rows[s][ds]["range"] is not None]
        macro[ds] = {
            "mean_of_scenario_means": float(np.mean(means)),
            "mean_within_scenario_sd": float(np.mean(sds)),
            "mean_within_scenario_range": float(np.mean(ranges)),
            "max_within_scenario_range": float(max(ranges)),
        }
    strata = {}
    for gs in (1, 2, 3):
        group = [s for s in rows if rows[s]["gt_steps"] == gs]
        strata[str(gs)] = {"n_scenarios": len(group)}
        for ds in ("DIV2K", "LSDIR"):
            sds = [rows[s][ds]["sd"] for s in group if rows[s][ds]["sd"] is not None]
            ranges = [rows[s][ds]["range"] for s in group if rows[s][ds]["range"] is not None]
            strata[str(gs)][ds] = {
                "mean_within_scenario_sd": float(np.mean(sds)) if sds else None,
                "mean_within_scenario_range": float(np.mean(ranges)) if ranges else None,
                "max_within_scenario_range": float(max(ranges)) if ranges else None,
            }
    groups = {}
    for label, cond in (("pipeline_equal", True), ("pipeline_different", False)):
        sids = [s for s in rows if rows[s]["pipeline_equal"] == cond]
        groups[label] = {"n_scenarios": len(sids)}
        for ds in ("DIV2K", "LSDIR"):
            ad = [abs(rows[s][ds]["range"]) for s in sids
                  if rows[s][ds]["range"] is not None]
            groups[label][ds] = {"mean_abs_delta": float(np.mean(ad)) if ad else None,
                                 "max_abs_delta": float(max(ad)) if ad else None}
    return {"scenarios": rows, "macro": macro,
            "stratified_by_gt_steps": strata,
            "pipeline_agreement_groups": groups}


def main():
    out = {"scenarios": {}, "macro": {}, "note":
           "Fixed-LQ uses one materialized realization shared by all checkpoints "
           "(E1E2_fixed_lq); original uses per-process LQ generation."}
    for kind in ("best", "last", "final"):
        orig = load(KINDS[kind])
        fix = load(FIXED[kind])
        rec = {}
        deltas = {"DIV2K": [], "LSDIR": []}
        for sid in SCEN:
            s = {}
            for rep in (1, 2):
                e = {}
                for ds in ("DIV2K", "LSDIR"):
                    o = float(np.mean(orig[(sid, rep)][ds])) if orig[(sid, rep)][ds] else None
                    f = float(np.mean(fix[(sid, rep)][ds])) if fix[(sid, rep)][ds] else None
                    e[ds] = {"orig": o, "fixed": f,
                             "delta_fixed_minus_orig": (f - o) if (o is not None and f is not None) else None}
                    if e[ds]["delta_fixed_minus_orig"] is not None:
                        deltas[ds].append(e[ds]["delta_fixed_minus_orig"])
                s[f"rep{rep}"] = e
            for ds in ("DIV2K", "LSDIR"):
                for key, src in (("orig", orig), ("fixed", fix)):
                    v = [float(np.mean(src[(sid, r)][ds])) for r in (1, 2)
                         if src[(sid, r)][ds]]
                    s.setdefault(ds, {})[key] = _stats(v)
            rec[sid] = s
        out[kind] = rec
        out["macro"][kind] = {
            ds: {"mean_delta": float(np.mean(v)) if v else None,
                 "max_abs_delta": float(max(abs(x) for x in v)) if v else None,
                 "n_runs": len(v)} for ds, v in deltas.items()}
        out.setdefault("corrected_fixed_lq", {})[kind] = corrected_macro(kind, orig, fix)
    (OUT / "fixed_lq_sensitivity.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# E6-A full24 fixed-LQ sensitivity (7 nondeterministic scenarios)", "",
             "Original pass generated LQ per process (rep1/rep2 and passes differ); "
             "fixed pass uses one shared materialized realization.", ""]
    for kind in ("best", "last", "final"):
        m = out["macro"][kind]
        lines.append(f"- {kind}: mean Δ(fixed−orig) DIV2K {m['DIV2K']['mean_delta']:+.3f} dB "
                     f"(max |Δ| {m['DIV2K']['max_abs_delta']:.3f}); "
                     f"LSDIR {m['LSDIR']['mean_delta']:+.3f} dB "
                     f"(max |Δ| {m['LSDIR']['max_abs_delta']:.3f})")
    lines += ["", "## Corrected macro (fixed LQ for 7 scenarios; original for other 17)", "",
              "| kind | DIV2K macro | within-SD | max range | LSDIR macro | within-SD | max range |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for kind in ("best", "last", "final"):
        cm = out["corrected_fixed_lq"][kind]["macro"]
        d, l = cm["DIV2K"], cm["LSDIR"]
        lines.append(f"| {kind} | {d['mean_of_scenario_means']:.3f} | "
                     f"{d['mean_within_scenario_sd']:.3f} | "
                     f"{d['max_within_scenario_range']:.3f} | "
                     f"{l['mean_of_scenario_means']:.3f} | "
                     f"{l['mean_within_scenario_sd']:.3f} | "
                     f"{l['max_within_scenario_range']:.3f} |")
    lines += ["", "| Scenario | rep | kind | orig DIV2K | fixed DIV2K | Δ | orig range | fixed range |",
              "|---|---:|---|---:|---:|---:|---:|---:|"]
    for kind in ("best", "final"):
        for sid in SCEN:
            s = out[kind][sid]
            for rep in (1, 2):
                e = s[f"rep{rep}"]
                lines.append(f"| {sid} | {rep} | {kind} | {e['DIV2K']['orig']:.3f} | "
                             f"{e['DIV2K']['fixed']:.3f} | {e['DIV2K']['delta_fixed_minus_orig']:+.3f} | "
                             f"{s['DIV2K']['orig']['range']:.3f} | {s['DIV2K']['fixed']['range']:.3f} |")
    (OUT / "fixed_lq_sensitivity.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
