#!/usr/bin/env python3
"""Aggregate E6-A full24: run-to-run blind-ID variance + fixed-LQ test PSNR."""
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rebuttal_runs/E6A_full24"
SCENARIOS = [f"blind_{i:04d}" for i in range(1, 25)]
REPS = [1, 2]


def pipe_key(p):
    if isinstance(p[0], dict):
        return [(x["function"], int(x["severity"])) for x in p]
    return [(x[0], int(x[1])) for x in p]


def gt_pipe(sid):
    p = json.loads((ROOT / "exp41/degradation_gt" / f"{sid}_params.json").read_text())
    return pipe_key(p["pipeline"])


KIND_FILES = {
    "best": ("selection.jsonl", "test_metrics_gpu*.jsonl"),
    "last": ("selection_last.jsonl", "test_metrics_last_gpu*.jsonl"),
    "final": ("selection_final.jsonl", "test_metrics_final_gpu*.jsonl"),
}


def load_metrics(kind):
    pat = KIND_FILES[kind][1]
    rows = []
    for p in glob.glob(str(OUT / pat)):
        rows += [json.loads(l) for l in open(p) if l.strip()]
    by_run = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_run[r["run_id"]][r["dataset"]].append((r["psnr_rgb"], r.get("ssim_rgb")))
    return rows, by_run


def collect(kind):
    sel_name = KIND_FILES[kind][0]
    sel_path = OUT / sel_name
    if not sel_path.exists():
        return None
    sel = {r["run_id"]: r for r in map(json.loads, open(sel_path))}
    rows, by_run = load_metrics(kind)
    if not rows:
        return None
    scenarios = {}
    for sid in SCENARIOS:
        g = gt_pipe(sid)
        rec = {"gt_pipeline": [list(x) for x in g], "gt_steps": len(g), "reps": {}}
        psnrs = {"DIV2K": [], "LSDIR": []}
        for rep in REPS:
            rid = f"E6A24_{sid}_rep{rep}"
            s = sel.get(rid)
            if s is None:
                rec["reps"][f"rep{rep}"] = None
                continue
            pred = pipe_key(s["predicted_pipeline"])
            entry = {
                "predicted_pipeline": [list(x) for x in pred],
                "predicted_verdict": s.get("predicted_verdict"),
                "predicted_ci": s.get("predicted_ci"),
                "support_psnr": s.get("support_psnr"),
                "checkpoint_step": s.get("checkpoint_step"),
                "selected_round": s.get("selected_round"),
                "pipeline_exact_gt": pred == g,
            }
            for ds in ("DIV2K", "LSDIR"):
                vals = by_run[rid].get(ds, [])
                entry[ds] = float(np.mean([v[0] for v in vals])) if vals else None
                entry[f"n_{ds}"] = len(vals)
                entry[f"ssim_{ds}"] = float(np.mean([v[1] for v in vals
                                                      if v[1] is not None])) if vals else None
                if vals:
                    psnrs[ds].append(entry[ds])
            rec["reps"][f"rep{rep}"] = entry
        rec["pipeline_equal"] = (len(psnrs["DIV2K"]) == 2 and
                                 rec["reps"]["rep1"]["predicted_pipeline"] ==
                                 rec["reps"]["rep2"]["predicted_pipeline"])
        for ds in ("DIV2K", "LSDIR"):
            v = psnrs[ds]
            rec[ds] = {
                "mean": float(np.mean(v)) if v else None,
                "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                "range": float(max(v) - min(v)) if len(v) > 1 else None,
                "delta_rep2_minus_rep1": (float(rec["reps"]["rep2"][ds] -
                                                rec["reps"]["rep1"][ds])
                                          if len(v) == 2 and None not in v else None),
            }
        scenarios[sid] = rec

    def macro_stats(ds):
        means = [scenarios[s][ds]["mean"] for s in SCENARIOS
                 if scenarios[s][ds]["mean"] is not None]
        sds = [scenarios[s][ds]["sd"] for s in SCENARIOS
               if scenarios[s][ds]["sd"] is not None]
        ranges = [scenarios[s][ds]["range"] for s in SCENARIOS
                  if scenarios[s][ds]["range"] is not None]
        deltas = [scenarios[s][ds]["delta_rep2_minus_rep1"] for s in SCENARIOS
                  if scenarios[s][ds]["delta_rep2_minus_rep1"] is not None]
        return {
            "n_scenarios": len(means),
            "mean_of_scenario_means": float(np.mean(means)) if means else None,
            "between_scenario_sd": float(np.std(means, ddof=1)) if len(means) > 1 else None,
            "mean_within_scenario_sd": float(np.mean(sds)) if sds else None,
            "max_within_scenario_range": float(max(ranges)) if ranges else None,
            "mean_within_scenario_range": float(np.mean(ranges)) if ranges else None,
            "delta_mean": float(np.mean(deltas)) if deltas else None,
            "delta_median": float(np.median(deltas)) if deltas else None,
            "delta_sd": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None,
            "abs_delta_gt_1db": int(sum(abs(x) > 1.0 for x in deltas)),
            "abs_delta_gt_2db": int(sum(abs(x) > 2.0 for x in deltas)),
        }

    stratified = {}
    for gs in (1, 2, 3):
        group = [s for s in SCENARIOS if scenarios[s]["gt_steps"] == gs]
        stratified[str(gs)] = {"n_scenarios": len(group)}
        for ds in ("DIV2K", "LSDIR"):
            sds = [scenarios[s][ds]["sd"] for s in group
                   if scenarios[s][ds]["sd"] is not None]
            ranges = [scenarios[s][ds]["range"] for s in group
                      if scenarios[s][ds]["range"] is not None]
            means = [scenarios[s][ds]["mean"] for s in group
                     if scenarios[s][ds]["mean"] is not None]
            stratified[str(gs)][ds] = {
                "mean_of_scenario_means": float(np.mean(means)) if means else None,
                "mean_within_scenario_sd": float(np.mean(sds)) if sds else None,
                "mean_within_scenario_range": float(np.mean(ranges)) if ranges else None,
                "max_within_scenario_range": float(max(ranges)) if ranges else None,
            }

    groups = {}
    for label, cond in (("pipeline_equal", lambda r: r["pipeline_equal"]),
                        ("pipeline_different", lambda r: not r["pipeline_equal"])):
        sids = [s for s in SCENARIOS if cond(scenarios[s])]
        entry = {"n_scenarios": len(sids), "scenarios": sids}
        for ds in ("DIV2K", "LSDIR"):
            ad = [abs(scenarios[s][ds]["delta_rep2_minus_rep1"]) for s in sids
                  if scenarios[s][ds]["delta_rep2_minus_rep1"] is not None]
            entry[ds] = {"mean_abs_delta": float(np.mean(ad)) if ad else None,
                         "max_abs_delta": float(max(ad)) if ad else None}
        groups[label] = entry

    return {
        "selection_kind": kind,
        "n_metric_rows": len(rows),
        "scenarios": scenarios,
        "macro": {ds: macro_stats(ds) for ds in ("DIV2K", "LSDIR")},
        "stratified_by_gt_steps": stratified,
        "pipeline_agreement_groups": groups,
    }


def _best_psnr(log):
    if not log.exists():
        return None
    txt = log.read_text(errors="ignore")
    if "TRAIN END" not in txt:
        return None
    m = re.search(r"Loaded best checkpoint: step (\d+) \(Set14 PSNR_RGB=([0-9.]+|nan)\)", txt)
    if not m or m.group(2) == "nan":
        return None
    return float(m.group(2)), int(m.group(1))


def round_flow():
    plan = []
    if (OUT / "reflection_plan.json").exists():
        plan = json.load(open(OUT / "reflection_plan.json"))
    need_r1 = [x for x in plan if x.get("need_r1")]
    decs = []
    for p in glob.glob(str(OUT / "challenges/*/r1_decision.json")):
        try:
            decs.append(json.load(open(p)))
        except Exception:
            pass
    changes = [d for d in decs if d.get("decision") == "CHANGE"]
    r1_trained, r1_delta = [], []
    for d in decs:
        name = d.get("challenge_id")
        if not name:
            continue
        r0 = _best_psnr(OUT / "logs" / f"{name}.log")
        r1 = _best_psnr(OUT / "logs" / f"{name}_R1.log")
        if r1:
            r1_trained.append(name)
            if r0:
                r1_delta.append(r1[0] - r0[0])
    r2plan = []
    if (OUT / "r2_plan.json").exists():
        r2plan = json.load(open(OUT / "r2_plan.json"))
    need_r2 = [x for x in r2plan if x.get("need_r2")]
    r2_trained = [Path(p).name[:-7] for p in glob.glob(str(OUT / "logs/*_R2.log"))
                  if _best_psnr(Path(p))]
    selected = {}
    if (OUT / "selection_final.jsonl").exists():
        for r in map(json.loads, open(OUT / "selection_final.jsonl")):
            selected[r["selected_round"]] = selected.get(r["selected_round"], 0) + 1
    over = [{"challenge_id": d.get("challenge_id"),
             "candidates_tested": d.get("candidates_tested")}
            for d in decs if (d.get("candidates_tested") or 0) > 15]
    return {
        "r1_triggered": len(need_r1),
        "r1_decision_files": len(decs),
        "r1_change": len(changes),
        "r1_trained": len(r1_trained),
        "r1_support_delta_mean": float(np.mean(r1_delta)) if r1_delta else None,
        "r1_support_delta_median": float(np.median(r1_delta)) if r1_delta else None,
        "r1_candidate_budget_over_15": over,
        "r2_triggered": len(need_r2),
        "r2_trained": len(r2_trained),
        "selected_round_counts": selected,
    }


def fmt(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


def main():
    blind = json.load(open(OUT / "blind_agreement.json"))
    audit = json.load(open(OUT / "blind_id_audit.json"))
    out = {
        "scope_notes": [
            "E6-A full24: 24 exp41 scenarios x 2 independent Skill agent blind-ID runs; "
            "same 5 support pairs (sha256-identical) and same fixed training config/seed.",
            "R0-only: no post-training R1/R2 reflection. Measures blind-ID + single-round R0 variance, "
            "not the full AutoIR R0-R2 pipeline.",
            "best = support-PSNR-selected checkpoint; last = fixed final step (isolates training/selection jitter).",
            "Test: fixed GT-LQ DIV2K 100 + LSDIR 250, eval_checkpoint.py --seed-salt v1, no LPIPS.",
            "API model/version/temperature UNKNOWN; variance is for this agent configuration only.",
            "R1/R2 extension added 2026-10-29 at user request; trigger rules pre-registered before R1 started.",
            "Independent evaluation passes can differ by <=0.01 dB on unchanged checkpoints (GPU nondeterminism).",
        ],
        "blind_id_agreement": blind["summary"],
        "blind_id_stratified_by_gt_steps": blind["stratified_by_gt_steps"],
        "blind_id_audit": {
            "n_runs": len(audit),
            "verdicts": {v: sum(1 for r in audit if r["verdict"] == v)
                         for v in sorted({r["verdict"] for r in audit})},
            "predicted_steps": {str(k): sum(1 for r in audit if r["steps"] == k)
                                for k in sorted({r["steps"] for r in audit})},
            "candidate_tests_over_25": [
                {"run": r["cid"], "candidate_tests": r["candidates"]}
                for r in audit if (r["candidates"] or 0) > 25],
            "agent_wall_seconds_median": float(np.median(
                [r["wall_s"] for r in audit if r["wall_s"]])),
            "agent_wall_seconds_max": max(r["wall_s"] for r in audit if r["wall_s"]),
        },
    }
    out["round_flow"] = round_flow()
    for kind in ("best", "last", "final"):
        try:
            sec = collect(kind)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {kind}: {e}")
            sec = None
        if sec:
            out[kind] = sec

    if "best" in out and "final" in out:
        fv = {"scenarios": {}, "macro": {}}
        for ds in ("DIV2K", "LSDIR"):
            deltas = []
            for sid in SCENARIOS:
                b = out["best"]["scenarios"][sid][ds]["mean"]
                f = out["final"]["scenarios"][sid][ds]["mean"]
                rnd = {r["selected_round"] for r in out["final"]["scenarios"][sid]["reps"].values()
                       if r} or {None}
                d = (f - b) if (b is not None and f is not None) else None
                fv["scenarios"].setdefault(sid, {})[ds] = {
                    "r0_mean": b, "final_mean": f, "delta_final_minus_r0": d,
                    "selected_round": sorted(x for x in rnd if x)}
                if d is not None:
                    deltas.append(d)
            fv["macro"][ds] = {
                "mean_delta": float(np.mean(deltas)) if deltas else None,
                "median_delta": float(np.median(deltas)) if deltas else None,
                "improved": int(sum(x > 0.05 for x in deltas)),
                "worsened": int(sum(x < -0.05 for x in deltas)),
                "unchanged": int(sum(abs(x) <= 0.05 for x in deltas)),
                "note": "|Δ|<=0.05 dB treated as unchanged (evaluation nondeterminism)",
            }
        out["final_vs_r0"] = fv

    lines = ["# E6-A full24 — agent run-to-run (same support / training seed)", "",
             "24 scenarios x 2 independent blind-ID runs; fixed Direct Swin+SG.",
             "R0-only tables plus R0→R1→R2 final selection (user-requested extension).",
             "Test: fixed GT-LQ DIV2K 100 / LSDIR 250 (seed-salt v1).", ""]
    ba = blind["summary"]
    lines += ["## Blind-ID agreement (rep1 vs rep2)", "",
              f"- chain exact: **{ba['chain_exact']}/24**; type-only: {ba['chain_type_only']}/24; "
              f"steps equal: {ba['steps_equal']}/24",
              f"- position type: {ba['position_type_rate']:.3f}; spec: {ba['position_spec_rate']:.3f} "
              f"({ba['position_denom']} positions)",
              f"- rep1 vs GT exact: {ba['rep1_vs_gt_chain_exact']}/24; "
              f"rep2 vs GT exact: {ba['rep2_vs_gt_chain_exact']}/24", ""]
    rf = out["round_flow"]
    lines += ["## R0→R1→R2 flow", "",
              f"- R1 triggered: {rf['r1_triggered']}/48; decisions: {rf['r1_decision_files']}; "
              f"CHANGE: {rf['r1_change']}; R1 trained: {rf['r1_trained']}",
              f"- R1 support Δ (R1−R0): mean {fmt(rf['r1_support_delta_mean'],2)} dB, "
              f"median {fmt(rf['r1_support_delta_median'],2)} dB",
              f"- R2 triggered: {rf['r2_triggered']}; R2 trained: {rf['r2_trained']}",
              f"- selected rounds: {rf['selected_round_counts']}", ""]
    for kind in ("best", "last", "final"):
        sec = out.get(kind)
        if not sec:
            continue
        title = {"best": "support-PSNR-selected checkpoint (R0-only)",
                 "last": "fixed final step (R0-only, training jitter)",
                 "final": "best across R0/R1/R2"}[kind]
        lines += [f"## Test PSNR — {title}", "",
                  "| Scenario | GT steps | pipe eq | rep1 | rep2 | mean | SD | range | Δ(rep2-rep1) |",
                  "|---|---:|:--:|---:|---:|---:|---:|---:|---:|"]
        for sid in SCENARIOS:
            r = sec["scenarios"][sid]
            a = r["reps"]["rep1"]; b = r["reps"]["rep2"]
            lines.append(
                f"| {sid} | {r['gt_steps']} | {'Y' if r['pipeline_equal'] else 'N'} | "
                f"{fmt(a['DIV2K'] if a else None)} | {fmt(b['DIV2K'] if b else None)} | "
                f"{fmt(r['DIV2K']['mean'])} | {fmt(r['DIV2K']['sd'])} | "
                f"{fmt(r['DIV2K']['range'])} | "
                f"{fmt(r['DIV2K']['delta_rep2_minus_rep1'],2)} |")
        m = sec["macro"]["DIV2K"]; l = sec["macro"]["LSDIR"]
        lines += ["",
                  f"DIV2K macro mean {fmt(m['mean_of_scenario_means'])} dB; "
                  f"mean within-scenario SD {fmt(m['mean_within_scenario_sd'])}; "
                  f"max range {fmt(m['max_within_scenario_range'])}; "
                  f"Δ mean {fmt(m['delta_mean'],2)} (|Δ|>1 dB: {m['abs_delta_gt_1db']}/24).",
                  f"LSDIR macro mean {fmt(l['mean_of_scenario_means'])} dB; "
                  f"mean within-scenario SD {fmt(l['mean_within_scenario_sd'])}; "
                  f"max range {fmt(l['max_within_scenario_range'])}; "
                  f"Δ mean {fmt(l['delta_mean'],2)} (|Δ|>1 dB: {l['abs_delta_gt_1db']}/24).", ""]

    fixp = OUT / "fixed_lq_sensitivity.json"
    if fixp.exists():
        fx = json.load(open(fixp))
        out["fixed_lq_sensitivity"] = {
            "note": fx.get("note"),
            "macro_delta": fx.get("macro"),
            "corrected_macro": {k: fx.get("corrected_fixed_lq", {}).get(k, {}).get("macro")
                                for k in ("best", "last", "final")},
        }
        lines += ["## Fixed-LQ sensitivity (7 nondeterministic scenarios)", "",
                  "Main pass generated LQ per process for 0003/0005/0017/0018/0020/0022/0023; "
                  "fixed-LQ re-evaluation uses one shared realization.", ""]
        for kind in ("best", "last", "final"):
            m = out["fixed_lq_sensitivity"]["macro_delta"][kind]
            cm = out["fixed_lq_sensitivity"]["corrected_macro"][kind]
            if not m or not cm:
                continue
            lines.append(
                f"- {kind}: Δfixed−orig mean {fmt(m['DIV2K']['mean_delta'],3)} dB "
                f"(max |Δ| {fmt(m['DIV2K']['max_abs_delta'],3)}); "
                f"corrected macro {fmt(cm['DIV2K']['mean_of_scenario_means'])} dB, "
                f"within-SD {fmt(cm['DIV2K']['mean_within_scenario_sd'])}, "
                f"max range {fmt(cm['DIV2K']['max_within_scenario_range'])}")
        lines.append("")

    if "final_vs_r0" in out:
        fv = out["final_vs_r0"]
        lines += ["## Final (R0/R1/R2) vs R0-only", ""]
        for ds in ("DIV2K", "LSDIR"):
            m = fv["macro"][ds]
            lines.append(f"- {ds}: mean Δ(final−R0) {fmt(m['mean_delta'],3)} dB; "
                         f"median {fmt(m['median_delta'],3)}; "
                         f"improved {m['improved']}/24, worsened {m['worsened']}/24, "
                         f"unchanged {m['unchanged']}/24")
        changed = [(sid, fv["scenarios"][sid]["DIV2K"])
                   for sid in SCENARIOS
                   if fv["scenarios"][sid]["DIV2K"]["selected_round"] != ["R0"]]
        lines += ["", "| Scenario | selected | R0 DIV2K | final DIV2K | Δ | R0 LSDIR | final LSDIR | Δ |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for sid, d in changed:
            e = fv["scenarios"][sid]["LSDIR"]
            lines.append(f"| {sid} | {','.join(d['selected_round'])} | "
                         f"{fmt(d['r0_mean'])} | {fmt(d['final_mean'])} | "
                         f"{fmt(d['delta_final_minus_r0'],2)} | "
                         f"{fmt(e['r0_mean'])} | {fmt(e['final_mean'])} | "
                         f"{fmt(e['delta_final_minus_r0'],2)} |")
        lines.append("")
    (OUT / "e6a_full24_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "e6a_full24_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
