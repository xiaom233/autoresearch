#!/usr/bin/env python3
"""E5a: recompute degradation-identification metrics from public predictions.

Inputs are aggregate prediction files only. For exp41, reads the actual
training params recorded in the A0 manifest plus GT params (evaluation side
only; never used for model/round selection).
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PAT = re.compile(r"([a-zA-Z0-9_]+)\((\d+)\)")


def parse_pipe(s):
    s = (s or "").strip()
    if not s:
        return []
    out = []
    for part in s.split("|"):
        m = PAT.search(part.strip())
        if m:
            out.append((m.group(1), int(m.group(2))))
    return out


def parse_params(path):
    d = json.load(open(path))
    return [(s["function"], int(s.get("severity", 1))) for s in d.get("pipeline", [])]


def multiset_match(gt, pred, with_sev):
    """Greedy one-to-one matching; returns matched pairs list."""
    pool = list(pred)
    matched = []
    for g in gt:
        for j, p in enumerate(pool):
            if p[0] == g[0] and (not with_sev or p[1] == g[1]):
                matched.append((g, p))
                pool.pop(j)
                break
    return matched


def metrics_one(gt, pred):
    n_gt, n_pred = len(gt), len(pred)
    pos_type = sum(1 for i in range(min(n_gt, n_pred)) if gt[i][0] == pred[i][0])
    pos_spec = sum(1 for i in range(min(n_gt, n_pred)) if gt[i] == pred[i])
    exact_type = int(n_gt == n_pred and all(a[0] == b[0] for a, b in zip(gt, pred)))
    exact_spec = int(gt == pred)
    step_ok = int(n_gt == n_pred)
    mt = multiset_match(gt, pred, with_sev=False)
    ms = multiset_match(gt, pred, with_sev=True)
    tp_t, tp_s = len(mt), len(ms)
    p_t = tp_t / n_pred if n_pred else 0.0
    r_t = tp_t / n_gt if n_gt else 0.0
    p_s = tp_s / n_pred if n_pred else 0.0
    r_s = tp_s / n_gt if n_gt else 0.0
    f1_t = 2 * p_t * r_t / (p_t + r_t) if (p_t + r_t) else 0.0
    f1_s = 2 * p_s * r_s / (p_s + r_s) if (p_s + r_s) else 0.0
    sev_mae = (sum(abs(g[1] - p[1]) for g, p in mt) / len(mt)) if mt else None
    return {
        "gt_steps": n_gt, "pred_steps": n_pred,
        "pos_type": pos_type, "pos_spec": pos_spec,
        "exact_type": exact_type, "exact_spec": exact_spec,
        "step_ok": step_ok,
        "over": max(0, n_pred - n_gt), "under": max(0, n_gt - n_pred),
        "multiset_type_p": p_t, "multiset_type_r": r_t, "multiset_type_f1": f1_t,
        "multiset_spec_p": p_s, "multiset_spec_r": r_s, "multiset_spec_f1": f1_s,
        "sev_mae_matched_func": sev_mae,
    }


def aggregate(rows):
    n = len(rows)
    gt_total = sum(r["gt_steps"] for r in rows)
    out = {
        "challenges": n,
        "gt_steps_total": gt_total,
        "positional_type_micro_pct": 100 * sum(r["pos_type"] for r in rows) / gt_total,
        "positional_spec_micro_pct": 100 * sum(r["pos_spec"] for r in rows) / gt_total,
        "positional_type_macro_pct": 100 * sum(r["pos_type"] / r["gt_steps"] for r in rows) / n,
        "positional_spec_macro_pct": 100 * sum(r["pos_spec"] / r["gt_steps"] for r in rows) / n,
        "exact_type_pct": 100 * sum(r["exact_type"] for r in rows) / n,
        "exact_spec_pct": 100 * sum(r["exact_spec"] for r in rows) / n,
        "step_accuracy_pct": 100 * sum(r["step_ok"] for r in rows) / n,
        "over_steps": sum(r["over"] for r in rows),
        "under_steps": sum(r["under"] for r in rows),
        "challenges_over": sum(1 for r in rows if r["over"] > 0),
        "challenges_under": sum(1 for r in rows if r["under"] > 0),
        "multiset_type_pct": 100 * sum(r["multiset_type_p"] for r in rows) / n,
        "multiset_type_recall_pct": 100 * sum(r["multiset_type_r"] for r in rows) / n,
        "multiset_type_f1_pct": 100 * sum(r["multiset_type_f1"] for r in rows) / n,
        "multiset_spec_pct": 100 * sum(r["multiset_spec_p"] for r in rows) / n,
        "multiset_spec_recall_pct": 100 * sum(r["multiset_spec_r"] for r in rows) / n,
        "multiset_spec_f1_pct": 100 * sum(r["multiset_spec_f1"] for r in rows) / n,
    }
    mae = [r["sev_mae_matched_func"] for r in rows if r["sev_mae_matched_func"] is not None]
    out["sev_mae_matched_func"] = sum(mae) / len(mae) if mae else None
    return out


def exp40(csv_path):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        lines = [l for l in f if l.strip() and not l.lstrip('"').startswith("#")]
    rows = list(csv.DictReader(lines))
    methods = {
        "Random": "Random_Pred", "Qwen": "Qwen_Pred",
        "R0": "R0_Pred", "R1": "R1_Pred",
    }
    out = {"source": str(csv_path), "methods": {}, "note": "Ours column has no pipeline; only old positional scores are available."}
    per = {k: [] for k in methods}
    ours_old = []
    for row in rows:
        gt = parse_pipe(row["GT"])
        for name, col in methods.items():
            per[name].append(metrics_one(gt, parse_pipe(row[col])))
        ours_old.append({
            "challenge": row["Challenge"],
            "gt_steps": int(row["GT_steps"]),
            "ours_type": float(row["Ours_Type"].split("/")[0]),
            "ours_spec": float(row["Ours_Spec"].split("/")[0]),
        })
    for name, rs in per.items():
        out["methods"][name] = aggregate(rs)
    out["ours_reported_best_of_r0_r1"] = {
        "positional_type_micro_pct": 100 * sum(r["ours_type"] for r in ours_old) / sum(r["gt_steps"] for r in ours_old),
        "positional_spec_micro_pct": 100 * sum(r["ours_spec"] for r in ours_old) / sum(r["gt_steps"] for r in ours_old),
        "selection_basis": "UNKNOWN — file only states per-challenge best of R0/R1",
    }
    return out


def exp41(manifest, gt_dir, only_rounds=("R0", "R1", "R2")):
    rows = [json.loads(l) for l in open(manifest)]
    by_round = {}
    for row in rows:
        if row["round"] not in only_rounds or not row.get("params_path"):
            continue
        gt = parse_params(ROOT / "exp41" / "degradation_gt" / f"{row['scenario_id']}_params.json")
        pred = parse_params(ROOT / row["params_path"])
        by_round.setdefault(row["round"], []).append(metrics_one(gt, pred))
    return {
        "source_manifest": str(manifest),
        "gt_dir": "exp41/degradation_gt",
        "rounds": {r: aggregate(v) for r, v in by_round.items()},
    }


def fmt_table(d):
    lines = ["| Method/Round | Challenges | Pos Type micro | Pos Spec micro | Exact Type | Exact Spec | Step Acc | Multi Type F1 | Multi Spec F1 | Over/Under | Sev MAE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in d.items():
        lines.append(
            f"| {name} | {m['challenges']} | {m['positional_type_micro_pct']:.2f}% | "
            f"{m['positional_spec_micro_pct']:.2f}% | {m['exact_type_pct']:.2f}% | "
            f"{m['exact_spec_pct']:.2f}% | {m['step_accuracy_pct']:.2f}% | "
            f"{m['multiset_type_f1_pct']:.2f}% | {m['multiset_spec_f1_pct']:.2f}% | "
            f"{m['over_steps']}/{m['under_steps']} | "
            f"{m['sev_mae_matched_func']:.2f} |" if m['sev_mae_matched_func'] is not None else "")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp40-csv", default="benchmark_results/blind_id_detail.csv")
    ap.add_argument("--exp41-manifest", default="rebuttal_runs/A0/exp41_selection_manifest.jsonl")
    ap.add_argument("--out-dir", default="rebuttal_runs/E5a")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    e40 = exp40(ROOT / args.exp40_csv)
    e41 = exp41(ROOT / args.exp41_manifest, ROOT / "exp41/degradation_gt")
    json.dump(e40, open(out / "exp40_dc_metrics.json", "w"), indent=2, ensure_ascii=False)
    json.dump(e41, open(out / "exp41_dc_metrics.json", "w"), indent=2, ensure_ascii=False)
    md = ["# E5a DC metric recomputation", "",
          "## exp40 (public blind_id_detail.csv)", "", fmt_table(e40["methods"]), "",
          f"Ours (reported best-of-R0/R1): positional Type {e40['ours_reported_best_of_r0_r1']['positional_type_micro_pct']:.2f}%, "
          f"Spec {e40['ours_reported_best_of_r0_r1']['positional_spec_micro_pct']:.2f}%; selection basis UNKNOWN.", "",
          "## exp41 (actual training params, evaluation-side GT)", "", fmt_table(e41["rounds"]), ""]
    (out / "E5a_DC_metrics.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
