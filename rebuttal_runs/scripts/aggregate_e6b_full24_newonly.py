#!/usr/bin/env python3
"""Aggregate E6-B full24 setA/setB (new experiments only).

Primary: 24 scenarios x {setA,setB}, scenario-equal macro, within-scenario SD/range.
Reference column: E6-A final (DIV2K_valid support; different support source, labeled).
exp41 old chain is NOT included.
"""
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rebuttal_runs/E6B_full24"
E6A = ROOT / "rebuttal_runs/E6A_full24/e6a_full24_summary.json"
SCEN = [f"blind_{i:04d}" for i in range(1, 25)]


def gt_steps(sid):
    return len(json.load(open(ROOT / f"exp41/degradation_gt/{sid}_params.json"))["pipeline"])


def load_metrics2():
    by = defaultdict(lambda: defaultdict(list))
    ss = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(OUT / "test_metrics_final_gpu*.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["scenario_id"], r["support_set"])
            by[key][r["dataset"]].append(r["psnr_rgb"])
            ss[key][r["dataset"]].append(r["ssim_rgb"])
    return by, ss


def main():
    by, ss = load_metrics2()
    missing = [(sid, s) for sid in SCEN for s in ("setA", "setB") if not by[(sid, s)]["DIV2K"]]
    if missing:
        raise SystemExit(f"missing metrics: {missing}")

    table = {}
    rows = []
    for sid in SCEN:
        rec = {"gt_steps": gt_steps(sid), "sets": {}}
        for s in ("setA", "setB"):
            e = {}
            for ds in ("DIV2K", "LSDIR"):
                e[ds] = float(np.mean(by[(sid, s)][ds]))
                e[f"{ds}_ssim"] = float(np.mean(ss[(sid, s)][ds]))
                e[f"n_{ds}"] = len(by[(sid, s)][ds])
            rec["sets"][s] = e
            rows.append({"scenario": sid, "gt_steps": rec["gt_steps"], "set": s,
                         "DIV2K_PSNR": e["DIV2K"], "LSDIR_PSNR": e["LSDIR"],
                         "DIV2K_SSIM": e["DIV2K_ssim"], "LSDIR_SSIM": e["LSDIR_ssim"]})
        for ds in ("DIV2K", "LSDIR"):
            rec[f"sd_{ds}"] = float(np.std([rec["sets"][s][ds] for s in ("setA", "setB")], ddof=1))
            rec[f"range_{ds}"] = float(abs(rec["sets"]["setA"][ds] - rec["sets"]["setB"][ds]))
            rec[f"mean_{ds}"] = float(np.mean([rec["sets"][s][ds] for s in ("setA", "setB")]))
        table[sid] = rec

    out = {"note": "E6-B full24, new experiments only: 24 scenarios x {setA,setB}, each 5 random "
                   "DIV2K_train supports, full LLM pipeline. Macro = 24 scenarios equal weight. "
                   "E6-A reference uses DIV2K_valid support (different source); not directly mergeable. "
                   "exp41 old chain excluded.",
           "macro": {}, "within_scenario": {}, "selected_rounds": {}, "stratified_by_gt_steps": {},
           "scenarios": table}

    sel = [json.loads(l) for l in open(OUT / "selection_final.jsonl") if l.strip()]
    cnt = Counter((r["support_set"], r["selected_round"]) for r in sel)
    out["selected_rounds"] = {f"{s}_{rd}": int(cnt.get((s, rd), 0))
                              for s in ("setA", "setB") for rd in ("R0", "R1", "R2")}

    for ds in ("DIV2K", "LSDIR"):
        out["macro"][ds] = {}
        for s in ("setA", "setB", "mean"):
            key = "mean" if s == "mean" else None
            vals = [table[sid][f"mean_{ds}"] if s == "mean" else table[sid]["sets"][s][ds] for sid in SCEN]
            out["macro"][ds][s] = {"mean": float(np.mean(vals)),
                                   "sd": float(np.std(vals, ddof=1)),
                                   "median": float(np.median(vals)),
                                   "min": float(np.min(vals)), "max": float(np.max(vals))}
        sd = [table[sid][f"sd_{ds}"] for sid in SCEN]
        rg = [table[sid][f"range_{ds}"] for sid in SCEN]
        out["within_scenario"][ds] = {
            "mean_sd": float(np.mean(sd)), "median_sd": float(np.median(sd)),
            "max_sd": float(np.max(sd)), "mean_range": float(np.mean(rg)),
            "median_range": float(np.median(rg)), "max_range": float(np.max(rg)),
            "worst_scenario": SCEN[int(np.argmax(rg))]}
        d = np.array([table[sid]["sets"]["setB"][ds] - table[sid]["sets"]["setA"][ds] for sid in SCEN])
        rng = np.random.default_rng(20261030)
        boots = [float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(10000)]
        out["within_scenario"][ds]["setB_minus_setA"] = {
            "mean": float(np.mean(d)), "median": float(np.median(d)),
            "sd": float(np.std(d, ddof=1)),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "wins_setB": int(np.sum(d > 0)), "wins_setA": int(np.sum(d < 0)), "n": len(d)}

    for gs in (1, 2, 3):
        group = [sid for sid in SCEN if table[sid]["gt_steps"] == gs]
        out["stratified_by_gt_steps"][str(gs)] = {"n_scenarios": len(group)}
        for ds in ("DIV2K", "LSDIR"):
            out["stratified_by_gt_steps"][str(gs)][ds] = {
                "setA": float(np.mean([table[sid]["sets"]["setA"][ds] for sid in group])),
                "setB": float(np.mean([table[sid]["sets"]["setB"][ds] for sid in group])),
                "mean": float(np.mean([table[sid][f"mean_{ds}"] for sid in group])),
                "mean_range": float(np.mean([table[sid][f"range_{ds}"] for sid in group]))}

    # E6-A reference (different support source)
    e6a = json.load(open(E6A))["final"]["scenarios"]
    out["e6a_reference"] = {"support_source": "DIV2K_valid (test split)", "macro": {}}
    for ds in ("DIV2K", "LSDIR"):
        vals = [np.mean([e6a[sid]["reps"][rep][ds] for rep in ("rep1", "rep2")]) for sid in SCEN]
        out["e6a_reference"]["macro"][ds] = float(np.mean(vals))

    (OUT / "e6b_full24_newonly_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# E6-B full24 结果（新实验，setA/setB）", "",
             "24 场景 × {setA,setB}，每组 5 张随机 `DIV2K_train` support；完整 LLM 流程；"
             "固定 GT-LQ test（DIV2K 100 + LSDIR 250）。宏平均 = 24 场景等权。", "",
             "| 数据集 | setA | setB | 两组合并 mean | 场景间 SD |",
             "|---|---:|---:|---:|---:|"]
    for ds in ("DIV2K", "LSDIR"):
        m = out["macro"][ds]
        lines.append(f"| {ds} | {m['setA']['mean']:.3f} | {m['setB']['mean']:.3f} | "
                     f"{m['mean']['mean']:.3f} | {m['mean']['sd']:.3f} |")
    lines += ["", "## within-scenario（support-set 差异）", "",
              "| 数据集 | mean SD | median SD | max SD | mean range | median range | max range |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for ds in ("DIV2K", "LSDIR"):
        w = out["within_scenario"][ds]
        lines.append(f"| {ds} | {w['mean_sd']:.3f} | {w['median_sd']:.3f} | {w['max_sd']:.3f} | "
                     f"{w['mean_range']:.3f} | {w['median_range']:.3f} | {w['max_range']:.3f} |")
    lines += ["", "setB − setA（场景配对，描述性）："]
    for ds in ("DIV2K", "LSDIR"):
        d = out["within_scenario"][ds]["setB_minus_setA"]
        lines.append(f"- {ds}: {d['mean']:+.3f} dB "
                     f"[{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]，setB 胜 {d['wins_setB']}/24")
    lines += ["", "## 选点轮次", "",
              f"- setA: {out['selected_rounds']['setA_R0']} R0 / {out['selected_rounds']['setA_R1']} R1 / {out['selected_rounds']['setA_R2']} R2",
              f"- setB: {out['selected_rounds']['setB_R0']} R0 / {out['selected_rounds']['setB_R1']} R1 / {out['selected_rounds']['setB_R2']} R2",
              "", "## E6-A 参考（不同 support 来源）", "",
              f"E6-A 使用 DIV2K_valid support（与 test 同批图），仅作 run-to-run 参考："
              f"DIV2K {out['e6a_reference']['macro']['DIV2K']:.3f} / LSDIR {out['e6a_reference']['macro']['LSDIR']:.3f}。"
              "不能与 setA/setB 无标注合并。", "",
              "## 分层（按 GT 步数）", ""]
    for gs in ("1", "2", "3"):
        s = out["stratified_by_gt_steps"][gs]
        lines.append(f"- {gs} 步（n={s['n_scenarios']}）：DIV2K {s['DIV2K']['mean']:.3f}"
                     f"（range {s['DIV2K']['mean_range']:.3f}）；LSDIR {s['LSDIR']['mean']:.3f}"
                     f"（range {s['LSDIR']['mean_range']:.3f}）")
    (OUT / "e6b_full24_newonly_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
