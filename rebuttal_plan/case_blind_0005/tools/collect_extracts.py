#!/usr/bin/env python3
"""Collect compact, traceable extracts for the blind_0005 case report."""
import glob
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CASE = Path(__file__).resolve().parents[1]


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def write_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def mean_metric(rows, ds):
    rs = [r for r in rows if r["dataset"] == ds]
    return {
        "n": len(rs),
        "psnr": float(np.mean([r["psnr_rgb"] for r in rs])),
        "ssim": float(np.mean([r["ssim_rgb"] for r in rs])),
        "lpips": float(np.mean([r["lpips_vgg"] for r in rs])),
    }


def parse_training_log(path):
    txt = path.read_text(errors="ignore")
    meta = None
    m = re.search(r"=== EXP_META: (\{.*?\}) ===", txt, re.S)
    if m:
        meta = json.loads(m.group(1))
    curve = re.findall(r"\[Ckpt @ step (\d+)/\d+\] Set14: PSNR_RGB=([0-9.]+)", txt)
    loaded = re.search(r"Loaded best checkpoint: step (\d+) \(Set14 PSNR_RGB=([0-9.]+)\)", txt)
    summary = {}
    for key in ["best_step", "training_seconds", "num_steps", "num_params_M",
                "batch_size", "learning_rate", "best_train_psnr", "final_loss"]:
        mm = re.search(rf"^{re.escape(key)}:\s*(\S+)", txt, re.M)
        if mm:
            summary[key] = mm.group(1)
    return {
        "log": str(path.relative_to(ROOT)),
        "exp_meta": meta,
        "checkpoint_curve": [{"step": int(a), "val_psnr_rgb": float(b)} for a, b in curve],
        "loaded_best": ({"step": int(loaded.group(1)), "val_psnr_rgb": float(loaded.group(2))}
                        if loaded else None),
        "paired_val_5_support": "PairedValDataset: 5 image pairs" in txt,
        "training_summary": summary,
    }


def main():
    # E1 rule controller extracts for blind_0005
    e1_sel = [r for r in load_jsonl(ROOT / "rebuttal_runs/E1/selection.jsonl")
              if r.get("scenario_id") == "blind_0005"]
    write_json(CASE / "data/rule_reference/E1_selection_blind_0005.json", e1_sel)
    e1_support = []
    for p in glob.glob(str(ROOT / "rebuttal_runs/E1/support_metrics/support_gpu*.jsonl")):
        e1_support += [r for r in load_jsonl(p) if r.get("scenario_id") == "blind_0005"]
    (CASE / "data/rule_reference/E1_support_metrics_blind_0005.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in e1_support), encoding="utf-8")
    e1_test = []
    for p in glob.glob(str(ROOT / "rebuttal_runs/E1/test_metrics/rule_gpu*.jsonl")):
        e1_test += [r for r in load_jsonl(p) if r.get("scenario_id") == "blind_0005"]
    (CASE / "metrics/E1_rule_test_metrics_blind_0005.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in e1_test), encoding="utf-8")

    # A0 AutoIR extracts for blind_0005
    a0_sel = [r for r in load_jsonl(ROOT / "rebuttal_runs/A0/autoair_test_selection.jsonl")
              if r.get("scenario_id") == "blind_0005"]
    write_json(CASE / "data/rule_reference/A0_autoair_selection_blind_0005.json", a0_sel)
    a0_test = []
    for p in glob.glob(str(ROOT / "rebuttal_runs/A0/test_metrics/*.jsonl")):
        a0_test += [r for r in load_jsonl(p)
                    if r.get("scenario_id") == "blind_0005" and r.get("controller") == "AutoIR"]
    (CASE / "metrics/A0_autoair_test_metrics_blind_0005.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in a0_test), encoding="utf-8")

    # E1 summary numbers
    e1_sum = json.load(open(ROOT / "rebuttal_runs/E1/e1_final_summary.json"))
    comparison = {}
    for ctrl in ["AutoIR", "rule"]:
        comparison[ctrl] = {}
        for ds in ["DIV2K", "LSDIR"]:
            x = e1_sum["full"][ctrl][ds]
            comparison[ctrl][ds] = {
                "psnr": x["psnr"], "ssim": x["ssim"], "lpips": x["lpips"],
                "blind_0005_psnr": x["per_scenario_psnr"].get("blind_0005"),
            }
    write_json(CASE / "metrics/E1_rule_vs_AutoIR_blind_0005.json", comparison)

    # Training logs R0-R2 (R3/R4 are symlinks to R2)
    rounds = {}
    excerpts = []
    for r in range(3):
        p = ROOT / f"exp41/logs/exp41_R{r}_blind_0005.log"
        info = parse_training_log(p)
        info["experiment_dir"] = f"exp41/experiments/exp41_R{r}_blind_0005"
        info["symlink"] = False
        rounds[f"R{r}"] = info
        excerpts.append(f"########## R{r}  ({p.relative_to(ROOT)}) ##########\n")
        excerpts.append("\n".join(p.read_text(errors="ignore").splitlines()[:12]) + "\n")
        excerpts.append("--- checkpoint lines ---\n")
        for line in p.read_text(errors="ignore").splitlines():
            if "Ckpt @ step" in line or "Loaded best checkpoint" in line:
                excerpts.append(line + "\n")
        excerpts.append("--- final summary ---\n")
        for line in p.read_text(errors="ignore").splitlines()[-40:]:
            if re.match(r"^(best_step|training_seconds|num_steps|num_params_M|batch_size|learning_rate|best_train_psnr|final_loss):", line):
                excerpts.append(line + "\n")
        excerpts.append("\n")
    for r in (3, 4):
        rounds[f"R{r}"] = {
            "experiment_dir": f"exp41/experiments/exp41_R{r}_blind_0005",
            "symlink_to": "exp41_R2_blind_0005" if r == 3 else "exp41_R3_blind_0005",
            "symlink": True,
            "note": "no new training; evaluation copies R2",
        }
    write_json(CASE / "metrics/training_summary.json", rounds)
    (CASE / "logs/key_log_extract.txt").write_text("".join(excerpts), encoding="utf-8")

    # Metrics provenance
    def pick(path, key="blind_0005"):
        d = json.load(open(path))
        return d.get(key) if key in d else d

    metrics = {
        "note": "All PSNR values are RGB. Exp41 gt_reeval_R* uses step7546 (lexicographic last) in the consolidated summary; A0 AutoIR uses R1 step15092; E1 rule uses R1 step7546. See report §3/§8.",
        "gt_posthoc": json.load(open(ROOT / "exp41/degradation_gt/blind_0005_params.json")),
        "exp41_consolidated_final": {
            f"R{r}": pick(ROOT / f"exp41/results/gt_reeval_R{r}.json") for r in range(5)
        },
        "exp41_737_gate": pick(ROOT / "exp41/results/gt_reeval.json"),
        "A0_AutoIR_fixed_lq": {
            "DIV2K": mean_metric(a0_test, "DIV2K"),
            "LSDIR": mean_metric(a0_test, "LSDIR"),
        },
        "E1_rule_fixed_lq": {
            "DIV2K": mean_metric(e1_test, "DIV2K"),
            "LSDIR": mean_metric(e1_test, "LSDIR"),
        },
        "E1_rule_selection": e1_sel,
        "A0_AutoIR_selection": a0_sel,
        "exp40_final_strategy": pick(ROOT / "exp41/results/exp40_final_strategy.json"),
    }
    write_json(CASE / "metrics/metrics_provenance.json", metrics)

    # Iteration table (28 persisted entries)
    refl = json.load(open(ROOT / "exp41/challenges/phase4/blind_0005/reflection.json"))
    rows = ["| # | 轮次 | 假设管线 | 测试结果 | 决策 | 归因 |",
            "|---:|:--:|---|---|---|---|"]
    labels = {
        1: ("R0", "L", "初始 global 假设；PSNR 12.21 / CI 7/10"),
        2: ("R0", "L", "brightness alone；12.79 / 6/10"),
        3: ("R0", "L", "gamma brightness；13.07 / 5/10"),
        4: ("R0", "L", "saturation alone；12.33 / 7/10"),
        5: ("R0", "L", "contrast stretch；10.73 / 3/10"),
        6: ("R0", "R+L", "impulse alone；12.86 / 8/10（规则要求扫 noise）"),
        7: ("R0", "L", "saturation YCrCb；8.41 / 7/10"),
        8: ("R0", "L", "RGB brightness + sat；12.17 / 7/10"),
        9: ("R0", "R", "impulse severity 1（severity 扫描）"),
        10: ("R0", "R", "impulse severity 3（severity 扫描）"),
        11: ("R0", "L", "impulse:3 + brightness:1 反向组合；12.14 / 10/10"),
        12: ("R0", "L", "brightness:1 + impulse:3；12.13 / 10/10"),
        13: ("R0", "L", "brightness severity 2；11.52 / 9/10"),
        14: ("R1", "L", "gamma+impulse；WEAK"),
        15: ("R1", "L", "saturation+impulse；WEAK"),
        16: ("R1", "L", "baseline signal re-verify；WEAK"),
        17: ("R2", "L", "brightness+sat3+impulse；WEAK"),
        18: ("R2", "L", "impulse:4+brightness；WEAK"),
        19: ("R2", "L", "impulse:4+brightness 反向；WEAK"),
        20: ("R2", "L", "impulse:3+brightness 反向；WEAK"),
        21: ("R2", "L", "brightness+sat2+impulse；WEAK"),
        22: ("R2", "L", "baseline 复核；WEAK"),
        23: ("R4", "L", "sat5+impulse；11.11 / 7/10"),
        24: ("R4", "L", "brightness+sat5；12.21 / 7/10"),
        25: ("R4", "L", "brightness+sat5+impulse；11.09 / 8/10"),
        26: ("R4", "L", "JPEG-only；13.55 / 5/10"),
        27: ("R4", "L", "blur+impulse+brightness；11.71 / 6/10"),
        28: ("R4", "L", "quantization+brightness+impulse；8.42 / 3/10"),
    }
    for it in refl["iterations"]:
        n = it["round"]
        hyp = it["hypothesis"]["pipeline"]
        tr = it.get("test_result", {})
        result = []
        if "psnr_rgb" in tr:
            result.append(f"PSNR {tr['psnr_rgb']}")
        if "ci_pass" in tr:
            result.append(f"CI {tr['ci_pass']}")
        if "verdict" in tr:
            result.append(f"signal {tr['verdict']}")
        rnd, cat, note = labels.get(n, ("?", "?", ""))
        rows.append(f"| {n} | {rnd} | `{hyp}` | {' / '.join(result)} | {it['decision']} | {cat}：{note} |")
    (CASE / "data/challenge/iterations_table.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("wrote extracts to", CASE)


if __name__ == "__main__":
    main()
