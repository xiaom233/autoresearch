#!/usr/bin/env python3
"""Collect compact, traceable extracts for the blind_0003 case report."""
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CASE = Path(__file__).resolve().parents[1]


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def write_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_training_log(path):
    txt = path.read_text(errors="ignore")
    meta = None
    m = re.search(r"=== EXP_META: (\{.*?\}) ===", txt, re.S)
    if m:
        meta = json.loads(m.group(1))
    best = re.findall(r"\[Ckpt @ step (\d+)/\d+\] Set14: PSNR_RGB=([0-9.]+)", txt)
    loaded = re.search(r"Loaded best checkpoint: step (\d+) \(Set14 PSNR_RGB=([0-9.]+)\)", txt)
    summary = {}
    for key in ["best_step", "training_seconds", "num_steps", "num_params_M",
                "batch_size", "learning_rate", "best_train_psnr", "final_loss"]:
        mm = re.search(rf"^{re.escape(key)}:\s*(\S+)", txt, re.M)
        if mm:
            summary[key] = mm.group(1)
    paired = "PairedValDataset: 5 image pairs" in txt
    return {
        "log": str(path.relative_to(ROOT)),
        "exp_meta": meta,
        "checkpoint_curve": [{"step": int(a), "val_psnr_rgb": float(b)} for a, b in best],
        "loaded_best": ({"step": int(loaded.group(1)), "val_psnr_rgb": float(loaded.group(2))}
                        if loaded else None),
        "paired_val_5_support": paired,
        "training_summary": summary,
    }


def main():
    # E1 rule controller: decision, selection, support and test rows for blind_0003
    e1_sel = [r for r in load_jsonl(ROOT / "rebuttal_runs/E1/selection.jsonl")
              if r.get("scenario_id") == "blind_0003"]
    write_json(CASE / "data/rule_reference/E1_selection_blind_0003.json", e1_sel)
    e1_test_sel = [r for r in load_jsonl(ROOT / "rebuttal_runs/E1/test_selection.jsonl")
                   if r.get("scenario_id") == "blind_0003"]
    write_json(CASE / "data/rule_reference/E1_test_selection_blind_0003.json", e1_test_sel)
    e1_support = []
    for p in glob.glob(str(ROOT / "rebuttal_runs/E1/support_metrics/support_gpu*.jsonl")):
        e1_support += [r for r in load_jsonl(p) if r.get("scenario_id") == "blind_0003"]
    (CASE / "data/rule_reference/E1_support_metrics_blind_0003.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in e1_support), encoding="utf-8")
    e1_test = []
    for p in glob.glob(str(ROOT / "rebuttal_runs/E1/test_metrics/rule_gpu*.jsonl")):
        e1_test += [r for r in load_jsonl(p) if r.get("scenario_id") == "blind_0003"]
    (CASE / "data/rule_reference/E1_test_metrics_blind_0003.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in e1_test), encoding="utf-8")

    # A0 AutoIR selected model and rows for blind_0003
    a0_sel = [r for r in load_jsonl(ROOT / "rebuttal_runs/A0/autoair_test_selection.jsonl")
              if r.get("scenario_id") == "blind_0003"]
    write_json(CASE / "data/rule_reference/A0_autoair_selection_blind_0003.json", a0_sel)
    a0_test = []
    for p in glob.glob(str(ROOT / "rebuttal_runs/A0/test_metrics/*.jsonl")):
        a0_test += [r for r in load_jsonl(p)
                    if r.get("scenario_id") == "blind_0003" and r.get("controller") == "AutoIR"]
    (CASE / "metrics/A0_autoair_test_metrics_blind_0003.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in a0_test), encoding="utf-8")

    # E1 summary per-scenario values
    e1_sum = json.load(open(ROOT / "rebuttal_runs/E1/e1_final_summary.json"))
    out = {}
    for ctrl in ["AutoIR", "rule"]:
        out[ctrl] = {}
        for ds in ["DIV2K", "LSDIR"]:
            x = e1_sum["full"][ctrl][ds]
            out[ctrl][ds] = {
                "psnr": x["psnr"], "ssim": x["ssim"], "lpips": x["lpips"],
                "blind_0003_psnr": x["per_scenario_psnr"].get("blind_0003"),
            }
    write_json(CASE / "metrics/E1_rule_vs_AutoIR_blind_0003.json", out)

    # Training log summaries and key extracts
    rounds = {}
    excerpts = []
    for r in range(5):
        p = ROOT / f"exp41/logs/exp41_R{r}_blind_0003.log"
        info = parse_training_log(p)
        rounds[f"R{r}"] = info
        excerpts.append(f"########## R{r}  ({p.relative_to(ROOT)}) ##########\n")
        head = p.read_text(errors="ignore").splitlines()[:12]
        excerpts.append("\n".join(head) + "\n")
        excerpts.append("--- checkpoint lines ---\n")
        for line in p.read_text(errors="ignore").splitlines():
            if "Ckpt @ step" in line or "Loaded best checkpoint" in line:
                excerpts.append(line + "\n")
        excerpts.append("--- final summary ---\n")
        for line in p.read_text(errors="ignore").splitlines()[-40:]:
            if re.match(r"^(best_step|training_seconds|num_steps|num_params_M|batch_size|learning_rate|best_train_psnr|final_loss):", line):
                excerpts.append(line + "\n")
        excerpts.append("\n")
    write_json(CASE / "metrics/training_summary.json", rounds)
    (CASE / "logs/key_log_extract.txt").write_text("".join(excerpts), encoding="utf-8")

    # GT re-eval rows used at each gate / final metric view
    def pick(path, key="blind_0003"):
        d = json.load(open(path))
        return d.get(key) if key in d else d

    metrics = {
        "note": "All PSNR values are RGB. See report.md §4 for checkpoint caveats.",
        "gt_posthoc": json.load(open(ROOT / "exp41/degradation_gt/blind_0003_params.json")),
        "exp41_consolidated_final": {
            f"R{r}": pick(ROOT / f"exp41/results/gt_reeval_R{r}.json") for r in range(5)
        },
        "exp41_737_gate": {
            "R2_19.33_source": pick(ROOT / "exp41/results/gt_reeval_gpu2.json"),
            "R3_24.33_source": pick(ROOT / "exp41/results/gt_reeval_R3_blind_0003.json"),
        },
        "exp41_numeric_last_step15092_from_logs": {
            "note": "Read from exp41/logs/reeval_R*_gpu*.log; the corresponding per-GPU JSON files were partly overwritten by another eval pass and are kept under raw_per_gpu/ for inspection.",
            "R0": {"DIV2K": {"psnr": 25.48, "ssim": 0.7448, "lpips": 0.3161},
                   "LSDIR": {"psnr": 22.27, "ssim": 0.6546, "lpips": 0.3830}},
            "R1": {"DIV2K": {"psnr": 25.32, "ssim": 0.7405, "lpips": 0.3151},
                   "LSDIR": {"psnr": 22.18, "ssim": 0.6501, "lpips": 0.3781}},
            "R2": {"DIV2K": {"psnr": 25.33, "ssim": 0.7400, "lpips": 0.3161},
                   "LSDIR": {"psnr": 22.22, "ssim": 0.6453, "lpips": 0.3812}},
            "R3": {"DIV2K": {"psnr": 26.25, "ssim": 0.7441, "lpips": 0.3326},
                   "LSDIR": {"psnr": 23.01, "ssim": 0.6449, "lpips": 0.3997}},
            "R4": {"DIV2K": {"psnr": 26.52, "ssim": 0.7715, "lpips": 0.3104},
                   "LSDIR": {"psnr": 23.11, "ssim": 0.6800, "lpips": 0.3761}},
        },
        "E1_rule_vs_AutoIR_fixed_lq": out,
        "E1_rule_selection": e1_sel,
        "A0_AutoIR_selection": a0_sel,
    }
    write_json(CASE / "metrics/metrics_provenance.json", metrics)

    # Iteration table (35 entries) from reflection.json
    refl = json.load(open(ROOT / "exp41/challenges/phase4/blind_0003/reflection.json"))
    rows = ["| # | 轮次 | 假设管线 | 仿真 PSNR | CI | 决策 | 归因类别 | 依据/备注 |",
            "|---:|:--:|---|---:|---:|---|---|---|"]
    # Hand-written attribution labels, keyed by iteration number.
    label = {
        1: ("R+L", "gaussian 主假设；gm_ratio=0.532；PSNR 模式；30-40dB → UNCERTAIN"),
        2: ("R", "severity 扫描（SKILL 强制）"),
        3: ("R", "severity 扫描（SKILL 强制）"),
        4: ("L", "dct_zero_ratio=0.624 触发 JPEG 假设；PSNR 29.76 拒绝"),
        5: ("R", "severity 扫描；更重 blur 不优"),
        6: ("R", "SKILL 要求至少测试 3 种 blur；motion 仅 25.76"),
        7: ("L", "noise 假设（sigma/spatial_corr）；PSNR 21.40 拒绝"),
        8: ("R", "blur 跨族测试（lens）；34.61 接近 gaussian"),
        9: ("L", "违反物理顺序的对照（JPEG→blur）；31.20 拒绝"),
        10: ("L", "JPEG severity 细化；32.04 拒绝"),
        11: ("L", "R1 自主补测未测过的 glass：36.65/10/10 = GT 类型"),
        12: ("R", "glass severity 扫描"),
        13: ("R", "glass severity 扫描"),
        14: ("R", "verify_signals 强制验证 → MATCH(2✓)"),
        15: ("L", "gamma 假设（cross_channel_corr=0.731）；22.79 拒绝"),
        16: ("L", "gamma 在 blur 后；22.71 拒绝"),
        17: ("L", "gamma+glass；22.88 拒绝"),
        18: ("L", "JPEG sev=2 补测；31.16 拒绝"),
        19: ("L", "JPEG sev=4 补测；28.09 拒绝"),
        20: ("L", "contrast 假设（variance_ratio=0.954）；16.84 拒绝"),
        21: ("L", "R4 重新验证 glass 基线；36.58/10/10"),
        22: ("L", "glass+JPEG:1；33.10 拒绝"),
        23: ("L", "glass+JPEG:2；32.02 拒绝"),
        24: ("L", "glass+JPEG:3；30.38 拒绝"),
        25: ("L", "glass+JPEG:4；28.52 拒绝"),
        26: ("L", "glass+JPEG:5；26.04 拒绝，JPEG 假阳性封闭"),
        27: ("L", "glass severity 4；30.45 拒绝"),
        28: ("L", "glass severity 5；28.60 拒绝"),
        29: ("L", "glass+YCrCb 噪声；21.47 拒绝"),
        30: ("L", "glass+RGB 噪声；25.72 拒绝"),
        31: ("L", "lens:3；34.60 低于 glass"),
        32: ("L", "motion:1；26.70 拒绝"),
        33: ("L!", "双 blur glass:2+gaussian:1；37.30 但 CI 6/10；违反每次最多一次 blur 的管线规则"),
        34: ("L!", "反向双 blur；36.75 CI 7/10；同样违反管线规则"),
        35: ("L", "lens:2 复测；34.61"),
    }
    for it in refl["iterations"]:
        n = it["round"]
        hyp = it["hypothesis"]["pipeline"]
        tr = it.get("test_result", {})
        psnr = tr.get("psnr_rgb", "")
        ci = tr.get("ci_pass", tr.get("verdict", ""))
        dec = it.get("decision", "")
        cat, note = label.get(n, ("?", ""))
        rows.append(f"| {n} | {'R0' if n <= 10 else ('R1' if n <= 14 else ('R2' if n <= 20 else 'R4'))} | "
                    f"`{hyp}` | {psnr} | {ci} | {dec} | {cat} | {note} |")
    (CASE / "data/challenge/iterations_table.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("wrote extracts to", CASE)


if __name__ == "__main__":
    main()
