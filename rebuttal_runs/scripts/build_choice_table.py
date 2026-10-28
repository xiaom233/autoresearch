#!/usr/bin/env python3
"""Compare LLM/AutoIR vs rule-controller degradation choices per scenario."""
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rebuttal_runs/E1_oneshot/choice_table.json"
MD = ROOT / "rebuttal_plan/LLM_vs_rule_choices.md"


def cat(f):
    for p in ("compression_", "noise_", "blur_", "contrast_", "brightness_", "saturate_"):
        if f.startswith(p):
            return "saturation" if p == "saturate_" else p[:-1]
    return f.split("_")[0]


def pipe(meta):
    return [(x[0], int(x[1])) for x in meta["degradation_pipeline"]]


def load_params(path):
    d = json.load(open(path))
    return [(x["function"], int(x["severity"])) for x in d["pipeline"]]


def fmt(p):
    return " → ".join(f"`{f}:{s}`" for f, s in p)


def arch(meta):
    m = meta["model"]
    a = "OCAB" if m.get("attention_type") == "ocab" else "Swin"
    bits = [a]
    if m.get("simple_gate"):
        bits.append("SG")
    if m.get("gcm"):
        bits.append("GCM")
    if m.get("color_pre"):
        bits.append("ColorPre")
    if m.get("dual_branch"):
        bits.append("DualBranch")
    return "+".join(bits)


def load_test(pattern, ctrl):
    rows = defaultdict(lambda: defaultdict(list))
    for p in glob.glob(str(ROOT / pattern)):
        for line in open(p):
            r = json.loads(line)
            if r.get("controller") != ctrl:
                continue
            rows[r["scenario_id"]][r["dataset"]].append(r)
    out = {}
    for sid, by_ds in rows.items():
        out[sid] = {}
        for ds, rs in by_ds.items():
            out[sid][ds] = {
                "psnr": sum(x["psnr_rgb"] for x in rs) / len(rs),
                "n": len(rs),
            }
    return out


def match(p, gt):
    if p == gt:
        return "exact"
    if [f for f, _ in p] == [f for f, _ in gt]:
        return "type"
    if [cat(f) for f, _ in p] == [cat(f) for f, _ in gt]:
        return "category"
    if len(p) == len(gt) and Counter(cat(f) for f, _ in p) == Counter(cat(f) for f, _ in gt):
        return "order"
    if len(p) == len(gt):
        return "same_len"
    return "no"


def diff_desc(llm, rule):
    if llm == rule:
        return "相同"
    bits = []
    cl, cr = Counter(f for f, _ in llm), Counter(f for f, _ in rule)
    only_l = cl - cr
    only_r = cr - cl
    if only_l:
        bits.append("LLM 多 " + ", ".join(f"{f}:{n}" for f, n in only_l.items()))
    if only_r:
        bits.append("rule 多 " + ", ".join(f"{f}:{n}" for f, n in only_r.items()))
    sev = []
    for f in sorted(set(cl) & set(cr)):
        sl = [s for ff, s in llm if ff == f]
        sr = [s for ff, s in rule if ff == f]
        if sl and sr and sl[0] != sr[0]:
            sev.append(f"{f} {sl[0]} vs {sr[0]}")
    if sev:
        bits.append("severity: " + "; ".join(sev))
    # subtype differences by category
    sub = []
    for c in sorted({cat(f) for f, _ in llm} & {cat(f) for f, _ in rule}):
        fl = [f for f, _ in llm if cat(f) == c]
        fr = [f for f, _ in rule if cat(f) == c]
        if fl and fr and fl != fr:
            sub.append(f"{c}: {fl[0]} vs {fr[0]}")
    if sub:
        bits.append("子类型: " + "; ".join(sub))
    if [f for f, _ in llm] != [f for f, _ in rule] and \
            Counter(f for f, _ in llm) == Counter(f for f, _ in rule):
        bits.append("顺序不同")
    if not bits:
        bits.append("类型/步数不同")
    return "；".join(bits)


def tags_of(llm, rule):
    tags = set()
    if llm == rule:
        return tags
    if len(llm) != len(rule):
        tags.add("step_count")
    if [f for f, _ in llm] != [f for f, _ in rule] and \
            Counter(f for f, _ in llm) == Counter(f for f, _ in rule):
        tags.add("order")
    cl, cr = Counter(f for f, _ in llm), Counter(f for f, _ in rule)
    for f in set(cl) & set(cr):
        sl = [s for ff, s in llm if ff == f]
        sr = [s for ff, s in rule if ff == f]
        if sl and sr and sl[0] != sr[0]:
            tags.add("severity")
    for c in {cat(f) for f, _ in llm} & {cat(f) for f, _ in rule}:
        fl = [f for f, _ in llm if cat(f) == c]
        fr = [f for f, _ in rule if cat(f) == c]
        if fl and fr and fl != fr:
            tags.add("subtype")
    return tags


def main():
    a0 = {r["scenario_id"]: r for r in map(
        json.loads, open(ROOT / "rebuttal_runs/A0/support_selection.jsonl"))}
    e1 = {r["scenario_id"]: r for r in map(
        json.loads, open(ROOT / "rebuttal_runs/E1/selection.jsonl"))}
    a0_test = load_test("rebuttal_runs/A0/test_metrics/*.jsonl", "AutoIR")
    e1_test = load_test("rebuttal_runs/E1/test_metrics/rule_gpu*.jsonl", "rule")
    # 7 nondeterministic scenarios: use the materialized fixed-LQ paired metrics
    fixed = json.load(open(ROOT / "rebuttal_runs/E1E2_fixed_lq_summary.json"))
    fixed_sids = set(fixed["fixed_scenarios"])
    for sid in fixed_sids:
        for ctrl, table in (("AutoIR", a0_test), ("rule", e1_test)):
            for ds in ("DIV2K", "LSDIR"):
                v = fixed.get("e1", {}).get("full", {}).get(ctrl, {}).get(ds, {}).get("per_scenario", {}).get(sid)
                if v is not None and sid in table and ds in table[sid]:
                    table[sid][ds]["psnr"] = float(v)

    rows = []
    for sid in sorted(a0):
        gt = [(x["function"], int(x["severity"])) for x in
              json.load(open(ROOT / f"exp41/degradation_gt/{sid}_params.json"))["pipeline"]]
        llm = pipe(a0[sid]["exp_meta"])
        rule = pipe(e1[sid]["exp_meta"])
        # rule initial R0: R0_fixed preferred
        r0p = ROOT / "rebuttal_runs/E1_fix/params" / f"{sid}_R0.json"
        if not r0p.exists():
            r0p = ROOT / "rebuttal_runs/E1/params" / f"{sid}_R0.json"
        rule_r0 = load_params(r0p) if r0p.exists() else []
        llm_r0 = [(x[0], int(x[1])) for x in
                  a0[sid]["per_round"]["R0"]["selected"]["exp_meta"]["degradation_pipeline"]]
        div = a0_test[sid]["DIV2K"]["psnr"]
        rule_div = e1_test[sid]["DIV2K"]["psnr"]
        ls = a0_test[sid]["LSDIR"]["psnr"]
        rule_ls = e1_test[sid]["LSDIR"]["psnr"]
        rec = {
            "scenario_id": sid, "gt": gt,
            "llm_round": a0[sid]["selected_round"], "llm": llm,
            "rule_round": e1[sid]["selected_round"], "rule": rule,
            "llm_arch": arch(a0[sid]["exp_meta"]), "rule_arch": arch(e1[sid]["exp_meta"]),
            "llm_support": a0[sid]["support_psnr_mean"],
            "rule_support": e1[sid]["support_psnr_mean"],
            "llm_div2k": div, "rule_div2k": rule_div,
            "llm_lsdir": ls, "rule_lsdir": rule_ls,
            "div2k_delta": div - rule_div, "lsdir_delta": ls - rule_ls,
            "support_delta": a0[sid]["support_psnr_mean"] - e1[sid]["support_psnr_mean"],
            "gt_match_llm": match(llm, gt), "gt_match_rule": match(rule, gt),
            "diff": diff_desc(llm, rule),
            "llm_r0": llm_r0, "rule_r0": rule_r0,
            "r0_diff": diff_desc(llm_r0, rule_r0),
            "tags": sorted(tags_of(llm, rule)),
            "r0_tags": sorted(tags_of(llm_r0, rule_r0)),
            "gt_match_llm_r0": match(llm_r0, gt), "gt_match_rule_r0": match(rule_r0, gt),
        }
        rows.append(rec)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    # summary counts
    diff_counts = Counter()
    for r in rows:
        if r["llm"] == r["rule"]:
            diff_counts["same"] += 1
        else:
            diff_counts["different"] += 1
    match_llm = Counter(r["gt_match_llm"] for r in rows)
    match_rule = Counter(r["gt_match_rule"] for r in rows)
    wins_div = sum(1 for r in rows if r["div2k_delta"] > 0.05)
    losses_div = sum(1 for r in rows if r["div2k_delta"] < -0.05)
    ties_div = len(rows) - wins_div - losses_div
    wins_ls = sum(1 for r in rows if r["lsdir_delta"] > 0.05)
    losses_ls = sum(1 for r in rows if r["lsdir_delta"] < -0.05)
    ties_ls = len(rows) - wins_ls - losses_ls
    avg_div = sum(r["div2k_delta"] for r in rows) / len(rows)
    avg_ls = sum(r["lsdir_delta"] for r in rows) / len(rows)
    tag_counts = Counter()
    for r in rows:
        for tag in r["tags"]:
            tag_counts[tag] += 1
    llm_more = sum(1 for r in rows if len(r["llm"]) > len(r["rule"]))
    rule_more = sum(1 for r in rows if len(r["llm"]) < len(r["rule"]))
    summary = {
        "n": len(rows), "same_pipeline": diff_counts["same"],
        "fixed_lq_scenarios": sorted(fixed_sids),
        "different_pipeline": diff_counts["different"],
        "tag_counts": dict(sorted(tag_counts.items())),
        "llm_more_steps": llm_more, "rule_more_steps": rule_more,
        "gt_exact_llm": match_llm["exact"], "gt_exact_rule": match_rule["exact"],
        "gt_type_llm": match_llm["type"], "gt_type_rule": match_rule["type"],
        "gt_order_llm": match_llm["order"], "gt_order_rule": match_rule["order"],
        "div2k_llm_wins": wins_div, "div2k_rule_wins": losses_div, "div2k_ties": ties_div,
        "div2k_avg_delta": avg_div,
        "lsdir_llm_wins": wins_ls, "lsdir_rule_wins": losses_ls, "lsdir_ties": ties_ls,
        "lsdir_avg_delta": avg_ls,
    }
    (OUT.parent / "choice_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# LLM/AutoIR vs rule controller：逐场景退化选择对比", "",
          "> GT 为事后标注；LLM 为 AutoIR 选中轮；rule 为 E1 选中轮。"
          "DIV2K/LSDIR 为固定 LQ test（同 A0/E1 指标）。"
          "`exact` = 函数+严重度+顺序全匹配；`type` = 函数+顺序匹配但 severity 不同；"
          "`category` = 类别顺序匹配；`order` = 类别相同但顺序不同；`same_len` = 步数相同但类别不同；`no` = 其他。", "",
          f"- 24 场景中，LLM 与 rule 选中管线相同 **{summary['same_pipeline']}** 个，不同 **{summary['different_pipeline']}** 个。",
          "- LQ 配对：17/24 场景 A0/E1 的 LQ hash 完全一致；7/24 非确定性场景（0003/0005/0017/0018/0020/0022/0023）使用 `E1E2_fixed_lq` 的同一份固定 LQ 重新评估。下表 test 数字已对 7 个场景使用 fixed-LQ 值。",
          f"- GT exact：LLM **{summary['gt_exact_llm']}**，rule **{summary['gt_exact_rule']}**；GT type-only（函数+顺序匹配、severity 不同）：LLM **{summary['gt_type_llm']}**，rule **{summary['gt_type_rule']}**；GT order-only（类别相同、顺序不同）：LLM **{summary['gt_order_llm']}**，rule **{summary['gt_order_rule']}**。",
          f"- 差异标签计数：severity **{tag_counts.get('severity',0)}**、subtype **{tag_counts.get('subtype',0)}**、step_count **{tag_counts.get('step_count',0)}**、order **{tag_counts.get('order',0)}**；LLM 步数更多 **{llm_more}** 个场景，rule 更多 **{rule_more}** 个。",
          f"- DIV2K test：LLM 胜 **{wins_div}**、rule 胜 **{losses_div}**、平 **{ties_div}**；平均 Δ(LLM−rule) **{avg_div:+.3f} dB**。",
          f"- LSDIR test：LLM 胜 **{wins_ls}**、rule 胜 **{losses_ls}**、平 **{ties_ls}**；平均 Δ **{avg_ls:+.3f} dB**。", "",
          "## 表 1：选择差异（最终选中轮）", "",
          "| 场景 | GT | LLM 选中 | rule 选中 | 差异类型/关键差异 | GT match (LLM / rule) |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['scenario_id']} | {fmt(r['gt'])} | **{r['llm_round']}**: {fmt(r['llm'])} | "
                  f"**{r['rule_round']}**: {fmt(r['rule'])} | {r['diff']} | "
                  f"{r['gt_match_llm']} / {r['gt_match_rule']} |")
    md += ["", "## 表 2：选择与性能（最终选中轮）", "",
           "| 场景 | LLM 轮/架构 | rule 轮/架构 | support Δ | DIV2K LLM | DIV2K rule | Δ | 胜者 | LSDIR LLM | LSDIR rule | Δ | 胜者 |",
           "|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|"]
    for r in rows:
        wd = "LLM" if r["div2k_delta"] > 0.05 else ("rule" if r["div2k_delta"] < -0.05 else "tie")
        wl = "LLM" if r["lsdir_delta"] > 0.05 else ("rule" if r["lsdir_delta"] < -0.05 else "tie")
        md.append(f"| {r['scenario_id']} | {r['llm_round']}/{r['llm_arch']} | {r['rule_round']}/{r['rule_arch']} | "
                  f"{r['support_delta']:+.2f} | {r['llm_div2k']:.2f} | {r['rule_div2k']:.2f} | "
                  f"{r['div2k_delta']:+.2f} | {wd} | {r['llm_lsdir']:.2f} | {r['rule_lsdir']:.2f} | "
                  f"{r['lsdir_delta']:+.2f} | {wl} |")
    md += ["", "## 表 3：初始 R0 选择（无反思/初始预测）", "",
           "| 场景 | GT | LLM R0 | rule R0（R0_fixed 优先） | 差异 | GT match (LLM / rule) |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['scenario_id']} | {fmt(r['gt'])} | {fmt(r['llm_r0'])} | {fmt(r['rule_r0'])} | "
                  f"{r['r0_diff']} | {r['gt_match_llm_r0']} / {r['gt_match_rule_r0']} |")
    md += ["", "## 模式总结", "",
           "- **LLM 常见的额外步骤**：brightness、blur、saturation、noise；例如 0001 多 blur、0009 多 brightness、0017 多 impulse、0021 多 motion/impulse。",
           "- **rule 常见的简单管线**：在多数场景只保留 1-2 步（如 0001 poisson:5、0005 impulse:5、0008 saturate:5），severity 往往取 5。",
           "- **子类型分歧**：0003 blur gaussian vs glass；0010 brightness gamma/HSV vs shift/RGB；0018 contrast stretch vs scale；0019 blur gaussian vs lens（两者都错，但 LLM 保留 JPEG/sat）。",
           "- **顺序分歧**：0010 LLM gamma→blur，GT 是 blur→gamma；0019 rule sat→JPEG，GT/LLM 是 JPEG→sat。",
           "- **LLM 大胜场景**：0008（severity 4 vs 5）、0014（LLM 精确 GT）、0005（rule 修正 impulse:5）、0001（rule poisson:5）等——注意这些并不全是“LLM 反思”单独贡献，有的是一开始 R0 就选对。",
           "- **rule 大胜场景**：0005（impulse:5 vs brightness+impulse:3）、0003（glass vs gaussian）、0009（rule 17.8 vs LLM 11.27）。",
           "- **两边都失败**：0009/0011/0017/0024 等仍有漏检/子类型错误；exact GT 不是性能的充分条件，性能也不是 GT 正确的充分条件。", "",
           "## 关键 caveat", "",
           "- 选中轮不同（LLM 有 5 R0/8 R1/11 R2；rule 有 5 R0/1 R0_fixed/18 R1/0 R2），因此表 1/2 的差异同时包含 controller 和 reflection 的差异；初始 R0 差异见表 3。",
           "- test 指标：17 个确定性场景 A0/E1 的 LQ hash 一致；7 个非确定性场景使用 E1E2_fixed_lq 的同一份固定 LQ。checkpoint 选择规则仍不同；不要把 Δ 全部归因于管线。",
           "- GT 是单场景单次的隐藏参数；exact/type 是描述性匹配，不是因果结论。",
           "- blind_0020 的 rule R0 训练 NaN；表 3 的 rule R0 管线仍可读，但其模型不可用。",
           "- 本表只覆盖 24 个 exp41 场景，不外推到新场景。"]
    MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {MD} and {OUT}")


if __name__ == "__main__":
    main()
