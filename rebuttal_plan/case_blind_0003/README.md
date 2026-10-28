# blind_0003 案例档案

这个文件夹是为审稿人问题单独建立的 blind_0003 复盘档案：复制原始预测、反思、信号、训练日志、评估日志、规则控制器产物和 support 图像，并用 `report.md` 逐轮拆解“每一步决策是规则还是 LLM”。

## 先读什么

1. **`report.md`** — 主报告，含一页结论、逐轮步骤、规则/LLM 归因矩阵、checkpoint 审计、对审稿人的建议回答。
2. **`metrics/metrics_provenance.json`** — 所有 PSNR/SSIM/LPIPS 数字的单一索引，标明来源文件。
3. **`data/challenge/reflection.json`** — 35 条持久化 reflection iteration 的原始记录。
4. **`data/rule_reference/E1_decision_blind_0003.json`** — 规则控制器在同一场景的候选打分和 R1 选择。

## 一句话结论

GT 是单步 `blur_glass:2`。LLM 在 R1 主动补测了 R0 漏掉的 `blur_glass:2`（36.65 dB / CI 10/10），这是真实的候选空间扩展；但 blur 反思规则（PSNR 改善 ≠ 子类型正确，2% 命中率）不允许因 PSNR 切换，因此 R1/R2 仍保留 `gaussian:2`，直到 R3 用 GT 训练 PSNR=19.33 触发修正才切到 glass。规则控制器 E1 在同一场景 R1 直接选 glass，固定 LQ 27.94/24.19，优于 AutoIR 的 25.53/22.33。因此本案例应作为“LLM 候选发现 + 规则过度保守 + GT 反馈修复”的局限性案例，而不是 LLM 独立选择或优于规则的证据。

## 目录

```
case_blind_0003/
├── report.md
├── README.md
├── MANIFEST.md              ← 文件清单 + sha256 + 来源路径
├── data/
│   ├── challenge/           ← predicted_params R0/R1/R3/R4、reflection、full_analysis、training_strategy
│   ├── qwen/                ← Qwen gate + Qwen pipeline
│   ├── rule_reference/      ← E1 规则控制器决策/params/selection/support/test metrics
│   └── gt_posthoc/          ← GT params（仅事后）
├── images/                  ← clean/degraded 5 对 + 单图（本地；.png 被 .gitignore 忽略）
├── logs/                    ← exp41 + E1 训练日志、评估日志、关键摘录
├── metrics/                 ← 最终指标、737 门控、E1 vs AutoIR、训练摘要
└── tools/
    ├── collect_extracts.py  ← 生成 metrics/ 下的抽取物
    └── make_manifest.py     ← 生成 MANIFEST.md
```

## 重要说明

- **checkpoint 没有复制**（`.pt` 大且不是决策证据）。训练日志里保留了 checkpoint 路径/step。
- **图像复制到了 `images/`，但 `.gitignore` 全局忽略 `*.png` 和 `*.log`**；因此这些文件在本地可查，但默认不会推到远程。如果需要上传，使用 `git add -f` 或调整 `.gitignore`。
- **时间戳审计：** R0/R1 训练在 2026-06-30 / 07-01；challenge 目录里的 `predicted_params.json`、`full_analysis.json`、`qwen_noise_gate.json`、`predicted_params_R1.json` 是 2026-07-14 06:32 的版本，可能是后来重建的。不要把它们的 mtime 当作决策时刻的证据。详见 `report.md §2.2` 和 `§10`。
- **checkpoint 选择审计：** `reeval_all.py` 的 `pts[-1]` 是字典序最后（step7546），不是数值最后（step15092），也不是训练时 best support。`report.md §3` 同时给出两套数字；引用轮次级 dB 时请注明口径。
- **Qwen 模型名：** 实际记录是 `qwen3.7-plus-2026-05-26`，不是文档中的 Qwen-Max/Qwen-VL-Max；Qwen 输出没有被最终采用。

## 复现

```bash
# 重新生成 metrics/ 下的抽取物（不复制图像/日志）
python3 rebuttal_plan/case_blind_0003/tools/collect_extracts.py

# 查看 35 条 reflection iteration
python3 - <<'PY'
import json
d=json.load(open('rebuttal_plan/case_blind_0003/data/challenge/reflection.json'))
for it in d['iterations']:
    t=it.get('test_result',{})
    print(it['round'], it['hypothesis']['pipeline'], t.get('psnr_rgb'), t.get('ci_pass'), it['decision'])
PY
```
