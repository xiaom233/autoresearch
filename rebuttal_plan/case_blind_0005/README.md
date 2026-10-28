# blind_0005 案例档案

这个文件夹回答审稿人问题 2：**CLR 如何区分“错误的退化假设”与“不够强的 restorer”，以及为什么没有纠正 impulse-noise 案例？**

主报告：`report.md`。

## 一句话结论

GT 是 `noise_impulse:5`。AutoIR 预测成 `brightness_brighten_shfit_HSV:1 + noise_impulse:3`：brightness 是 impulse:5 造成 35% clipping 的补偿性假阳性，severity 3 是低估。R0–R2 三次训练都使用同一条错误 DC，R3/R4 只是分析、没有重新训练（symlink→R2）。CLR 把错误管线的 CI=10/10 当作“理论最优”，把 GT PSNR=21.84 归因于模型容量，却从未测试 `noise_impulse:5` 单独。  
同一套 0.4M Swin+SG、同一 2-epoch/1.5 GPU-h 预算下，规则控制器只改 DC 为 `noise_impulse:5`，固定 LQ 得到 **44.92/41.14 dB**；AutoIR 为 **21.92/22.17 dB**。因此失败在 DC 搜索和 CLR 判据，不在 impulse noise 或 restorer 容量。

## 目录

```
case_blind_0005/
├── report.md                 ← 详细分析（含审稿人问题 2 的中英文回答）
├── README.md
├── MANIFEST.md               ← 文件清单 + sha256 + 来源
├── data/
│   ├── challenge/            ← predicted_params R0/R1、reflection、full_analysis、training_strategy、Qwen
│   ├── rule_reference/       ← E1 规则控制器决策/params/selection/support metrics/rules_frozen
│   └── gt_posthoc/           ← GT params（仅事后）
├── logs/
│   ├── exp41_R0/R1/R2_blind_0005.log
│   ├── E1_rule_R0/R1_blind_0005.log
│   └── key_log_extract.txt
├── metrics/
│   ├── gt_reeval_R0..R4.json
│   ├── gate_737_R2_21.84.json
│   ├── A0_autoair_test_metrics_blind_0005.jsonl
│   ├── E1_rule_test_metrics_blind_0005.jsonl
│   ├── E1_rule_vs_AutoIR_blind_0005.json
│   ├── training_summary.json
│   └── metrics_provenance.json
└── tools/collect_extracts.py
```

## 关键数字

| 项目 | AutoIR | 规则控制器 |
|---|---:|---:|
| 盲选管线 | `brightness:1 + impulse:3` | `impulse:5` |
| R0 support PSNR | —（R0 用 Set14） | 35.13 |
| R1 support PSNR | —（R2 support 21.66） | 43.17 |
| 固定 LQ DIV2K | 21.9155 | **44.9185** |
| 固定 LQ LSDIR | 22.1700 | **41.1387** |
| 架构 / 预算 | Swin+SG, 0.4M, 2 epoch, 1.5 GPU-h | 同左 |

## 重要限制

- **R3/R4 不是新训练**：`exp41_R3_blind_0005 -> exp41_R2_blind_0005`，`exp41_R4_blind_0005 -> exp41_R3_blind_0005`；只有 R0/R1/R2 三次训练，且 DC 相同。
- **R0/R1 的 Set14 41 dB 是预测退化验证**，不能证明 DC 正确；R2 support 21.66 才暴露问题。
- **checkpoint 口径不同**：`gt_reeval_R*` 用 step7546；A0 AutoIR 用 R1 step15092；规则控制器用 R1 step7546。不要混用。
- **规则控制器是评估侧 proxy**：它的固定候选表里有 `impulse:5`，不是盲 CLR；E5c 的 4 格受控实验尚未执行。
- **原始图像未复制**（约 50MB）；路径为 `exp41/challenges/phase4/blind_0005/{clean,degraded}_*.png`。checkpoint 未复制。
