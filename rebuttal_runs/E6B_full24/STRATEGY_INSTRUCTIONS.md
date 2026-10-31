# E6-B full24 训练策略子 Agent 指令

你的挑战目录在 prompt 中给出。

## 强制流程

1. 完整 Read `/data/zyli/projects/autoresearch/finetune_strategy.md`（不可凭记忆）。
2. Read 本挑战目录的 `predicted_params.json`、`full_analysis.json`、`reflection.json`（若存在）。
3. 基于 per-step verdict 与原始信号，按 `finetune_strategy.md` §七 的信号门控规则，逐挑战分析并选择训练策略/架构。
   - 默认纯 Swin + SimpleGate + Direct。
   - 只有 Tier S/A 信号明确时才启用组件：OCAB 需 `anisotropy > 1.7` + motion；FiLM-GCM 需 contrast + 结构化（motion/jpeg）的 Tier A `variance_ratio`；ColorPre 仅 contrast Tier A。
   - DualBranch / CSN / SCA / FPro 不使用。不要套模板，写明本挑战的信号数值依据。
4. 只写本挑战目录的 `training_strategy.json`：
```json
{"strategy":"direct","architecture":{"attention_type":"swin","window_size":8,"use_simple_gate":true,"use_gcm":false,"use_color_pre":false,"use_color_mlp":false,"use_dual_branch":false,"use_csn":false,"use_channel_curve":false,"use_pcp":false,"use_sca":false,"use_gdfn":false,"use_fpro":false,"use_freqmod":false,"embed_dim":64},"lr":0.0005,"epoch_budget":2,"batch_size":16,"signal_basis":["具体信号与数值来源"],"rationale":"..."}
```
5. 禁止读取：`exp41/`、`.gt_mappings/`、`.ground_truth/`、任何 `degradation_gt`；`E6B_full24/` 下除本挑战目录与本指令文件外的任何文件（`support_plan_gt.json`、其他 `challenges/`、`logs/`、`experiments/`）；`E6A_full24/`、`E6_LLM/`、`E6/`、`E6_A/`、`E6A_repeat/` 等；任何 `test_metrics`/`results`/`logs`/既有预测。只允许读本挑战目录、`finetune_strategy.md`、`CLAUDE.md §Phase 5`。
6. 最终回复 ≤100 字：architecture 关键开关 + 一句依据。
