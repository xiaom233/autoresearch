# E7：exp41 盲任务重复两次的随机性实验（优先）

> 目的：回答“系统的随机性有多大”。在同一 24 个 exp41 挑战、同一 5 对 support、同一 Skill/LLM 配置下，
> **独立跑两次盲识别**（Rep A / Rep B），测量 LLM/Skill 系统的 run-to-run 随机性；如果资源允许，
> 再用**固定训练配置**各训练 R0 并在固定 LQ test 上评估，测量端到端随机性。
>
> 日期：2026-10-29。状态：**已按用户要求停止所有实验**。Phase 1 曾启动 Rep A 前 8 个 agent，但只有部分中间文件（如 `blind_0008_repA/reflection.json`），**没有最终 `predicted_params.json`，不作为结果**。重启前先按 §9 的 DC/RS/CLR 计时规范执行；Phase 2（训练/评估）未执行。

## 1. 问题与预注册

- 问题：相同输入、相同 prompt、相同工具、相同 support 下，LLM 盲识别输出是否稳定？
- 预注册：
  1. 两次 run 的 challenge/support/GT 完全相同；GT 只用于最终评估，不进入盲识别。
  2. 两次 run 使用同一 Skill 版本、同一 prompt 模板、同一 agent 配置；不根据第一次结果修改第二次 prompt。
  3. 两次 run 都必须在 worktree 隔离中执行；禁止读取 exp41/degradation_gt、既有 predicted_params、test_metrics、其他 rep 的输出。
  4. 分析指标在跑之前固定（见 §4）；不因结果选择指标或删除场景。
  5. test 只在最终评估使用一次；不用 test 选择 rep/checkpoint/管线。

## 2. 设计

### Phase 1：两次独立盲识别（核心）

| 项 | 设置 |
|---|---|
| 场景 | exp41 `blind_0001–blind_0024`（24 个，8 单/8 双/8 三） |
| support | 每个场景的 5 对 support，来自 `exp41/challenges/phase4/blind_XXXX/`，两次 run 完全相同 |
| 重复 | Rep A、Rep B 各 24 次独立 Skill agent run（共 48 次） |
| 隔离 | 每个 challenge 复制到 `rebuttal_runs/E7/challenges/{sid}_repA/` 与 `_repB/`，只含 support 图像；无既有预测/GT |
| agent | 每 challenge 一个独立 Skill 子 Agent；worktree 隔离；同一 prompt 模板；同一工具集；同一模型/温度（配置记录，若 UNKNOWN 则标注） |
| 输出 | `predicted_params.json`、`reflection.json`、`full_analysis.json`（每 rep 每场景） |
| 禁止 | exp41/degradation_gt、E6_A/E6_LLM 既有预测、test_metrics、其他 challenge/rep、训练日志中的 GT |
| 记录 | 每次 run 的开始/结束时间（文件时间戳/agent 记录），用于 API 墙钟统计 |

### Phase 2：固定训练配置的端到端随机性（条件执行）

| 项 | 设置 |
|---|---|
| 触发 | Phase 1 完成且两次输出存在可测差异；Phase 2 不根据 test 结果选择 |
| 训练 | 每个 rep 的每个场景训练 1 次 R0：Direct，Swin+SimpleGate，EPOCH2/15,094 steps，batch16，LR5e-4，seed42，`AR_PARAMS_PATH` = 该 rep 的预测管线，`AR_CHALLENGE_DIR` = 5 对 support |
| 选择 | 固定训练配置；按 support PSNR 选 best checkpoint；不做 R1/R2（本轮只测随机性，不测反思收益） |
| test | 固定 GT-LQ DIV2K 100 + LSDIR 250；两次 rep 使用同一 LQ（`lq_sha256` 校验） |
| 对照 | 同一 rep 的管线重复训练一次（可选训练 seed 控制）用于分离训练 jitter；若做，单独报告 |
| 成本 | 48 次训练 × ~1.42 GPU-h ≈ 68 GPU-h；8 卡约 8–9 h 墙钟；评估另 ~10 min |

> Phase 2 是“盲识别 + 固定训练”的端到端随机性；不包含策略/架构选择、R1/R2 反思的方差。
> 若要测完整 AutoIR 全流程方差，需要另设实验；本计划先回答用户指定的“两次盲任务”。

## 3. 执行顺序

1. 冻结本计划与 prompt 模板；记录 Skill/模型版本。
2. 复制 24×2 个 challenge 目录（只含 support 图像）。
3. 启动 Rep A：24 个 Skill 子 Agent（按 CPU/并发分批，建议每批 8 个）。
4. Rep A 完成后，启动 Rep B：同 prompt、同配置、独立 agent；不读取 Rep A 输出。
5. Phase 1 冻结后分析；如需 Phase 2，再启动 48 个 R0 训练。
6. 生成 `E7/e7_summary.json/md`，更新 `02`、`05`、`STATUS`、`README`，推送远程。

## 4. 指标（预注册）

### Phase 1（盲识别）

- **整链 exact**：A/B 的函数+severity+顺序完全一致的比例（24 场景）。
- **整链 type-only**：函数+顺序一致、severity 可不同的比例。
- **位置级 type / spec**：两 rep 在同一位置的函数 / 函数+severity 一致率（分母 48 GT 位置）。
- **步数一致率**：A/B 预测步数相同的比例。
- **严重度差**：同一位置同函数时 severity 的绝对差（mean/median/max）。
- **逐场景变化**：24 场景中任一步改变的数量；按单/双/三分层。
- **verdict/CI 变化**：LIKELY/UNCERTAIN/POOR 与 CI 的分布变化。
- **API 墙钟**：每 rep 每场景从 challenge 读取到输出完成的时间（文件时间戳；batch 并发导致的上界，需注明）。

### Phase 2（端到端）

- 固定 LQ DIV2K/LSDIR 每场景 rep A/B PSNR；**ΔPSNR = B − A** 的 mean、median、SD、range、|Δ|>1 dB 场景数。
- 与 Phase 1 的 DC 差异关联：管线完全相同的场景 vs 不同的场景，ΔPSNR 分布对比。
- 训练/checkpoint 抖动：同一 rep 重复训练（若做）的 support/test 差异，单独列出。

## 5. 预期与判读

- 如果 A/B 整链 exact 高（例如 ≥70%），说明盲识别系统对同输入较稳定；低（例如 <40%）说明 run-to-run 随机性大。
- 位置级 type/spec 与整链 exact 必须同时报告，避免用位置微平均掩盖整链错误。
- 端到端 ΔPSNR 的 SD 是“系统随机性”的直接度量；若 DC 完全相同但 ΔPSNR 仍大，说明训练/选点抖动主导，需报告为训练方差而非 LLM 方差。
- 不预设“LLM 随机性大/小”；按实际结果写。

## 6. 产物

```text
rebuttal_plan/12_exp41_repeat_randomness.md          # 本计划（先冻结）
rebuttal_runs/E7/challenges/{sid}_rep{A,B}/          # support + agent 输出
rebuttal_runs/E7/logs/                               # 训练/评估日志
rebuttal_runs/E7/runs/{sid}_rep{A,B}/R0/             # Phase 2 checkpoints
rebuttal_runs/E7/selection.jsonl
rebuttal_runs/E7/test_metrics_gpu*.jsonl
rebuttal_runs/E7/e7_summary.json / .md
rebuttal_runs/scripts/e7_*                           # 生成/聚合脚本
```

## 9. DC/RS/CLR 每轮计时（强制执行）

> 完整规范见 `13_end_to_end_cost_and_timing.md`；模板见 `rebuttal_runs/E7/timing_schema.json`，训练/评测 wrapper 见 `rebuttal_runs/scripts/timing_wrapper.sh`。

每个 `scenario × rep × round` 必须写 `timing.json`，至少包含：

| 阶段 | 必须字段 |
|---|---|
| **DC** | start/end UTC、wall_seconds、mode（blind/reflection）、agent、api_calls/tokens（可 UNKNOWN）、candidate_sims、cpu_seconds、frozen_pipeline、verdict、CI |
| **RS** | start/end、wall_seconds、strategy_agent_seconds、train_start/end、train_seconds、checkpoint_selection_seconds、gpu_hours |
| **CLR** | start/end、wall_seconds、api_calls/tokens、candidate_sims、cpu_seconds、decision（CHANGE/NO_CHANGE/SKIP） |
| **EVAL** | start/end、wall_seconds、datasets |
| **totals** | online_wall_seconds、gpu_hours、cpu_hours、api_calls、tokens、api_cost_usd；离线开发单列，不摊入 |

归属规则：R0 DC = 盲识别；R1/R2 DC = 上一轮反思冻结 `predicted_params_R{r}.json` 前的 DC 决策；CLR_r = 训练后反思分析；若同一反思产生下一轮 DC，在冻结时刻切分，无法切分则整体计入 CLR_r 并标 `DC_{r+1}=included_in_CLR_r`，不重复计时。没有记录就填 null/UNKNOWN，禁止回溯编造。

## 7. 成本与限制

- Phase 1：48 次 agent run；CPU/API 成本，模型 ID/版本/温度/费用若无法记录则标 UNKNOWN。
- Phase 2：约 68 GPU-h（8 卡约 8–9 h），不含 R1/R2。
- 限制：两次 run 不能覆盖所有 API 随机性来源；prompt/模型版本固定；support 只有 5 对；test 固定 LQ；单次实验不能外推到其他 LLM/工具版本。
- 本实验优先级高于 E6-A/E3-G/E2-RS/E5b；E6-A 已停止，不混用其 partial 结果。

## 8. 需要阅读的文档（执行前）

- `.claude/skills/image-degradation-simulator/SKILL.md`（盲识别协议）
- `CLAUDE.md` §子 Agent 盲识别协议 / §Agent 上下文隔离 / §Phase 4
- `program.md`（退化管线与同图盲识别约束）
- `BLIND_ISOLATION.md`（如适用）
- Phase 2 启动前：`finetune_strategy.md` §六/§七（固定 Swin+SG 的依据）与 `CLAUDE.md` §Phase 5
