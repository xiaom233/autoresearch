# E6-B（LLM）：agent-based support-set variation

> 目的：回答 reviewer 的 Support variation 质疑。E6-B 规则控制器版本只能说明**规则**对 support 采样敏感，
> 不能说明 AutoIR/LLM 的 support 稳定性。本实验用**完整 LLM 推理流程**重跑同一组 6 场景 × 3 组 support：
> Skill 盲识别 → 策略子 Agent → Direct R0 → Skill R1/R2 反思 → support PSNR 选择 → 固定 GT-LQ test。
>
> 日期：2026-10-28。产物：`rebuttal_runs/E6_LLM/`。

## 1. 设计

| 项 | 设置 |
|---|---|
| 场景 | `blind_0006/0008/0010/0013/0017/0024`（2 单/2 双/2 三） |
| support | 与 E6-B 规则版完全相同的 3 组各 5 对（`E6/support/set1-3`，来自 DIV2K_train，与 test 不重叠） |
| set1 盲识别 | 复用 E6-A rep1 的 Skill agent 输出（同一 5 对 support；逐文件 hash 相同） |
| set2/set3 盲识别 | 新跑 12 个 Skill 子 Agent（worktree 隔离，support-only） |
| 策略选择 | 18 个独立策略子 Agent，读 `finetune_strategy.md` Section 七，逐 challenge 输出 `training_strategy.json` |
| R0 | Direct，2 epoch / 15,094 steps，batch16，LR5e-4，seed42，`AR_CHALLENGE_DIR` = support |
| 反思触发 | **support-only**：verdict UNCERTAIN/POOR 或 support PSNR<35 → R1；LIKELY+support PSNR≥35 → 跳过；R1 提升 >2 dB → R2 |
| 选择 | R0-R2 中按 support PSNR 选 best checkpoint |
| test | 固定 GT-LQ DIV2K 100 + LSDIR 250；`eval_checkpoint.py test --seed-salt v1`，与 E6-B 完全相同（`lq_sha256` 已验证一致） |

## 2. 结果（DIV2K，fixed-LQ）

| 场景 | LLM set1 | set2 | set3 | mean | SD | range | rule set1 | rule set2 | rule set3 | LLM−rule mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blind_0006 | 31.590 | 31.620 | 31.605 | 31.605 | 0.015 | 0.030 | 31.705 | 31.667 | 31.508 | −0.022 |
| blind_0008 | 39.986 | 42.942 | 42.096 | 41.675 | 1.522 | 2.956 | 42.268 | 41.356 | 43.576 | −0.725 |
| blind_0010 | 31.593 | 31.625 | 31.151 | 31.456 | 0.265 | 0.474 | 15.472 | 15.820 | 18.010 | **+15.022** |
| blind_0013 | 29.206 | 29.197 | 29.086 | 29.163 | 0.067 | 0.120 | 26.029 | 26.221 | 26.005 | **+3.078** |
| blind_0017 | 23.496 | 24.194 | 23.500 | 23.730 | 0.402 | 0.699 | 23.244 | 23.217 | 23.346 | +0.461 |
| blind_0024 | 24.373 | 21.443 | 24.780 | 23.532 | 1.820 | 3.336 | 11.283 | 13.545 | 13.940 | **+10.609** |
| **macro** | | | | **30.194** | 0.682* | 1.269* | | | | **+4.737** |

\* `mean within-scenario SD` = 0.682；`mean within-scenario range` = 1.269。

**vs rule E6-B（primary：6 场景配对 bootstrap）**

| 数据集 | ΔPSNR (LLM−rule) | 95% CI | 场景胜 |
|---|---:|---:|---:|
| DIV2K | **+4.737** | [+0.341, +9.789] | **4/6** |
| LSDIR | **+4.026** | [+0.183, +8.725] | **4/6** |

- cell-level（18 个 scenario/set cell，**非独立**）：DIV2K +4.737 [+2.062,+7.705]，14/18；LSDIR +4.026 [+1.633,+6.719]，14/18。
- 主要增益来自 `blind_0010`（+15.02）和 `blind_0024`（+10.61）：规则在这两个场景严重错 DC；LLM 的盲识别/反思给出正确或接近正确的管线。
- `blind_0008` 规则略好（−0.73）；`blind_0006` 基本持平（−0.02）；`blind_0017` +0.46。
- **support 稳定性不是一致更好**：LLM 的 mean range 1.269 vs rule 1.326，但 `0008`（2.956 vs 2.221）、`0017`（0.699 vs 0.128）、`0024`（3.336 vs 2.658）的 range 反而更大；`0006/0010/0013` 更小。因此只能说“均值更好、稳定性场景相关”，不能声称 LLM 普遍更稳定。

## 3. 完整 LLM 流程实际做了什么

| 轮次 | 数量 | 说明 |
|---|---:|---|
| R0 | 18 | 全部完成，无 NaN |
| R1 反思 | 18 | 11 个 NO_CHANGE，4 个 CHANGE，3 个跳过（0008：LIKELY+support≥35） |
| R1 训练 | 4 | 仅 CHANGE 的链 |
| R2 反思 | 2 | `0013_set1`（NO_CHANGE）、`0017_set2`（CHANGE） |
| R2 训练 | 1 | `0017_set2` |
| **总训练** | **23** | **32.82 GPU-h**（rule E6-B：33 次 / 64.93 GPU-h） |

R1/R2 的具体变化：

| 链 | R0 → R1/R2 | support PSNR |
|---|---|---:|
| `0013_set1` | `jpeg:4 + speckle:1` → `speckle:3 + jpeg:2`（顺序+severity 修正） | 27.38 → **30.12** (+2.74)；R2 NO_CHANGE |
| `0017_set2` | 去掉假 `contrast_weaken_scale:1`；R2 再把 noise `YCrCb:3` → `RGB:4` | 20.56 → **23.63** (+3.07) → **23.72** (+0.09)；R2 选中 |
| `0017_set3` | `gaussian:4 + jpeg:4` → `gaussian:3 + jpeg:4` | 23.13 → **23.53** (+0.40) |
| `0024_set3` | noise `RGB:3` → `RGB:4` | 22.91 → **23.50** (+0.59) |

- 11 个 NO_CHANGE 链：R0 管线在 support 上像素级精确或已是最优；support PSNR<35 来自训练容量/不可逆信息损失，不是盲识别错误。
- 所有反思 Agent 只读 5 对 support + R0 输出；没有读 test/GT；R2 仅在 R1 support 提升 >2 dB 时触发。
- `0008` 三组按协议跳过反思（LIKELY + support PSNR 38.8–44.4 ≥35）。

## 4. 解释与限制

- **LLM 流程的收益主要来自 DC**：0010/0024 的 +15.0/+10.6 dB 是规则错 DC、LLM 选对（或接近）管线；不是架构差异。
- **不是普遍优势**：0008 规则更好，0006 持平；6 场景 CI 宽（[+0.34,+9.79]），且增益由 2 个场景主导。
- **support 稳定性**：LLM 的均值更好，但 range 在 3/6 场景更大；不能声称“LLM 对 support 更稳定”。
- **公平性**：两版使用同一 3 组 support、同一 test、同一 LQ（已验证 hash）；rule 版为规则控制器，LLM 版包含策略/反思，因此差异是“controller + reasoning”，不是单一变量。
- **set1 盲识别复用 E6-A rep1**；set2/set3 是新的 Skill agent run。每个 support 只有 1 次 agent run，不能分离 agent run-to-run 方差；E6-A 的 R0-only run-to-run 结果另行报告。
- **预算**：LLM 23 次训练/32.82 GPU-h 少于 rule 33 次/64.93 GPU-h，因为 11 个链反思 NO_CHANGE、11 个跳过 R1 训练；这是 controller 的早期停止优势。不能把“更少训练”当作 architecture 贡献。
- 6 场景、单 test LQ、support PSNR 选 checkpoint；CI 按 scenario 配对（6 个），不要把 18 个 cell 当独立样本。

## 5. 产物

```text
rebuttal_runs/E6_LLM/challenges/{sid}_set{k}/          # support + 盲识别/反思/策略输出
rebuttal_runs/E6_LLM/runs/{sid}_set{k}/R{0,1,2}/       # checkpoints
rebuttal_runs/E6_LLM/logs/                             # 训练/评测日志
rebuttal_runs/E6_LLM/selection.jsonl                   # R0/R1/R2 选择 + support PSNR
rebuttal_runs/E6_LLM/test_metrics_gpu*.jsonl           # 6300 行逐图 test
rebuttal_runs/E6_LLM/e6llm_summary.json / .md          # 聚合（含 scenario-level bootstrap）
rebuttal_runs/scripts/e6llm_collect_r0.py
rebuttal_runs/scripts/e6llm_reflection_plan.py
rebuttal_runs/scripts/e6llm_select.py
rebuttal_runs/scripts/run_e6llm_test.sh
rebuttal_runs/scripts/aggregate_e6llm.py
```
