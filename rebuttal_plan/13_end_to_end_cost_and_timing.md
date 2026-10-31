# 端到端成本与 DC/RS/CLR 每轮计时规范

> 目的：回答 Weakness 4：
> “Table 1 的效率只描述最终 0.42M restorer，排除了 LLM 调用、退化候选评估、重复 CLR 训练和离线框架开发。
> 而且 1.5 GPU-h/run 与最多 3 轮反思下‘每场景 5 GPU-h’的口径不清。完整端到端成本不清楚。”
>
> 本文做两件事：
> 1. **拆清口径**：5 GPU-h 只指什么、哪些成本没算、哪些能测/不能测。
> 2. **定义记录规范**：以后每个场景/每轮必须记录 **DC / RS / CLR** 的分段时间，以及 API、候选评估、训练、评测、离线开发成本。

## 0. 结论先行

- **5 GPU-h/场景不是端到端成本**。exp41 的实测 R0–R2 训练合计 **4.248 GPU-h/场景**（71 次训练 / 101.95 GPU-h / 24 场景）；“5”只是把 3×1.5h 的训练预算上界四舍五入后的说法。
- 5 GPU-h **不含**：LLM/Skill API 墙钟、DC/CLR 的候选模拟评估、R1/R2 反思调用、最终 test 评估、离线规则/框架开发、DFPIR 预训练。
- exp41 目前只能恢复部分成本：
  - 训练：精确（`training_seconds`）。
  - 盲识别 API：可用 exp40 challenge 文件时间戳近似 **26.7 min/任务**（批处理墙钟，不是纯 API 延迟）。
  - R1/R2 反思 API：**无法精确恢复**（reflection summary 只有日期，没有时刻/耗时）。
  - 候选评估：包含在 agent 墙钟里，但没有单独计时。
  - 离线开发：723+ 训练去重清单缺失（raw 日志 1,137 条），不能作为精确数字。
- 因此论文不能声称“5 GPU-h/场景 = 完整端到端成本”；应写成“**训练预算 4.25 GPU-h/场景（R0–R2），端到端 LLM/候选/评测/离线成本另列**”，并把 UNKNOWN 如实标出。
- 从下一个实验（E7）开始，按 §3 的 `timing.json` 强制记录 DC/RS/CLR 每轮时间；没有记录就标 UNKNOWN，不回溯编造。

## 1. 成本分解（必须分开报告）

### 1.1 在线、每场景成本

| 组成 | 含义 | exp41 现状 | 测量方式 |
|---|---|---|---|
| **DC**（degradation characterization） | 盲识别/反思中的退化决策：Skill/RFA、候选模拟、`test_candidate`/`verify_signals` | 盲识别代理 ~26.7 min/任务（批处理墙钟）；DC 候选评估未单独计时 | agent 开始/结束时间 + 候选脚本计时 |
| **RS**（restoration strategy） | 策略/架构选择 + 训练 + checkpoint/support 选择 | 训练 4.248 GPU-h/场景（R0–R2，71 次/24 场景）；策略选择时间未记录 | `EXP_META` 开始 + `training_seconds`；support eval 日志 birth/mtime |
| **CLR**（closed-loop reflection） | 训练后反思：信号重释、候选测试、DC 修正 | R1/R2 反思只有日期（2026-07-01/07-14/07-15），**无时长** | reflection agent 开始/结束 + 候选计时 |
| **EVAL** | 最终固定 LQ test（DIV2K 100 + LSDIR 250） | 逐模型 ~2–5 min GPU；未折入 5 GPU-h | 评测 wrapper start/end |
| **API** | LLM/Skill 调用次数、token、费用、模型版本 | **UNKNOWN**（没有 token/cost 日志） | API 响应日志/计费；无则 UNKNOWN |

### 1.2 离线、非每场景成本（必须单列）

| 组成 | 现状 | 是否可折算到每场景 |
|---|---|---|
| 规则/框架开发（exp9/10/12/37/38/39/40） | 723+ 训练声明；raw 日志 1,137 条（含重复/失败）；去重清单缺失 | **不能**按每场景摊薄后声称 5 GPU-h；应单列总成本或标 UNKNOWN |
| 校准（blur/JPEG/3840） | 128×5 + 1,320×2 + 3,840-case（manifest 缺失） | 单列 |
| DFPIR 预训练 | 项目 blinded DFPIR 的训练成本未计入 AutoIR 预算 | 单列；同预算比较只比较微调阶段 |
| 人工/提示词/规则维护 | UNKNOWN | 单列或标 UNKNOWN |

## 2. exp41 的可复算数字（当前能写进 rebuttal 的）

| 项 | 数值 | 来源/口径 |
|---|---:|---|
| R0–R2 真实训练次数 | 71 | `A0/exp41_selection_manifest.jsonl`；去重后 |
| 训练 GPU-h | **101.95** | Σ`training_seconds`；单卡每 run |
| 每场景训练 | **4.248 GPU-h** | 101.95 / 24；等价 R0+R1+R2 实际秒数 |
| 每次训练均值 | 1.436 h | 71 次；与 1.5h 预算一致 |
| 盲识别 API 代理 | **26.7 min/任务**（median 29.7，range 15.9–37.0） | exp40 challenge `degraded.png` birth → `predicted_params.json` birth；**批处理墙钟** |
| R0/R1/R2 support 评估 | 2.42 / 3.00 / 3.98 min 每任务 | `exp41/logs/reeval_R*_gpu*.log` birth→mtime / 每 GPU 任务数 |
| 最终 test 评估 | 未折入 5 GPU-h；E6-LLM 18 链 ×350 图约 10 min/8 卡 | 评测 wrapper/日志 |
| R1/R2 反思 API | **UNKNOWN** | reflection summary 只有日期 |
| 候选评估 CPU | **UNKNOWN**（包含在 agent 墙钟内） | 未单独计时 |
| API token/费用 | **UNKNOWN** | 无日志 |

**对 5 GPU-h 的准确表述**：
> “R0–R2 的训练预算是 3×1.5 GPU-h 的上界（实测 4.248 GPU-h/场景）；5 GPU-h 是四舍五入后的训练预算，不是包含 API、候选评估、CLR 反思、最终评测和离线开发的端到端成本。”

## 2.5 现有 log 覆盖审计（2026-10-29）

**结论：现有 exp41/E1/E2 日志不足以恢复完整 DC/RS/CLR 时间；只能恢复训练和部分评测。** 逐项：

| 阶段 | 现有记录 | 缺失 | 能否回填 |
|---|---|---|---|
| exp41 R0–R2 训练 | `EXP_META.timestamp`（开始）+ `training_seconds` | 无显式结束墙钟 | ✅ 精确（开始+秒数） |
| exp41 support eval | `exp41/logs/reeval_R*_gpu*.log` birth/mtime | 日志内无 start/end | ✅ 近似（birth→mtime / 每 GPU 任务数） |
| exp41 盲识别 DC | exp40 `challenges/.../degraded.png` birth → `predicted_params.json` birth | exp41 是复制时间，非原始；无 API 时长/token | ⚠️ 仅代理（批处理墙钟） |
| exp41 DC 候选评估 | 无单独计时 | `run_full_analysis`/`test_candidate` 次数/CPU 秒 | ❌ UNKNOWN |
| exp41 策略选择 RS | `training_strategy.json` 无 timestamp | 子 Agent 开始/结束 | ❌ UNKNOWN |
| exp41 R1/R2 反思 CLR | `reflection.json` 的 r1/r2 summary 只有 **日期** | 时刻、墙钟、候选次数、token | ❌ UNKNOWN |
| exp41 最终 test | `results/gt_reeval_*` mtime 是复制时间（2026-07-30） | start/end、GPU 数、逐数据集 | ⚠️ 仅 E6-LLM 有 wrapper 日志可参考 |
| E6-LLM 盲识别/反思 | challenge/agent 输出真实 mtime；训练精确；test wrapper 只有 DONE | 单次 agent start/end、token、候选计时 | ⚠️ 部分可测（墙钟），非纯 API |
| E1/E2/规则 | 训练 `training_seconds`；部分 eval log birth/mtime | 策略/反思/候选/评测 start/end | ⚠️ 部分 |
| 离线开发 | 723+ 去重清单缺失；raw 日志 1,137 条 | 精确 run 清单、校准 manifest | ❌ UNKNOWN |

**因此**：论文不能把“5 GPU-h/场景”写成完整端到端成本；必须写“训练预算 4.25 GPU-h/场景 + 在线 API/候选/评测/离线开发另列”，并把缺失项标 UNKNOWN。E7 起按 §3 强制记录；**不回溯编造**。

## 2.6 E6-A / E6-B 新实验的实测分段时间（2026-10-30）

E6-A full24 / E6-B full24 对每条链写入 `agent_timing*.json`（agent 墙钟、候选次数）、`timing*.json`（训练秒数）与评测 wrapper 日志，是第一批可引用的 DC/RS/CLR 实测值，可补上 exp41 的 UNKNOWN 缺口（exp41 本身仍是 UNKNOWN，不能用新实验回填）。

| 阶段 | 实测值 | 口径/来源 |
|---|---:|---|
| DC 盲识别（E6-B，48 run） | mean **19.4 min** / median 14.0 / range 2.0–100.9；mean 17.1 次 `test_candidate` | `E6B_full24/challenges/*/agent_timing.json`；墙钟含 Skill/API 与候选模拟 |
| DC 盲识别（E6-A，48 run） | mean **18.6 min** / median 16.3 / range 2.0–82.7；mean 17.0 次 | `E6A_full24/challenges/*/agent_timing.json` |
| RS 策略选择 | ~**0.3–1.5 min**/链 | 本会话策略子 Agent 实测（未单独持久化） |
| R0 训练（E6-B，40 run） | mean **1.477 GPU-h** / median 1.424 / max 1.965 | `timing.json` 的 `train_seconds` |
| R0 训练（E6-A，48 run） | mean **1.414 GPU-h**（合计 67.86 GPU-h） | 训练日志 `training_seconds` |
| support eval / checkpoint 选择（exp41） | R0 **2.42** / R1 **3.00** / R2 **3.98** min per task | `reeval_R*_gpu*.log` birth→mtime；E6-A/E6-B 为训练内验证，已含在 `train_seconds` |
| CLR 反思（E6-B，29 run） | mean **9.7 min** / median 5.4 / range 1.25–87；mean 8.6 次候选 | `agent_timing_R1.json`；含信号重释与候选模拟 |
| CLR 反思（E6-A，46 run） | mean **18.1 min** / median 15.2 / range 4.1–45.0；mean 10.6 次候选 | 同上 |
| R1 训练（仅 CHANGE） | exp41 mean **1.410 GPU-h**（23 run）；E6-A mean 1.412（9 run） | 训练日志 |
| R2 训练（仅 R1 +>2 dB） | exp41 mean **1.451 GPU-h**（24 run） | 训练日志 |
| 最终 test（E6-A） | 48 模型 ×350 图：8 卡 **23.3 min** = 3.11 GPU-h；**0.065 GPU-h/模型**；fixed-LQ 复评 14 模型 8 卡 5.5 min | `eval_final_gpu*.log` birth→mtime |
| API token / 费用 | **UNKNOWN** | 未记录；只报告 agent 墙钟 |

**分项合计（GPU 计算与 agent 墙钟分开）**：

| 实验 | GPU 计算/场景 | agent 墙钟/场景 | 说明 |
|---|---:|---:|---|
| exp41（24 场景） | 训练 4.248 + support eval 0.157 + test 0.065 ≈ **4.47 GPU-h** | DC ~26.7 min（exp40 代理）+ 策略 ~1 min + 反思实测代理 10–18 min/轮（最多 2 轮）≈ **0.6–1.1 h** | 串行端到端约 **5.0–5.6 h/场景**；71 次训练/101.95 GPU-h |
| E6-A（48 链） | R0 1.414 + R1 0.265（9/48 平均）+ test 0.065 ≈ **1.74 GPU-h/链**；每场景（2 链）≈ **3.5 GPU-h** | DC 18.6 min + 策略 ~1 min + CLR 18.1 min（46/48）≈ **0.63 h/链**；每场景 ≈ **1.25 h** | R1 反思 46/48 触发，9/48 CHANGE |
| E6-B（40/48 R0 完成） | R0 1.477 + R1 0.118（4/48 平均）+ test 0.065 ≈ **1.66 GPU-h/链**；每场景（2 链）≈ **3.3 GPU-h** | DC 19.4 min + 策略 ~0.7 min + CLR 9.7 min（29/48 触发）≈ **0.44 h/链**；每场景 ≈ **0.9 h** | 25/29 反思 NO_CHANGE，R1 训练显著少于 exp41 |

**结论口径**：exp41 的 4.248 GPU-h/场景是**三次训练（R0–R2）实测总和**（71 次，均值 1.44 h/次），不是“5 h 含一切”；加上 support eval 与最终 test 约 4.47 GPU-h，加上 DC/CLR agent 墙钟约 0.6–1.1 h，串行端到端约 5–5.6 h/场景。新实验（E6-A/E6-B）用逐阶段计时给出完整在线成本；API token/费用与离线框架开发（723+ 训练、3,840-case 校准、DFPIR 预训练）仍单列/UNKNOWN，不摊入每场景。

## 3. DC/RS/CLR 每轮计时规范（E7 起强制执行）

### 3.1 文件与字段

每个 `scenario × rep × round` 写一份 `timing.json`（绝对时间 UTC + 秒）：

```json
{
  "scenario_id": "blind_0001",
  "rep": "repA",
  "round": "R0",
  "stages": {
    "DC": {
      "start_utc": "2026-10-29T01:00:00Z",
      "end_utc": "2026-10-29T01:26:00Z",
      "wall_seconds": 1560,
      "mode": "blind",
      "agent": "skill",
      "api_calls": null,
      "tokens": null,
      "candidate_sims": null,
      "cpu_seconds": null,
      "frozen_pipeline": [["compression_jpeg", 1]],
      "verdict": "LIKELY",
      "ci": "10/10"
    },
    "RS": {
      "start_utc": "2026-10-29T01:26:00Z",
      "end_utc": "2026-10-29T02:52:00Z",
      "wall_seconds": 5160,
      "strategy_agent_seconds": null,
      "train_start_utc": null,
      "train_end_utc": null,
      "train_seconds": 5070,
      "checkpoint_selection_seconds": null,
      "gpu_hours": 1.408
    },
    "CLR": {
      "start_utc": null,
      "end_utc": null,
      "wall_seconds": 0,
      "api_calls": null,
      "tokens": null,
      "candidate_sims": null,
      "cpu_seconds": null,
      "decision": "SKIP"
    },
    "EVAL": {
      "start_utc": null,
      "end_utc": null,
      "wall_seconds": null,
      "datasets": ["DIV2K", "LSDIR"]
    }
  },
  "totals": {
    "online_wall_seconds": 6720,
    "gpu_hours": 1.408,
    "cpu_hours": null,
    "api_calls": null,
    "tokens": null,
    "api_cost_usd": null
  },
  "offline_development_included": false,
  "agent_model_version": "UNKNOWN",
  "notes": "template/example values; do not use as a measured result"
}
```

### 3.2 归属规则（避免重复计数）

- **DC_r**：第 r 轮的退化配置决策时间。R0 是盲识别 agent；R1/R2 是上一轮反思中**冻结 `predicted_params_R{r}.json` 之前**的 DC 决策部分。
- **RS_r**：第 r 轮策略选择 + 训练 + checkpoint/support 选择。GPU 训练用 `training_seconds`；策略子 Agent 时间单列。
- **CLR_r**：第 r 轮训练后的反思分析时间（信号重释、候选验证）。如果同一反思同时产生下一轮 DC，则在 `predicted_params_R{r+1}.json` 冻结时刻切分：冻结前计入 **DC_{r+1}**，冻结后候选/分析计入 **CLR_r**；若无法切分，整体计入 CLR_r，并把 DC_{r+1} 标为 `included_in_CLR_r`（值为 0），**不重复计时**。
- 跳过/NO_CHANGE 的轮次记录 `wall_seconds=0`、`decision=SKIP|NO_CHANGE`，不要用训练预算占位。
- 并行 agent 要同时记录 **累计 wall** 与 **关键路径 wall**；不能把并行 agent 的墙钟相加当作端到端延迟。
- 没有实际计时就填 `null`/UNKNOWN；**禁止回溯编造**。

### 3.3 必须记录的成本

1. LLM/Skill：模型 ID/版本、调用次数、token（输入/输出）、API 费用、agent 墙钟。
2. 候选评估：`run_full_analysis`/`test_candidate`/`verify_signals` 次数、CPU 秒、是否 GPU。
3. 训练：`EXP_META` 开始、`training_seconds`、checkpoint 选择/support eval 时间。
4. 评测：最终 test 的 start/end、GPU 数、每数据集耗时。
5. 离线开发：单独 ledger（训练次数、GPU-h、校准、规则维护），不摊入每场景。

## 4. 对 Weakness 4 的建议回复口径

- 明确承认：当前 Table 1 的 5 GPU-h **只是 R0–R2 训练预算**；实测 4.248 GPU-h/场景，不是端到端。
- 补一张**分项成本表**：训练（4.248 GPU-h/场景，71 次/101.95 GPU-h）、support eval（~9.4 min/场景）、盲识别 API（~26.7 min/任务代理，批处理）、最终 test（按实际评测 wrapper 记录）、R1/R2 反思 API（UNKNOWN，需补测）、候选评估（UNKNOWN，需补测）。
- 离线开发（723+、规则、校准、DFPIR 预训练）单列，不摊薄；723+ 目前缺 dedup manifest，需补或标 UNKNOWN。
- 不要写“5 GPU-h 覆盖端到端”；改写为“训练预算 4.25 GPU-h/场景 + 未计入的在线/离线成本”。
- 从 E7 起按 §3 记录每轮 DC/RS/CLR 时间；这是唯一能给出完整端到端成本的可靠方式。

## 5. exp41 的现有证据与缺口

- 可复算：`A0/exp41_selection_manifest.jsonl`、`exp41/logs/exp41_R{0,1,2}_blind_*.log`、`exp41/logs/reeval_R*_gpu*.log`、`exp40/challenges/phase4/blind_*` 文件时间。
- 缺口：reflection 时刻/耗时、候选评估单列时间、token/费用、723+ dedup、3840 manifest、prompt 版本。
- 这些缺口**不能**用“5 GPU-h”掩盖；rebuttal 应明确列为 UNKNOWN/未计入，并说明 E7 的 timing 规范将补齐。

## 6. 产物

```text
rebuttal_plan/13_end_to_end_cost_and_timing.md       # 本规范
rebuttal_runs/E7/timing_schema.json                  # 模板（不是结果）
rebuttal_runs/E7/timing.jsonl                        # 每轮记录（实验重启后）
rebuttal_runs/scripts/timing_wrapper.sh              # 训练/评测墙钟 wrapper
rebuttal_runs/scripts/e7_existing_timing_audit.py    # 现有 log 覆盖审计
rebuttal_runs/E7/existing_timing_coverage.json       # 现有 log 可恢复/缺失清单
rebuttal_runs/E7/logs/                               # 训练/评测日志
```
