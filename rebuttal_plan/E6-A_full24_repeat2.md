# E6-A（全 24 场景）：同一 support / 训练 seed 下的 Agent 重复运行

> 日期：2026-10-29。产物目录：`rebuttal_runs/E6A_full24/`（新目录，不覆盖 `E6_A` / `E6A_repeat`）。
> 本实验取代原 E6-A 的 6 场景 × 3 次设计，按用户要求改为 **exp41 全部 24 个场景 × 2 次独立 agent run**，
> 以获得对 1/2/3 步退化均有代表性（exp41 = 8 单步 / 8 双步 / 8 三步）的 run-to-run 方差。

## 1. 问题

相同 support（逐文件 sha256 相同）、相同训练配置与 seed 下，Agent 两次独立盲识别的输出是否稳定？
输出差异会导致多大的固定 LQ test PSNR 差异？

## 2. 设计（预注册）

| 项 | 设置 |
|---|---|
| 场景 | exp41 `blind_0001–blind_0024`（24 个） |
| 重复 | `rep1`、`rep2` 两次**独立** agent run；每个 challenge 一个独立子 Agent |
| support | 每场景 5 对，直接复制自 `exp41/challenges/phase4/blind_XXXX/{clean,degraded}_0..4.png`；两个 rep 与源文件 sha256 必须完全一致 |
| 隔离 | 子 Agent 使用 `isolation: "worktree"`；只允许读本 challenge 目录与 SKILL 脚本；禁止 exp41、GT、其他 challenge/rep、既有预测 |
| 盲识别 | Skill `image-degradation-simulator` 协议；R0-only（无训练后 R1/R2 反思） |
| 训练 | 固定 Direct，Swin+SimpleGate，embed 64 / ws 8，2 epoch / 15,094 steps，batch 16，LR 5e-4，seed 42；`AR_CHALLENGE_DIR` = 5 对 support |
| checkpoint | 两种：best（每 3,773 步用 support PSNR 选）与 last（固定 step 15092，用于分离训练抖动） |
| test | 固定 GT-LQ：DIV2K 100 + LSDIR 250；`eval_checkpoint.py test --seed-salt v1` |
| GT 用途 | 仅 test LQ 生成与事后匹配统计；不进入盲识别与训练 |

## 3. 预注册指标

### 3.1 盲识别一致性（rep1 vs rep2）

- 整链 exact：函数+严重度+顺序完全一致的比例（24 场景）。
- 整链 type-only：函数+顺序一致、严重度可不同的比例。
- 步数一致率。
- 位置级 type / spec：按两 rep 对齐位置统计（分母 48 个 rep1 GT 位置）。
- 同位置同函数时 severity 的绝对差（mean/median/max）。
- verdict / overall_verdict 分布变化。

### 3.2 端到端 test PSNR

- 每场景 rep1/rep2 的 DIV2K、LSDIR PSNR；within-scenario SD / range。
- macro：场景均值、场景间 SD、mean within-scenario SD、max range（分 best / last）。
- 分层：按 GT 步数 1/2/3 分别报告 within-scenario SD/range（代表性）。
- 关联：pipeline 完全相同的场景 vs 不同的场景，其 |ΔPSNR| 分布对比。
- `best` 混合 agent 方差 + 选点抖动；`last` 固定 step，隔离训练/选点抖动。

### 3.3 判读

- 不预设“LLM 方差大/小”；按实际结果报告。
- 不能外推到其他 LLM/API 版本；本实验 API model/温度等记为 UNKNOWN。
- E6-A 是 R0-only，不是完整 AutoIR R0–R2 流程方差；不能替代 E6-B 的 support-set 稳定性。

## 4. 失败处理（预注册）

- Agent 未产出完整 `predicted_params.json`/`reflection.json` → 该 run 无效，重跑一次；仍失败则标记缺失，不插补。
- 训练 NaN / 无有效 checkpoint → 按固定配置重训一次；仍失败则标记缺失，不插补。
- 重复 run 不因 test 结果挑选或删除场景。

## 4bis. 2026-10-29 扩展：R0 → R1 → R2（用户要求，无人值守）

> 原设计为 R0-only；用户要求自动推进 R1/R2。以下规则在 R1/R2 启动前写入，未依据任何 test 结果调整。

- **R1 触发（support-only，无 GT/test）**：R0 verdict ∈ {UNCERTAIN, POOR}，或 R0 best support PSNR < 35 dB，或训练 NaN → 运行 R1 Skill 反思子 Agent；`LIKELY + support PSNR ≥ 35` → 跳过 R1。
- **R1 反思**：worktree 隔离；只读本 challenge 的 5 对 support、R0 输出与 `r0_result.json`；Read SKILL.md；重跑 `run_full_analysis.py`；≤2 轮信号重释、≤5 个修正候选、≤15 次 `test_candidate.py`；噪声候选用 `verify_signals`，不用 PSNR 拒绝。输出 `predicted_params_R1.json`、`reflection.json`（追加 `reflection_R1`）、`r1_decision.json`、`agent_timing_R1.json`。
- **R1 训练**：仅当 R1 pipeline 与 R0 不同（CHANGE）才训练；配置与 R0 完全相同（Direct Swin+SG，2 epoch，seed 42），输出 `experiments/{name}_R1/`。
- **R2 触发**：R1 CHANGE 且 R1 support PSNR 比 R0 高 **>2.0 dB** → R2 反思；否则跳过。R2 规则同 R1（输出 `*_R2`），仅 pipeline 改变时训练。
- **最终选择**：在 R0/R1/R2 中按 best support PSNR 选一个 checkpoint（并列取较早轮次）；同一份固定 GT-LQ test 评估。
- **报告**：同时报告 R0-only 方差与 R0→R1→R2 全流程方差；分别列出 R1/R2 触发数、CHANGE 数、训练次数与 support PSNR 变化。
- **限制**：R1/R2 仍是同一 support、同一固定训练配置；不是超参/架构搜索。API model/温度等为 UNKNOWN。

## 5. 成本

- 盲识别：48 次 agent run（CPU，6 批 × 8 并行）。
- 训练：48 × ~1.42 GPU-h ≈ 68 GPU-h；8 卡约 8.5 h 墙钟。
- test：48 模型 × 350 张固定 LQ，8 卡并行。

## 6. 执行结果（2026-10-29）

### 6.1 执行审计

| 项 | 结果 |
|---|---|
| R0 盲识别 | 48/48 完成（24 场景 × rep1/rep2）；agent 墙钟合计 14.88 h（CPU）；`test_candidate` 合计 817 次 |
| 协议偏差 | `blind_0018_rep2` 使用 29 次 `test_candidate`（预算 25）；保留不删，作为偏差记录 |
| R0 训练 | 48/48，无 NaN；**68.32 GPU-h**（每次 ~5120 s） |
| R1 触发 | 46/48（0008 两条 LIKELY + support ≥35，按规则跳过） |
| R1 CHANGE / 训练 | 9 / 9；**12.80 GPU-h**；R1−R0 support Δ：mean **+4.45 dB**，median +2.19 dB |
| R2 触发 / CHANGE / 训练 | 6 / 0 / 0（6 条 R2 反思全部 NO_CHANGE） |
| 最终选轮 | R0 ×40，R1 ×8，R2 ×0 |
| 测试 | 3 套（R0-best / R0-last / final）×16,800 行；固定 GT-LQ；评估非确定性 ≤0.01 dB |
| 总训练成本 | **81.12 GPU-h**（≈1.69 GPU-h/chain，R1 计入） |

### 6.2 盲识别一致性（rep1 vs rep2）

- 整链 exact **10/24**；type-only 12/24；步数一致 19/24。
- 位置级 type **0.683**、spec **0.537**（41 个位置）。
- 事后对比 GT：rep1 整链 exact 9/24；rep2 15/24。

### 6.3 R0-only run-to-run（固定 support / 训练 seed）

| 口径 | DIV2K macro | within-SD (mean) | range (mean/max) | \|Δ\|>1 dB |
|---|---:|---:|---:|---:|
| best（support 选点） | 26.817 | 1.491 | 2.108 / 10.398 | 9/24 |
| last（固定 step 15092） | 25.526 | 1.535 | 2.170 / 10.398 | 10/24 |
| LSDIR best | 24.549 | 1.256 | 1.777 / 9.062 | 8/24 |

- 全场景 \|ΔPSNR\| 中位数仅 **0.048 dB**；但 7/24 超过 2 dB，呈现“多数稳定、少数灾难性”的双峰分布。
- **同 pipeline 的 10 个场景**：mean \|Δ\| **0.267 dB**（best）/ 0.421 dB（last），max 1.375 dB —— 训练/选点抖动。
- **pipeline 不同的 14 个场景**：mean \|Δ\| **3.423 dB**，max **10.398 dB** —— agent DC 差异主导。
- 按 GT 步数分层（DIV2K best，within-SD / mean range / max range）：1 步 1.326 / 1.875 / 9.894；2 步 **2.271 / 3.212 / 10.398**；3 步 0.876 / 1.239 / 5.860。
  - 2 步方差最大；3 步反而较小，部分原因是多个 3 步场景两次预测一致（如 0018/0019 bit-exact），不能解释为“3 步更容易”。

逐场景 R0-only best（DIV2K，原始评估；7 个不可复现 LQ 场景的敏感性见 §6.6）：

| 场景 | GT 步数 | rep1 | rep2 | mean | SD | range | pipeline 一致 |
|---|---:|---:|---:|---:|---:|---:|:--:|
| blind_0001 | 1 | 27.323 | 30.336 | 28.830 | 2.130 | 3.013 | N |
| blind_0002 | 1 | 32.162 | 32.005 | 32.083 | 0.111 | 0.157 | Y |
| blind_0003 | 1 | 28.129 | 28.171 | 28.150 | 0.030 | 0.043 | Y |
| blind_0004 | 1 | 24.990 | 34.883 | 29.936 | 6.996 | 9.894 | N |
| blind_0005 | 1 | 44.536 | 44.081 | 44.308 | 0.321 | 0.454 | Y |
| blind_0006 | 1 | 31.631 | 31.621 | 31.626 | 0.007 | 0.010 | Y |
| blind_0007 | 1 | 34.298 | 34.351 | 34.325 | 0.037 | 0.052 | Y |
| blind_0008 | 1 | 42.714 | 41.339 | 42.027 | 0.972 | 1.375 | Y |
| blind_0009 | 2 | 28.733 | 18.334 | 23.533 | 7.353 | 10.398 | N |
| blind_0010 | 2 | 20.741 | 30.978 | 25.860 | 7.239 | 10.237 | N |
| blind_0011 | 2 | 25.354 | 25.114 | 25.234 | 0.170 | 0.240 | N |
| blind_0012 | 2 | 23.428 | 23.590 | 23.509 | 0.114 | 0.162 | N |
| blind_0013 | 2 | 28.091 | 29.242 | 28.667 | 0.814 | 1.151 | N |
| blind_0014 | 2 | 27.055 | 26.900 | 26.977 | 0.109 | 0.155 | Y |
| blind_0015 | 2 | 29.502 | 29.220 | 29.361 | 0.200 | 0.282 | Y |
| blind_0016 | 2 | 21.673 | 24.741 | 23.207 | 2.169 | 3.068 | N |
| blind_0017 | 3 | 23.045 | 23.461 | 23.253 | 0.294 | 0.416 | N |
| blind_0018 | 3 | 17.500 | 17.571 | 17.535 | 0.051 | 0.072 | Y |
| blind_0019 | 3 | 20.019 | 20.092 | 20.055 | 0.051 | 0.072 | Y |
| blind_0020 | 3 | 16.694 | 15.993 | 16.344 | 0.496 | 0.702 | N |
| blind_0021 | 3 | 24.038 | 24.387 | 24.212 | 0.247 | 0.349 | N |
| blind_0022 | 3 | 25.506 | 23.342 | 24.424 | 1.530 | 2.164 | N |
| blind_0023 | 3 | 21.834 | 21.560 | 21.697 | 0.194 | 0.274 | N |
| blind_0024 | 3 | 15.521 | 21.381 | 18.451 | 4.144 | 5.860 | N |

### 6.4 全流程 R0→R1→R2

| 口径 | DIV2K macro | within-SD | range (mean/max) | LSDIR macro | within-SD |
|---|---:|---:|---:|---:|---:|
| R0-only best | 26.817 | 1.491 | 2.108 / 10.398 | 24.549 | 1.256 |
| Final（R0/R1/R2） | **27.654** | **0.856** | **1.210 / 13.043** | **25.240** | **0.724** |

- Final−R0：DIV2K **+0.837 dB**（7 个 R1 选中场景全部提升，+0.71…+5.56；其余 17 个 ±0.05 内不变）；LSDIR **+0.692 dB**（7/7 提升，+0.56…+4.73）。
- R1 使 pipeline 一致场景从 10/24 增至 **13/24**（0004、0010、0022 收敛）。
- **平均方差下降，但最坏情况上升**：0009 的 rep1 经 R1 提升 +2.65 test dB，rep2 为 NO_CHANGE，range 从 10.40 增至 **13.04 dB**。反思只在一条链生效时，会放大 rep 间差距。
- R1 选中场景的 test 增量（DIV2K / LSDIR）：0004 +4.87/+4.23、0009 +1.32/+0.90、0010 +5.56/+4.73、0016 +1.86/+1.61、0020 +0.71/+0.56、0022 +1.07/+0.86、0024 +4.71/+3.70。
- R2 在 6 条 support 提升 >2 dB 的链上全部 NO_CHANGE，说明 R1 已到达 support 可验证的上限（或残余为不可编码的随机角度/seed 失配）。

R1 CHANGE 明细（9 条；按 support PSNR 判定，未用 test）：

| 链 | R0 support | R1 support | Δ | R0 管线 | R1 管线 |
|---|---:|---:|---:|---|---|
| blind_0004_rep1 | 25.28 | 35.05 | +9.77 | blur_gaussian:1 | blur_motion:1 |
| blind_0009_rep1 | 31.64 | 33.83 | +2.19 | noise_speckle:1 + saturate_strengthen_YCrCb:3 | noise_poisson:2 + saturate_strengthen_YCrCb:3 |
| blind_0010_rep1 | 19.83 | 31.57 | +11.74 | blur_gaussian:1 + brightness_darken_shfit_HSV:4 | blur_gaussian:1 + brightness_darken_gamma_HSV:5 |
| blind_0016_rep1 | 20.25 | 25.71 | +5.46 | blur_gaussian:4 + compression_jpeg:2 | blur_motion:3 + compression_jpeg:2 |
| blind_0016_rep2 | 25.01 | 24.91 | -0.10 | blur_motion:4 + compression_jpeg:1 | blur_motion:4 + compression_jpeg:2 |
| blind_0020_rep1 | 16.44 | 18.20 | +1.76 | blur_gaussian:4 + contrast_weaken_stretch:4 | blur_glass:4 + contrast_weaken_stretch:4 |
| blind_0020_rep2 | 15.74 | 15.77 | +0.03 | blur_gaussian:4 + brightness_darken_gamma_RGB:1 + contrast_weaken_stretch:5 | blur_gaussian:4 + contrast_weaken_stretch:4 |
| blind_0022_rep2 | 21.26 | 23.28 | +2.02 | blur_gaussian:3 + compression_jpeg:1 | blur_glass:3 + compression_jpeg:1 |
| blind_0024_rep1 | 17.76 | 24.95 | +7.19 | brightness_darken_gamma_RGB:3 + saturate_strengthen_YCrCb:4 | noise_gaussian_RGB:4 + blur_gaussian:3 + saturate_strengthen_HSV:2 |

其余 37 条 R1 为 NO_CHANGE；0008_rep1/rep2 按 LIKELY+support≥35 跳过。最终选轮：40 R0 / 8 R1（0016_rep2 的 R1 support −0.10 dB，回退 R0）。

### 6.5 解释与限制

1. **方差是双峰的**：中位数 <0.05 dB，但 7/24 >2 dB；大差异来自离散的 DC 错误（漏/多噪声步、blur 子类型、全局步骤），不是连续 severity 漂移。
2. **训练/选点抖动参考**：同 pipeline 的 rep 间 mean |Δ| 0.27–0.42 dB、max 1.38 dB；评估本身非确定性 ≤0.01 dB。
3. **R1 不是普遍降低最坏方差**：平均 within-SD 从 1.49 降至 0.86 dB，但 0009 的 worst-case range 变大；应同时报告 mean 与 max。
4. **E6-A 全流程仍不是超参/架构搜索**：R0–R2 全部固定 Direct Swin+SG、seed 42；R1/R2 仅改退化管线。
5. API model/版本/温度 UNKNOWN；不能外推到其他 LLM。R1/R2 反思的模型配置与 R0 盲识别相同，但 API 随机性无法拆分。
6. 1 条 R1 反思链（0009_rep1）的 test 提升 +1.32 dB 小于其 support 提升 +2.19 dB；support→test 的迁移不保证等幅。
7. 3 步场景方差低是“预测一致”带来的选择效应，不代表 3 步退化本身更稳定。
8. **LQ 协议（重要）**：17/24 场景三套测试的 `lq_sha256` 完全一致；7 个已知私有 RNG 场景（0003/0005/0017/0018/0020/0022/0023）的 LQ 在 rep/进程间不同（0/350 匹配）。已用 `E1E2_fixed_lq` 物化 LQ 重评（§6.6）：mean Δ ≤0.002 dB、max 0.018 dB，修正后 macro 与结论不变。

### 6.6 fixed-LQ 敏感性（7 个不可复现 LQ 场景）

**问题**：主评估的 LQ 由各评估进程即时生成。7 个场景（0003/0005/0017/0018/0020/0022/0023）含 impulse/glass 私有 RNG，LQ 像素在 rep1/rep2 及 best/last/final 进程间不同（这 7 个场景 `lq_sha256` 0/350 匹配）；其余 17 个场景三套测试的 LQ 完全一致（11,900/11,900 行匹配）。

**方法**：复用 `rebuttal_runs/E1E2_fixed_lq` 中已物化的同一份 LQ（每场景 350 图），对 best/last/final 三套选点的 14 个受影响 run 重评（14 × 350 × 3 = 14,700 次）。其余 17 场景沿用主评估。

**结果**：

| 口径 | mean Δ(fixed−orig) DIV2K | max \|Δ\| | mean Δ LSDIR | max \|Δ\| | 修正 macro (DIV2K) | 修正 within-SD | 修正 max range |
|---|---:|---:|---:|---:|---:|---:|---:|
| best | −0.001 | 0.010 | +0.002 | 0.018 | 26.817 | 1.491 | 10.398 |
| last | −0.000 | 0.004 | +0.001 | 0.004 | 25.526 | 1.535 | 10.398 |
| final | −0.001 | 0.006 | +0.001 | 0.006 | 27.654 | 0.855 | 13.043 |

- 7 个场景逐 run 的最大偏差 0.018 dB（LSDIR best，blind_0005）；逐场景 rep range 变化 ≤0.003 dB。
- 修正后的 macro、within-SD、max range 与主表一致到 3 位小数；分层（1/2/3 步）与 pipeline 分组也不变。
- **结论**：LQ 进程差异对 E6-A 的 PSNR 与方差结论无实质影响；主表可直接使用，敏感性结果用于披露。

产物：`rebuttal_runs/E6A_full24/fixed_lq_sensitivity.json/.md`、`test_metrics_fixed_{best,last,final}_gpu*.jsonl`、`rebuttal_runs/scripts/eval_e6a_full24_fixed_lq.py`、`aggregate_e6a_full24_fixed_lq.py`。

### 6.7 泄露与口径审计（2026-10-30）

详见 `rebuttal_runs/E6A_full24/AUDIT.md`。要点：

- **support/test 重叠**：E6-A full24 的 support 全部复制自 exp41，像素 hash 全部匹配 **DIV2K_valid_HR（test）**（74/100 唯一 test 图）；E1/AutoIR 使用同一 support，属共用协议。排除本场景 5 张 support 后：R0 26.817→**26.819**，final 27.654→**27.659**（≤0.005 dB），重叠不解释高 PSNR。
- **Agent 禁读审计**：E6-B 98 个 Agent 0 次违规；E6-A 100 个 Agent 中仅 `blind_0001_rep2` 用 grep 读了 `rebuttal_plan/`，看到其他场景（0006/0011/0013/0017/0024）的 GT/失败提示，**未看到 0001 自身 GT**；剔除该 run 后 macro 约 −0.06 dB。
- **与 rule/LLM 的差异**：E6-A vs AutoIR mean +1.85 / **median −0.08 dB**；vs rule mean +2.54 / median +0.65 dB。增益集中在规则/AutoIR 灾难性 DC 错误场景（0008 +19.4、0010 +9.4、0015 +7.8、0024 +7.5、0009 +5.7、0021 +5.3 等）；E6-A rep2 GT 整链 15/24 vs 规则 5/24。26.8 是 DC 准确率与重尾，不是均匀抬高。
- **E6-B 三组对比 confound**：exp41 support 来自 DIV2K_valid，setA/setB 来自 DIV2K_train；三组表将同时报告 full test 与排除 exp41 5 张 support 的 common-test sensitivity。

## 7. 产物

```text
rebuttal_runs/E6A_full24/
├── challenges/{sid}_rep{1,2}/           # support + R0/R1/R2 输出（无 GT）
├── experiments/{name}[_R1]/            # checkpoints
├── logs/                                # 训练/评估日志
├── selection.jsonl / selection_last.jsonl / selection_final.jsonl
├── test_metrics_gpu*.jsonl              # R0 best，16,800 行
├── test_metrics_last_gpu*.jsonl         # R0 last，16,800 行
├── test_metrics_final_gpu*.jsonl        # 跨轮 final，16,800 行
├── blind_agreement.json/.md
├── blind_id_audit.json
├── reflection_plan.json / r2_plan.json
└── e6a_full24_summary.json/.md
rebuttal_runs/scripts/e6a_full24_*.py / e6a_full24_*.sh
```

