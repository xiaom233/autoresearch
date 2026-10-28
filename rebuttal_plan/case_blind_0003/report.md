# blind_0003 案例复盘：每一步决策是规则还是 LLM

> 本报告是**事后复盘**，不是盲识别过程的实时日志。文件夹内复制了原始 JSON、训练日志、评估日志、规则控制器产物和 5 对 support 图像（图像仅在本地，`.gitignore` 默认忽略 `.png`）。
>
> 场景：`exp41/challenges/phase4/blind_0003`
> GT（事后标注）：`blur_glass:2`，单步、局部 blur，无 noise / JPEG / 全局退化。
> 结论先说：**这个案例能证明 LLM 能补候选、排假阳性、提出新结构假设；不能证明 LLM 独立选出了最终正确管线，也不能证明 LLM 优于规则控制器。** 最终从 `gaussian:2` 切到 `glass:2` 发生在 R3，触发依据是 GT 训练 PSNR=19.33；这是一次 GT 反馈/近似 oracle 的修正，不是 support-only 的盲识别胜利。

---

## 1. 一页结论

| 问题 | 事实 | 归因 |
|---|---|---|
| R0 为什么选 `blur_gaussian:2`？ | `gm_ratio=0.532` 强 blur；PSNR 模式；gaussian:2=34.83 dB，lens:2=34.61，motion:2=25.76；30–40 dB → UNCERTAIN | **规则 + LLM**：候选由规则要求的三族 blur 保底，具体排序/severity 由 LLM 执行；gaussian 也是 DT 94% 的默认安全解 |
| R1 有没有新发现？ | 有。R1 补测了 R0 没测的 `blur_glass:2`，36.65 dB / CI 10/10，明显优于 gaussian:2 | **LLM 贡献**：glass 不在“至少 3 种 blur”强制清单里；是 Agent 自己发现 R0 漏测 glass 后补测的 |
| R1 为什么没直接采用正确解？ | `REFLECTION_MECHANISM.md §二` 规定“PSNR 改善 ≠ blur 子类型正确（63-case 仅 1/34=2%）”；因此只把 glass 放进 alternatives，主预测仍保留 gaussian | **规则压制 LLM**：规则明确要求不要因为 PSNR 更高就换 blur 子类型 |
| R2 做了什么？ | 重跑信号，探索 gamma、JPEG-only、lens severity、quantization 等；全部被 PSNR 拒绝；宣布 BEYOND_CAPABILITY，仍保留 gaussian | **LLM 探索 + 规则判定**：候选是 LLM 提的，拒绝依据是确定性退化的 PSNR 规则 |
| R3 为什么终于切到 glass？ | 触发条件：GT 训练 PSNR=19.33（737 张 GT 评估）< 30，按协议说明 gaussian:2 错误；把 R1 已发现的 glass:2 提升为主预测 | **GT 反馈 / 近似 oracle**：不是 support-only；LLM 的动作只是“把 R1 的候选提升”，不是新发现 |
| R4 做了什么？ | 提出双 blur `glass:2 + gaussian:1`，37.30 dB，但 CI 只有 6/10；反向顺序 36.75 / 7/10；最终测试 27.03，低于 R3 的 27.20 | **LLM 新假设 + 违规**：项目规定 blur 最多出现一次；双 blur 不可能成为 GT，属于 LLM 未检查硬约束 |
| 规则控制器（E1）表现如何？ | 同一场景，规则控制器 R1 直接选 `glass:2`；固定 LQ 测试 **27.94 / 24.19 dB**，AutoIR（R1 gaussian）**25.53 / 22.33 dB** | **规则控制器反而更好**：它的候选表里预置了 glass，并用 PSNR 排名直接采纳；AutoIR 的 2% 规则阻止了同样的采纳 |
| 这个案例能否作为“LLM 价值”的主证据？ | **不建议**。它更适合作为“LLM 候选发现 + 规则过度保守 + GT 反馈修复”的失败模式/局限性案例 |  |

---

## 2. 场景、数据和评估口径

### 2.1 场景与 GT（事后）

- 目标：`exp41/challenges/phase4/blind_0003/degraded.png`；clean：`clean.png`。
- 5 对 support：`clean_0..4.png` / `degraded_0..4.png`，与 GT 使用同一退化（`--same-image`）。
- GT：`blur_glass`，severity 2，单步。
- 测试集：DIV2K_valid_HR 100 张 + LSDIR val1/GTmod4 250 张；固定 LQ；指标 `psnr_rgb` / 高斯 11×11 σ=1.5 SSIM / VGG LPIPS。
- GT 只在**事后评估**和 R3/R4 的 GT 训练 PSNR 触发中使用；R0–R2 的盲识别阶段不应看到 GT。

### 2.2 时间线与产物可信度

| 产物 | 时间 | 说明 |
|---|---|---|
| R0 训练日志 `exp41_R0_blind_0003.log` | 2026-06-30 18:02 起 | 实际训练日志（原 exp40 `direct_swin_blind_0003`），纯 Swin，无 SimpleGate |
| R1 训练日志 `exp41_R1_blind_0003.log` | 2026-07-01 16:08 起 | 实际训练日志（原 exp40 `r1_blind_0003`），Swin+SimpleGate |
| R2 训练日志 | 2026-07-14 09:19 起 | exp41 `exp41_v1_blind_0003`，Swin+SimpleGate |
| R3 训练日志 | 2026-07-15 02:34 起 | exp41 `exp41_R3_blind_0003`，Swin+SimpleGate，glass:2 |
| R4 训练日志 | 2026-07-15 06:34 起 | exp41 `exp41_R4_blind_0003`，Swin+SimpleGate，双 blur |
| `predicted_params.json`、`full_analysis.json`、`qwen_noise_gate.json` | 2026-07-14 06:32 | 晚于 R0/R1（6/30、7/1），早于 R2（7/14 09:19）；**不是 R0/R1 当时的原始文件，可能是重建/整理后的版本** |
| `predicted_params_R1.json`、`predicted_params_qwen.json` | 2026-07-14 06:32 | 同上 |
| `training_strategy.json` | 2026-07-14 09:02 | 早于 R2，不是 R0 当时的决策记录 |
| `predicted_params_R3.json`、`full_analysis_R3.json` | 2026-07-15 01:37 / 01:29 | R3 前 |
| `predicted_params_R4.json`、`full_analysis_R4.json`、`reflection.json` | 2026-07-15 06:10 / 06:04 | R4 前/后 |
| 737 张 GT 门控 R2=19.33 | 2026-07-14 15:26 | R2 训练后、R3 前，是 R3 的实际触发证据 |
| 737 张 GT 门控 R3=24.33 | 2026-07-15 06:00 | R3 训练后、R4 前，是 R4 的实际触发证据 |
| 标准化 DIV2K/LSDIR 最终指标 | 2026-07-27 ～ 07-30 | 所有轮次结束后的统一重评估，**不是决策时看到的指标** |

**结论：** 训练日志是 contemporaneous 的；challenge 目录里的 JSON 是后来整理/重建的。因此本报告把每个结论分成“A=有 contemporaneous 日志”“B=有持久化 JSON 但时间晚于决策”“C=只有摘要、无原始测试日志”三个可信度等级。R2/R3 的部分候选测试只有摘要（见 §10），不能声称是完整逐步日志。

### 2.3 训练配置

| Round | 架构 | 训练管线 | 初始化 | best checkpoint | 训练时长 | 参数量 |
|---|---|---|---:|---:|---:|---:|
| R0 | Swin（无 SG） | gaussian:2 | scratch | step 7546（Set14 26.43） | 5206.6 s | 0.5M |
| R1 | Swin+SG | gaussian:2 | scratch | step 15092（Set14 25.24） | 5084.5 s | 0.4M |
| R2 | Swin+SG | gaussian:2 | scratch | step 11319（5-support 25.76*） | 5089.0 s | 0.4M |
| R3 | Swin+SG | glass:2 | scratch | step 11319（5-support 28.24*） | 5058.0 s | 0.4M |
| R4 | Swin+SG | glass:2 + gaussian:1 | scratch | step 7546（5-support 27.38*） | 5067.0 s | 0.4M |

\* R2–R4 的验证集是挑战自带 5 对 support（`PairedValDataset: 5 image pairs`），但训练日志的打印标签仍写成 “Set14”；数值是 5 对 support 的 PSNR，不是官方 Set14。R0/R1 的验证集是 Set14 14 张，没有 paired support。**验证集在 R1→R2 之间发生了变化，这是 R2 及以后 checkpoint 选择的一个混淆因素。**

所有轮次：15,094 步、2 epoch、batch 16、LR 5e-4、L1 loss、Direct from scratch。`train.py` 默认 seed 42（日志未打印；E1 任务显式 `AR_SEED=42`）。

---

## 3. 评估口径与 checkpoint 审计（重要）

### 3.1 三套不同的测试数字

同一轮模型在不同评估脚本/不同 checkpoint 下会得到不同的 DIV2K/LSDIR PSNR。文件夹里保留了三套证据：

| Round | 训练时选出的 best step | best step 的验证 PSNR | 737 张 GT 门控（决策时） | 最终统一评估 @step7546 | 数值最后一步 @step15092 |
|---:|---:|---:|---:|---:|---:|
| R0 | 7546 | Set14 26.43 | — | 23.50 / 20.63 | 25.48 / 22.27 |
| R1 | 15092 | Set14 25.24 | — | 25.50 / 22.31 | 25.32 / 22.18 |
| R2 | 11319 | support 25.76 | **19.33** | 21.75 / 18.87 | 25.33 / 22.22 |
| R3 | 11319 | support 28.24 | **24.33** | 27.20 / 23.52 | 26.25 / 23.01 |
| R4 | 7546 | support 27.38 | — | 27.03 / 23.40 | 26.52 / 23.11 |

格式：DIV2K PSNR / LSDIR PSNR（RGB）。完整 SSIM/LPIPS 见 `metrics/metrics_provenance.json`。

### 3.2 为什么会有两套“最后一步”？

- `exp41/scripts/reeval_parallel.sh` 明确选择 `*step15092.pt`（数值意义的最后一步），得到上表最后一列。
- `exp41/scripts/reeval_all.py` 使用 `pts = sorted(glob(...))`，再取 `pts[-1]`。文件名的字典序是：
  `step11319 < step15092 < step3773 < step7546`，
  所以 `pts[-1]` 实际是 **step7546**，不是 step15092。上表“最终统一评估 @step7546”就是由此而来。
- `exp41/summarize/all_rounds_summary.md` 把这一列标成 “last ckpt”，这个标签**不准确**；它是字典序最后，不是数值最后。`reeval_parallel.sh` 的 step15092 数字则没有进入该汇总。
- 因此，**R2 的 “-3.75 dB 崩溃” 和 R3 的 “+5.45 dB 恢复” 是 step7546 口径下的现象**；换到 step15092 口径，R2=25.33、R3=26.25，R2 并没有崩溃，R3 的增益也只有 +0.92 dB。737 张 GT 门控则显示 R2=19.33 → R3=24.33（+5.00 dB），这是决策时实际使用的证据，但它的 checkpoint 路径没有写进日志（只打印了 `arch=swin`）。
- **对审稿人应诚实报告：** 轮次级 PSNR 差异对 checkpoint 选择敏感；不能把 R2 的下降单独归因于“在错误退化上过拟合”，也不能把 R3 的 +5.45 当作稳定的方法增益。能稳定支持的是“glass 管线优于 gaussian 管线”这一方向（R1 的 CI 10/10、R3 的 GT 门控、E1 规则控制器都支持），而不是精确的 dB 数。

### 3.3 规则/LLM 归因的分类

| 标签 | 含义 |
|---|---|
| `[规则]` | 由确定性阈值/协议直接决定，有文档条款 |
| `[LLM]` | Agent 自己提出的候选、假设或排序，规则没有强制 |
| `[规则+LLM]` | 规则规定了搜索空间/验证方法，LLM 在其中做选择 |
| `[GT反馈]` | 使用了 GT 训练 PSNR 作为触发或证据，属于 oracle 反馈 |
| `[LLM违规]` | LLM 的候选违反了项目的硬约束 |
| `[训练随机性]` | 不是决策，而是同配置重训/不同 checkpoint 造成的波动 |

---

## 4. R0：盲识别（2026-06-30）

### 4.1 原始信号（`full_analysis.json`，A/B）

| 步骤 | 脚本 verdict | 关键信号 | 规则含义 |
|---|---|---|---|
| compression | `MILD_COMPRESSION_POSSIBLE` | `block_boundary=1.0014`（未 >1.1）、`unique_G=256`（未 <200）、`dct_zero_ratio=0.6244` | block_boundary/unique_G 是 0% FP 的 Tier S 信号，均未触发；dct_zero 只是“可能”，规则要求仍测 JPEG |
| noise | `GAUSSIAN_YCrCb` | `noise_prior_sigma=1.11`、`spatial_corr=0.713`、`cross_channel_corr=0.731`、`residual_intensity_corr=0.159` | sigma < 2 → `NO_NOISE`；`extreme_pct>50% AND sigma<5` → 忽略 IMPULSE 假阳性 |
| blur | `BLUR_UNLIKELY` | `gm_ratio=0.532`（<0.85 强 blur）、`spectral_slope_ratio=1.069`（<1.08 阈值） | 两个信号矛盾；`should_still_test=true`，规则要求测 2–3 个 blur 候选 |
| global | `NO_GLOBAL` | `variance_ratio=0.954`、`mean_shift=-0.5`、`clipped_fraction=0.0027` | 无 contrast/brightness/saturation/oversharpen |
| 推荐架构 | — | `AR_USE_DUAL_BRANCH=1`（工具建议） | 与 `finetune_strategy.md §七` 的 “DualBranch 盲识别下 0/24 胜” 冲突，最终未用 |

### 4.2 R0 逐步决策

| # | 决策 | 证据 | 归因 |
|---:|---|---|---|
| 1 | 测 `gaussian:2` | `gm_ratio=0.532`；PSNR 模式（sigma<2）；34.83 dB / CI 8/10 | `[规则+LLM]`：规则要求测 blur；选 gaussian:2 是 LLM 的首个假设，也是 DT 94% 的安全默认 |
| 2 | severity 扫 3 / 1 / 4 | 32.89 / 32.91 / 30.55，均低于 sev=2 | `[规则]`：SKILL.md 强制 severity 扫描；30–40 dB 只能微调 severity |
| 3 | 测 `gaussian:2 + JPEG` | `dct_zero_ratio=0.624`；JPEG:3 → 29.76，JPEG:1 → 32.04，均下降 | `[规则+LLM]`：规则要求对 block/dct 弱信号做 PSNR 验证；具体组合/severity 由 LLM 选择 |
| 4 | 测 `motion:2` | 25.76 dB，明显差 | `[规则]`：SKILL.md 要求至少覆盖 3 种 blur（gaussian+lens+motion） |
| 5 | 测 `gaussian:2 + noise` | 21.40 dB，下降 | `[规则+LLM]`：noise 假阳性排除；规则规定 sigma<2 → NO_NOISE |
| 6 | 测 `lens:2` | 34.61 dB，与 gaussian:2 仅差 0.22 dB | `[规则]`：跨族 blur 测试；MTF 不可区分 |
| 7 | 接受 `gaussian:2` 为主预测，`lens:2` 为 alternative | 34.83 vs 34.61；30–40 dB → UNCERTAIN；`lens:2` 是跨族备份 | `[规则+LLM]`：PSNR 排名 + 规则要求保留跨族候选；gaussian 是安全默认 |
| 8 | 不测 `glass` | R0 只测了 gaussian/lens/motion；glass 不在“至少 3 种”清单 | **LLM 的候选空间缺口**：这是后来 R1 发现 GT 的入口 |

**R0 的关键点：** `gm_ratio=0.532` 只说明“有 blur”，不能区分 gaussian/glass；`spectral_slope_ratio=1.069` 又低于阈值，脚本甚至给出 `BLUR_UNLIKELY`。LLM 正确地覆盖了 gaussian/lens/motion，但**没有覆盖 glass**。这不是规则强制要求的，而是 Agent 自己的候选空间不完整。

### 4.3 Qwen 辅助路径

- `qwen_noise_gate.json`：`has_noise=true`，`raw=YES`，28.1 s。
- `predicted_params_qwen.json`：`noise_gaussian_RGB:1 → blur_gaussian:2 → compression_jpeg:2`，模型字符串 `qwen3.7-plus-2026-05-26`，175.3 s，1 次尝试。
- 结果：**没有被采用**。noise prior sigma=1.11 < 2，规则判 NO_NOISE；Qwen 的 noise/JPEG 都被 PSNR 拒绝。
- **注意：** 文档曾写 Qwen-Max/Qwen-VL-Max，但实际记录的是 `qwen3.7-plus-2026-05-26`；这是一个模型名不一致的审计点。Qwen 是辅助 VLM，不是主 Skill Agent；不应把 Qwen 的输出与主 Agent 的推理混为一谈。

### 4.4 R0 架构：规则还是 LLM？

`training_strategy.json`（2026-07-14 09:02，晚于 R0 的 6/30）写着：

```json
{
  "strategy": "Direct",
  "architecture": "Swin",
  "components": ["SimpleGate"],
  "lr": 5e-4,
  "epoch_budget": 2,
  "signal_basis": "block_boundary=1.0014 ... variance_ratio=0.954 ... gm_ratio=0.532 ... 无motion blur → 不需OCAB ... SimpleGate默认常开"
}
```

逐项归因：

| 决策 | 值 | 依据 | 归因 |
|---|---|---|---|
| Direct vs Ft | Direct | `finetune_strategy.md` Section 六：纯局部单退化 → Direct | `[规则]` |
| Swin vs OCAB | Swin | 无 motion；`anisotropy_ratio` 不适用；Section 七 Tier B blur → Swin 基线 | `[规则]` |
| SimpleGate | 开（R1 起） | exp39：-32K 参数，8/9 退化安全，默认常开 | `[规则]` |
| FiLM-GCM | 不开 | `variance_ratio=0.954 ≈ 1.0`，无 contrast；无结构化 motion/JPEG 高置信信号 | `[规则]` |
| ColorPre | 不开 | 同上，`variance_ratio` 未触发 Tier A | `[规则]` |
| OCAB | 不开 | 无 motion，`anisotropy_ratio` 未触发 | `[规则]` |
| DualBranch | 不开 | `full_analysis` 建议开；`finetune_strategy.md §七` 明确“盲识别下不推荐，0/24 胜(exp37)”；Agent 选择了后者 | `[规则+LLM判断]`：两条规则冲突，Agent 用更具体的 §七 覆盖工具建议 |
| LR / epoch / batch | 5e-4 / 2 / 16 | CLAUDE.md + finetune_strategy.md | `[规则]` |

**但实际 R0（6/30）是纯 Swin，没有 SimpleGate。** `training_strategy.json` 是 7/14 才写的，不能作为 R0 当时的决策证据。R0→R1 的架构变化（加 SG）更像是 exp40 内部 R0→R1 的策略升级/重训，而不是 R0 盲识别阶段的 LLM 发现。这一点直接影响“R0→R1 +2.00 dB”的归因（见 §5.4）。

---

## 5. R1：反思（2026-07-01，文件重建于 7/14）

### 5.1 触发

- `predicted_params.json` 的 `overall_verdict=UNCERTAIN`，`analysis.tier=1`，PSNR 34.83 在 30–40 dB。
- CLAUDE.md 反思触发条件：`verdict=UNCERTAIN → 必须反思`。
- **归因：`[规则]`**。R1 的启动不是 LLM 自主决定，而是协议强制。

### 5.2 R1 逐步决策

| # | 决策 | 证据 | 归因 |
|---:|---|---|---|
| 11 | 补测 `blur_glass:2` | 36.65 dB / CI 10/10，优于 gaussian:2 的 34.83 / 8/10 | **`[LLM]`**：R0 已经测了 gaussian/lens/motion，满足“至少 3 种 blur”的最低要求；glass 是 Agent 自己发现“R0 没测过”后补的候选。这是本案例中 LLM 最有价值的动作 |
| 12 | `glass:1` | 35.80 dB / 8/10，低于 glass:2 | `[规则]`：severity 扫描 |
| 13 | `glass:3` | 33.05 dB / 8/10，低于 glass:2 | `[规则]`：severity 扫描 |
| 14 | `verify_signals` 验证 glass:2 | MATCH（2✓/0✗：blur:spectral_slope、blur:wiener_kernel） | `[规则]`：SKILL.md 要求信号验证，不能只用 PSNR |
| 15 | 保持 `gaussian:2` 为主预测，glass:2 只放 alternative | `REFLECTION_MECHANISM.md §二`：63-case 中 34 次 PSNR 改善只有 1 次 GT 匹配（2%）；gaussian↔glass 互换代价 2–7 dB | **`[规则]` 压制 `[LLM]`**：规则明确说“PSNR 更高 ≠ 子类型正确”，所以不允许因为 36.65 > 34.83 就切换 |

### 5.3 这一步的深层含义

R1 已经**找到了正确答案**：`glass:2`，CI 10/10，verify_signals MATCH。但协议不允许采纳。这是本案例最重要的“规则 vs LLM”冲突：

- LLM 的贡献是**候选空间扩展**：R0 漏掉 glass，R1 补上并验证。
- 规则的贡献是**防止 PSNR 误导**：63-case 经验说 blur 子类型 PSNR 只有 2% 的 GT 命中率。
- 结果是**规则赢了统计、输了本场景**：在这个具体场景，PSNR 和 CI 都指向了 GT，但规则一刀切地不信任它们，导致正确候选被压在 alternatives 里，直到 R3 才被 GT 反馈救回来。

### 5.4 R1 的 +2.00 dB 是什么？

R0 23.50 / 20.63 → R1 25.50 / 22.31（step7546 口径），表面上是 +2.00 / +1.68 dB。但不能直接归因于“LLM 反思”：

1. **管线没变**：R0 和 R1 都是 `gaussian:2`。
2. **架构变了**：R0 纯 Swin，R1 Swin+SimpleGate。
3. **验证集/checkpoint 选择变了**：R0 用 Set14 选 step7546；R1 用 Set14 选 step15092。
4. **同配置重训波动很大**：R2 与 R1 架构/管线相同，R2 step7546 只有 21.75 dB（比 R1 低 3.75 dB）；R2 step15092 是 25.33（与 R1 的 25.32 基本一样）。所以 R0→R1 的 +2.00 既包含 SG 的可能收益，也包含训练/checkpoint 波动，**不能作为 LLM 反思有效的干净证据**。
5. 7/14 才写的 `training_strategy.json` 建议 R0 也用 SimpleGate；实际 R0 没用。这说明 R0/R1 的架构差异更像执行策略差异，而不是 R0 阶段的规则/LLM 决策差异。

**建议在 rebuttal 中写：** R1 的 +2.00 dB 是“架构默认项 + 重训 + checkpoint 选择”的混合结果，不能单独归功于反思；本案例的反思贡献应写成“发现 glass 候选”，而不是“+2 dB”。

---

## 6. R2：第二轮反思（2026-07-14）

### 6.1 触发与探索

- 触发：`verdict=UNCERTAIN` + R1 未解决；反思协议允许多轮。`[规则]`
- R2 重跑 `full_analysis_R2.json`；信号与 R0 基本一致（`gm_ratio=0.532`、`dct_zero_ratio=0.624`、`sigma=1.11`、`variance_ratio=0.954`），只是 `residual_intensity_corr` 从 0.1589 变为 0.1551。**信号没有发生实质变化**。
- R2 的探索（摘要级，C 级证据）：
  - `compression_jpeg:3` → 28.12 dB / CI 4/10 → JPEG-only 明显差。
  - `blur_lens:1` → 33.19 / 8/10；`blur_lens:3` → 34.60 / 9/10 → lens 全 severity 仍不如 glass。
  - `quantization_median:3` → 25.68 / CI 1/10 → 量化假设排除。
  - 另有 gamma 组合（第 15–17 轮）：22.71–22.88 dB，全部拒绝。
  - JPEG sev 2/4（第 18–19 轮）：31.16 / 28.09，拒绝。
  - contrast 组合（第 20 轮）：16.84，拒绝。
- **归因：`[LLM]` 提出假设，`[规则]` 用 PSNR/信号判定拒绝。** 这些是真实的 LLM 探索，但都是负结果；没有改变主预测。

### 6.2 R2 的结论

- `r2_summary` 宣布 `BEYOND_CAPABILITY`，保留 gaussian:2，glass:2 仍为 alternative。
- **归因：`[规则]` 的终止条件**（连续无改善 → BEYOND_CAPABILITY）；LLM 接受了该结论。
- R2 没有再次提升 glass，尽管 R1 已经有 CI 10/10 的证据。这不是 LLM 没想到，而是规则不允许。**这是本案例中规则过度保守的最直接证据。**

### 6.3 R2 的指标争议

- 737 张 GT 门控：**R2 = 19.33 dB**（2026-07-14 15:26，R3 前的决策依据）。
- 标准 DIV2K/LSDIR：step7546 口径 21.75 / 18.87；step15092 口径 25.33 / 22.22。
- 因此“R2 崩溃”只在 step7546 口径成立；换 checkpoint 就不成立。R2 的 19.33 是 **737 张混合集** 的数值，不要和 DIV2K 100 张的 21.75 直接比较。

---

## 7. R3：GT 反馈后的修正（2026-07-15）

### 7.1 触发

- R3 summary 原文：`trigger: GT PSNR=19.33 (极低, gaussian:2 prediction clearly WRONG) + R2=BEYOND_CAPABILITY`。
- `SKILL.md` / CLAUDE.md：`GT PSNR < 30 + verdict=POOR/UNCERTAIN → 盲识别部分错误 → Skill 子 Agent 反思`。
- **归因：`[GT反馈]`**。R3 的启动不是 support-only；它使用了 GT 训练 PSNR。对于“盲识别系统”的公平评估，这一步应标成 oracle-assisted / GT-feedback，而不是 blind。

### 7.2 R3 逐步决策

| 决策 | 证据 | 归因 |
|---|---|---|
| 把 `blur_glass:2` 从 alternative 提升为 R3 主预测 | R1 已测得 36.65 / CI 10/10；R3 重验 36.59 / CI 10/10 / MATCH；GT PSNR=19.33 证明 gaussian:2 错误 | **`[GT反馈]` + `[LLM]`**：候选是 R1 的 LLM 发现；提升动作由 GT 触发；没有新的候选发现 |
| 不再测试新候选 | R3 只做 glass:2 验证；summary 明确 “No new tests beyond glass:2 verification” | `[规则]`：反思预算/终止；也是 `[LLM]` 的保守选择 |
| 保持 Swin+SG 架构 | 与 R1/R2 相同；无 motion/contrast 信号 | `[规则]` |

### 7.3 R3 的指标

- 737 张 GT 门控：**24.33 dB**（2026-07-15 06:00，R4 前的决策依据）。
- 标准 DIV2K/LSDIR：
  - step7546 口径：27.20 / 23.52（相对 R2 的 21.75 / 18.87，+5.45 / +4.65）。
  - step15092 口径：26.25 / 23.01（相对 R2 的 25.33 / 22.22，+0.92 / +0.79）。
- 方向一致：glass 优于 gaussian；幅度依赖 checkpoint。737 门控的 19.33→24.33（+5.00）是决策时实际看到的增益。

### 7.4 R3 到底证明了什么？

- **证明了**：R1 的 glass 候选是正确的；一旦允许采纳，GT 性能大幅改善。
- **没有证明**：LLM 能在 support-only 条件下自行采纳 glass；R1/R2 都选择了 gaussian；R3 的采纳依赖 GT PSNR=19.33。
- **也没有证明**：LLM 优于规则控制器。E1 规则控制器在同一场景 R1 直接选 glass，固定 LQ 27.94 / 24.19；AutoIR 的 R1 gaussian 只有 25.53 / 22.33。规则控制器没有 GT 反馈，反而更早选对。**这是本案例最不利于“LLM 独立价值”的证据，必须如实写。**

---

## 8. R4：双 blur 新假设（2026-07-15）

### 8.1 触发

- R4 summary：`trigger: GT PSNR=24.33 (R3 glass:2 +5dB improvement but still low) + unexplored directions exist`。
- **归因：`[GT反馈]`**：R4 的启动也用了 GT。它不是 support-only 的自主探索。

### 8.2 R4 逐步决策

| # | 决策 | 证据 | 归因 |
|---:|---|---|---|
| 21 | 重新验证 `glass:2` | 36.58 / CI 10/10 | `[LLM]` 基线复核 |
| 22–26 | `glass:2 + JPEG:1..5` | 33.10 / 32.02 / 30.38 / 28.52 / 26.04，全部下降 | `[LLM]` 系统排除 JPEG；`[规则]` PSNR 判定 |
| 27–28 | `glass:4/5` | 30.45 / 28.60，下降 | `[LLM]` severity 扩展 |
| 29–30 | `glass:2 + noise` | 21.47 / 25.72，下降 | `[LLM]` 排除 noise |
| 31 | `lens:3` | 34.60，低于 glass | `[LLM]` |
| 32 | `motion:1` | 26.70，低于 glass | `[LLM]` |
| 33 | **`glass:2 + gaussian:1`** | **37.30 / CI 6/10** | **`[LLM]` 新结构假设**；但 CI 从 10/10 掉到 6/10，增益仅 +0.72 dB |
| 34 | 反向 `gaussian:1 + glass:2` | 36.75 / CI 7/10 | `[LLM]` 顺序交叉验证 |
| 35 | `lens:2` 复测 | 34.61 / 8/10 | `[LLM]` |
| 最终 | 采用双 blur 作为 R4 主管线 | 37.30 > 36.58，且“gm_ratio 对单 blur 异常强” | **`[LLM违规]`**：见下 |

### 8.3 为什么 R4 是 LLM 失败，而不是 LLM 创新成功

1. **违反硬约束**：CLAUDE.md 明确规定“blur、noise、compression 各最多出现一次，最多 3 步”。`glass:2 + gaussian:1` 使用了两个 blur，**在这个项目里不可能成为合法 GT**。LLM 没有检查这条约束就提出了它。
2. **证据不足**：37.30 vs 36.58 只有 +0.72 dB，低于协议通常要求的 2 dB；CI 从 10/10 降到 6/10；反向顺序 36.75 / 7/10 也没有恢复 CI。`gm_ratio=0.532` 并没有“单 blur 通常 >0.6、所以双 blur”的校准依据，这是 LLM 自己发明的解释。
3. **动机是 GT 差距**：R4 summary 明确用“单 blur 训练只有 24.33 dB，双 blur 可能解释 12 dB gap”来支持假设。这是从 GT 结果反推结构，属于 oracle 反馈驱动的假设，不是信号驱动的盲发现。
4. **没有稳定提升**：R4 在 step7546 口径是 27.03 / 23.40，低于 R3 的 27.20 / 23.52；在 step15092 口径是 26.52 / 23.11，反而略高于 R3 的 26.25 / 23.01。两种口径方向不一致，说明 +0.72 dB 的仿真增益没有转化为稳定的测试增益。无论哪个口径，R4 都不是跨数据集一致的改进，且管线违规。

**归因写法：** R4 是 LLM 的“结构想象力”展示，也是一个反面教材：没有硬约束检查、没有 CI 门控、用 GT 反推，最终提出一个不可能正确的管线。它不能作为 LLM 价值的正面证据。

---

## 9. 规则控制器（E1）对比：本案例最重要的反证

> 更完整的 controller 层构造（7 层规则、冻结阈值、R0→R1→R2 状态机、24 场景整体结果）见 `rebuttal_plan/controller_comaprison.md`。

E1 的规则控制器是确定性的：候选表、PSNR 打分、支持集选 checkpoint。它在 blind_0003 上的决策如下：

### 9.1 候选打分（`E1_decision_blind_0003.json`）

| 候选 | PSNR（仿真） | 说明 |
|---|---:|---|
| **`blur_glass:2`** | **33.715** | 最高 |
| `blur_gaussian:2` | 32.339 | R0 聚合结果 |
| `blur_lens:2` | 32.079 | |
| `gaussian:2 + JPEG:1` | 30.405 | |
| `gaussian:3` | 30.298 | |
| `gaussian:1` | 29.976 | |
| `gaussian:4` | 28.006 | |
| `motion:2` | 24.563 | |
| `gaussian:2 + noise` | 信号 0.66 / PSNR 25.18 | |

- R0：规则控制器选 `gaussian:2`，支持集 best 25.84。
- R1：规则控制器直接选 `glass:2`（最高 PSNR），支持集 best 28.40，训练并测试。
- `trigger_R2=false`：规则控制器认为 R1 已经改善，停止。

### 9.2 固定 LQ 测试（E1，350 张/场景）

| 控制器 | 选中管线 | DIV2K PSNR | DIV2K SSIM | DIV2K LPIPS | LSDIR PSNR | LSDIR SSIM | LSDIR LPIPS |
|---|---|---:|---:|---:|---:|---:|---:|
| **规则控制器** | R1 `glass:2` | **27.937** | **0.8151** | **0.3147** | **24.192** | **0.7414** | **0.3471** |
| AutoIR（Skill Agent） | R1 `gaussian:2` | 25.526 | 0.7402 | 0.3620 | 22.332 | 0.6480 | 0.4003 |

**规则控制器 +2.41 / +1.86 dB，且 SSIM/LPIPS 全面更好。** 注意口径：AutoIR 的 25.53/22.33 是 E1/A0 选定的 **R1 gaussian** 模型（step7546）；规则控制器是 **R1 glass** 模型（step3773）。两者都是固定 LQ 的同口径评估。AutoIR 的 R3 glass 27.20/23.52 是 GT 反馈后的模型，不属于公平盲选对比，且 checkpoint 选择不同；不能把 27.94 与 27.20 的差异直接当作“规则 vs LLM 架构”差异。这不是因为规则控制器“更懂 glass”，而是因为：

- 它的候选表里**预置了 glass**（规则枚举，不是发现）；
- 它用 PSNR 排名直接采纳最高分，没有“blur 子类型 PSNR 只有 2% 可信”的保守规则；
- AutoIR 的 LLM 在 R1 已经发现了 glass（36.65 / 10/10），但被 2% 规则压住，最终选了 gaussian。

**因此：**
- 如果审稿人问“这个案例是否说明 LLM 比规则更好？”答案是**否**。在这个场景，简单规则更好。
- 如果问“LLM 是否有任何规则没有的贡献？”答案是**有**：R1 的 glass 候选是 LLM 补出来的；规则控制器的 glass 是人工预置的。LLM 的贡献是**候选空间扩展**，不是**选择策略优越**。
- 这个案例最适合用来说明“规则控制器 + LLM 候选发现”的互补，而不是“LLM 取代规则”。

---

## 10. 数据一致性审计（必须告知审稿人）

### 10.1 `reflection.json` 不是完整账本

- `iterations` 数组只有 **35 条**（R0 10 条、R1 4 条、R2 6 条、R4 15 条）；**没有 R3 的独立 iteration 条目**。
- `r2_summary` 却引用了 `round 44`（JPEG-only）、`round 48/49`（lens:1/3）、`round 50`（quantization_median:3）等**不在 `iterations` / `psnr_ranking` 里的测试**。
- `r3_summary` 引用 `[round 3] blur_glass:2 → 36.59`，也不在 35 条里。
- 结论：真实候选测试数量 **至少 50 次**（按 R2 自己的编号），持久化账本只保留了其中 35 条；R2/R3 的部分测试只有摘要（C 级证据）。这影响“每一步都可追溯”的强声明，但不影响主要结论（glass 是 R1 发现、R3 由 GT 反馈采纳、R4 双 blur 违规且无提升）。

### 10.2 时间戳与文件状态

- R0/R1 的训练是 6/30、7/1；但 challenge 目录的 `predicted_params.json`、`full_analysis.json`、`qwen_noise_gate.json`、`predicted_params_R1.json` 都是 7/14 06:32 的版本。它们可能是后来重建的，不是 R0/R1 当时的文件。
- `training_strategy.json` 是 7/14 09:02，晚于 R0/R1，不能作为 R0 架构选择的实时证据。
- R3/R4 的 JSON 是 7/15 的，时间上先于对应训练（R3 01:37 → 训练 02:34；R4 06:10 → 训练 06:34），这两轮倒是 contemporaneous 的。
- **建议：** 在 rebuttal 中把“决策过程”写为“根据持久化 reflection ledger 重建”，不要声称每个 JSON 都是当时写下的原始文件。

### 10.3 checkpoint 选择 bug

- 见 §3.2。`reeval_all.py` 的字典序 `pts[-1]` = step7546，导致最终汇总不是数值最后一步，也不是训练时选出的 best step。
- 这会影响 R0/R2/R3/R4 的轮次级 dB 归因。建议后续统一为“训练时 best support/Set14 checkpoint”或“数值最后一步”，并在所有表格中写明。
- 本报告同时列出 step7546 和 step15092 两套数字，避免单一数字误导。

### 10.4 Qwen 模型名不一致

- 实际记录：`qwen3.7-plus-2026-05-26`（`predicted_params_qwen.json`），不是文档里的 Qwen-Max/Qwen-VL-Max。
- Qwen gate (`has_noise=YES`) 与主 Agent 的 sigma<2 (`NO_NOISE`) 冲突；最终采用主 Agent 的信号判断，Qwen pipeline 没有被使用。
- 如果审稿人问“Qwen 是否参与了最终决策”，答案是：**没有参与最终管线；只作为辅助 gate/候选，且被信号规则否决。**

---

## 11. 逐步归因总表（按决策类型）

### 11.1 规则决定的部分（可重复、可审计）

| 决策 | 规则来源 | 说明 |
|---|---|---|
| R1/R2 反思触发 | CLAUDE.md：UNCERTAIN 必须反思 | 不是 LLM 自主决定；R0 是首轮盲识别，不是反思 |
| R3/R4 反思触发 | SKILL.md/CLAUDE.md：GT PSNR<30 + UNCERTAIN → 反思 | 使用 GT，属于 oracle 反馈 |
| PSNR 模式 vs 信号模式 | SKILL.md：sigma<2 → PSNR；sigma≥2 → 信号 | 本场景 sigma=1.11，主判据 PSNR；R1 额外用 verify_signals 确认 glass |
| PSNR 40/30 dB 分级 | SKILL.md：>40 LIKELY；30–40 UNCERTAIN；<30 POOR/UNCERTAIN | R0 34.83 → UNCERTAIN |
| severity 扫描 | SKILL.md：PARTIAL/WEAK/MISMATCH 时扫 ±1/±2 | gaussian 1/3/4、glass 1/3/4/5、lens 1/2/3 等 |
| blur 三族最低覆盖 | SKILL.md：至少测 3 种 blur | gaussian+lens+motion；glass 是额外 |
| 噪声假阳性规则 | SKILL.md：`extreme_pct>50% AND sigma<5 → 忽略 IMPULSE` | sigma=1.98 → NO_NOISE |
| JPEG block/unique_G 门控 | SKILL.md：block_boundary>1.1 / unique_G<200 才是 0% FP | 本场景未触发，但仍测试 |
| 架构 Direct/Swin/SG | finetune_strategy.md Section 六/七 | 无 motion → 无 OCAB；Tier B blur → Swin；SG 默认开 |
| 不使用 DualBranch | finetune_strategy.md §七：盲识别 0/24 胜 | 覆盖 full_analysis 的 generic 建议 |
| 不使用 GCM/ColorPre | variance_ratio≈1.0，无 contrast | Tier A 未触发 |
| EPOCH/LR/batch | CLAUDE.md + finetune_strategy.md | 2 / 5e-4 / 16 |
| R4 双 blur 违规 | CLAUDE.md：blur/noise/compression 各最多一次 | R4 应被硬约束直接拒绝 |

### 11.2 LLM 真正贡献的部分

| 决策 | 为什么不是规则 | 价值 | 结果 |
|---|---|---|---|
| R1 补测 `blur_glass:2` | “至少 3 种 blur”只要求 gaussian/lens/motion；glass 不在最低清单 | **高**：找到了 GT 子类型 | 36.65 / CI 10/10；但被 2% 规则压住，未采纳 |
| R2 探索 gamma | `cross_channel_corr=0.731` + `gamma_suspect`；规则没有强制测 gamma | 中：排除一个假阳性方向 | 22.7–22.9，拒绝 |
| R2 探索 JPEG-only / quantization | `dct_zero_ratio=0.624` / `unique_G=256`；规则要求测 JPEG，但没要求 quantization | 中：确认 dct_zero 是 blur 假阳性 | 28.12 / 25.68，拒绝 |
| R4 双 blur 假设 | 没有任何规则要求双 blur；是 LLM 的结构假设 | **负**：违反硬约束，证据弱 | 37.30 / CI 6/10；最终 27.03，低于 R3 27.20 |
| R4 JPEG/Noise/severity 系统排除 | 部分是规则要求，部分是 LLM 扩展 | 中：封闭假阳性空间 | 全部下降 |

### 11.3 规则压制 LLM 的部分（本案例的核心）

| 时点 | LLM 的建议 | 规则 | 结果 |
|---|---|---|---|
| R1 | glass:2 明显优于 gaussian:2 | `REFLECTION_MECHANISM.md §二`：blur 子类型 PSNR 改善只有 2% GT 命中，不得因 PSNR 切换 | 保留 gaussian:2；glass 仅 alternative |
| R2 | 可以继续探索，但未再提 glass | 连续无改善 → BEYOND_CAPABILITY | 保留 gaussian:2；R2 训练 19.33（737）触发 R3 |
| R3 | 把 glass:2 提升 | GT PSNR=19.33 → 允许修正 | 终于采用 glass:2；测试 27.20/23.52 |
| R4 | 双 blur | CLAUDE.md 管线约束 + PSNR/CI 门控 | 规则本应拒绝；实际 LLM 违规采用了它，结果更差 |

### 11.4 GT 反馈的部分

| 时点 | GT 信号 | 用途 | 影响 |
|---|---|---|---|
| R3 触发 | R2 737 GT PSNR=19.33 | 证明 gaussian:2 错误 | 允许把 R1 的 glass 提升 |
| R4 触发 | R3 737 GT PSNR=24.33 | 证明 glass 仍不够好 | 允许双 blur 假设；结果失败 |
| 最终评估 | GT params `glass:2` | 仅事后评分 | 27.20/23.52（step7546） |

**结论：** 如果把“盲识别”定义为不使用 GT 的预测，那么 blind_0003 的最终正确答案（glass:2）不是盲识别系统独立选出的；它是 R1 的 LLM 候选发现 + R3 的 GT 反馈采纳。**规则控制器反而在 R1 用 PSNR 排名独立选出了 glass:2（27.94/24.19）**，所以这个案例不能用来论证 LLM 的独立选择能力。

### 11.5 规则条款索引

| 条款 | 文件/行 | 在本案例的作用 |
|---|---|---|
| PSNR >40 确认；30–40 微调 severity；<30 UNCERTAIN/POOR | `SKILL.md` 72–75、181–183 | gaussian:2 34.83 → UNCERTAIN；glass:2 36.65 仍不足以单独确认子类型 |
| severity 扫描 ±1/±2 | `SKILL.md` 85–96、440–441 | gaussian 1/3/4、glass 1/3/4/5、lens 1/2/3 |
| 噪声下 PSNR 不可靠；sigma<2 才可用 PSNR | `SKILL.md` 98–126 | 本场景 sigma=1.11 → 全程 PSNR 模式 |
| `extreme_pct>50% AND sigma<5` → IMPULSE 假阳性 | `SKILL.md` 233 | peeled residual 的 extreme_pct=61.83 被忽略 |
| blur 至少测 3 种（gaussian+lens+motion） | `SKILL.md` 421–422 | R0 覆盖 3 种；glass 是 R1 额外补测 |
| GT PSNR<30 + UNCERTAIN → 反思 | `SKILL.md` 241–243；`CLAUDE.md` 413 | R3/R4 的触发 |
| blur 子类型 PSNR 改善 ≠ GT 正确（2%） | `REFLECTION_MECHANISM.md` 129 | R1/R2 不允许因 36.65>34.83 切换 glass |
| gaussian↔glass 互换代价 2–7 dB | `REFLECTION_MECHANISM.md` 103–111 | R1 把 glass 作为强替代而不是主预测 |
| Tier B blur → Swin 基线 | `finetune_strategy.md` 433、447 | 不用 OCAB/GCM/ColorPre |
| SimpleGate 默认常开（-32K，8/9 安全） | `finetune_strategy.md` 176、236、376、480、498 | R1–R4 全部 Swin+SG |
| DualBranch 盲识别 0/24 胜，不推荐 | `finetune_strategy.md` 284 | 覆盖 `full_analysis` 的 DualBranch 建议 |
| blur/noise/compression 各最多一次 | `CLAUDE.md` 574–576 | R4 双 blur 违规，应被硬约束拒绝 |

---

## 12. 对审稿人问题的建议回答

### Q1：这个案例是否说明 LLM 有超出规则的贡献？

**建议回答：**

> 有，但不是“独立选出最终答案”。blind_0003 的 R0 在 blur 子类型上只测了 gaussian/lens/motion，漏掉了 glass；R1 的 LLM 反思主动补测了 `blur_glass:2`，得到 36.65 dB / CI 10/10，而 GT 正是 `blur_glass:2`。这说明 LLM 能扩展候选空间、发现规则最低清单之外的子类型，并能系统排除 JPEG/noise/gamma 假阳性。  
> 但最终采纳发生在 R3，触发依据是 GT 训练 PSNR=19.33；在 support-only 条件下（R1/R2），系统两次选择保留 gaussian。规则控制器 E1 反而在 R1 用 PSNR 排名直接选了 glass，固定 LQ 27.94/24.19，优于 AutoIR 的 25.53/22.33。因此这个案例不能证明 LLM 独立选择或优于规则；它证明的是“LLM 候选发现 + 规则安全门控”的互补，以及当前 2% 规则在本场景过度保守。

### Q2：R0→R1 的 +2.0 dB 是不是 LLM 反思的贡献？

**建议回答：**

> 不能这样写。R0 和 R1 的训练管线都是 gaussian:2；R1 新增了 SimpleGate，且 R0/R1 的验证集与 checkpoint 选择不同；R2 与 R1 同配置重训在 step7546 口径下反而低 3.75 dB。R0→R1 的 +2.0 dB 是架构默认项、重训和 checkpoint 选择的混合结果，不能单独归因于反思。本案例的反思贡献应写成“发现 glass 候选”，而不是“+2 dB”。

### Q3：R3 的 +5.45 dB 是否说明修正有效？

**建议回答：**

> 方向有效，幅度不能单独引用。R3 的触发是 GT PSNR=19.33（737 张混合集），不是 support-only；737 门控从 19.33→24.33（+5.00）是决策时实际看到的。标准 DIV2K/LSDIR 在 step7546 口径是 21.75→27.20（+5.45），但换到数值最后一步 step15092 只有 25.33→26.25（+0.92）。轮次级 dB 对 checkpoint 敏感；应报告方向（glass 优于 gaussian）和 737 门控的 +5.00，而不是把 +5.45 当作稳定增益。

### Q4：R4 的双 blur 是不是 LLM 的创新？

**建议回答：**

> 是 LLM 提出的结构假设，但不是有效的创新。`glass:2 + gaussian:1` 违反项目“blur 最多出现一次”的硬约束，不可能成为合法 GT；仿真增益只有 +0.72 dB，CI 从 10/10 降到 6/10；最终测试 27.03/23.40，低于 R3 的 27.20/23.52（step7546）。这说明 LLM 能提出新结构，但需要硬约束检查、CI 门控和“不要从 GT 反推结构”的约束。

### Q5：那这个案例应该怎么用？

**建议回答：**

> 作为**局限性与失败模式**案例，而不是 LLM 成功案例。它说明：
> 1. LLM 的候选空间扩展是真实价值（glass 是 R0 没测的）；
> 2. 当前 blur 反思的 2% 规则在本场景过度保守，压制了正确候选；
> 3. 最终修正依赖 GT 反馈，不能算盲识别胜利；
> 4. 规则控制器在候选预置时反而更好；
> 5. R4 暴露了缺硬约束检查的问题。
>
> 改进方向：对高 CI（10/10）+ 高 PSNR 的跨族候选，允许并行训练 top-2 而不是直接压成 alternative；增加管线合法性检查；统一 checkpoint 选择协议；在报告中同时给出 support-only 与 GT-feedback 两组结果。

---

## 13. 复现与文件索引

### 13.1 目录

```
rebuttal_plan/case_blind_0003/
├── report.md                     ← 本报告
├── README.md                     ← 一页索引
├── MANIFEST.md                   ← 文件清单 + sha256 + 来源路径
├── data/
│   ├── challenge/                ← predicted_params R0/R1/R3/R4、reflection、full_analysis、training_strategy、iterations_table
│   ├── qwen/                     ← Qwen gate + Qwen pipeline
│   ├── rule_reference/           ← E1 规则控制器决策/params/selection/support/test metrics
│   └── gt_posthoc/               ← GT params（仅事后）
├── images/                       ← clean/degraded 5 对 + 单图（本地；.png 被 .gitignore 忽略）
├── logs/
│   ├── exp41_R0..R4_blind_0003.log
│   ├── E1_rule_R0/R1_blind_0003.log
│   ├── reeval_lpips_gpu*.log     ← 最终 VGG-LPIPS 重评估
│   └── numeric_last_step15092/   ← step15092 口径的评估日志
├── metrics/
│   ├── gt_reeval_R0..R4.json     ← step7546 口径的最终评估
│   ├── gate_737_R2_19.33.json    ← R3 触发证据
│   ├── gate_737_R3_24.33.json    ← R4 触发证据
│   ├── metrics_provenance.json   ← 所有数字的单一索引
│   ├── training_summary.json     ← 每轮 checkpoint 曲线/训练摘要
│   ├── E1_rule_vs_AutoIR_blind_0003.json
│   ├── A0_autoair_test_metrics_blind_0003.jsonl
│   └── raw_per_gpu/              ← 原始 per-GPU eval JSON（部分被后续 pass 覆盖，仅供追溯）
└── tools/
    └── collect_extracts.py       ← 生成本目录的抽取脚本
```

### 13.2 复现命令

```bash
# 1) 查看规则控制器决策
python3 -m json.tool rebuttal_plan/case_blind_0003/data/rule_reference/E1_decision_blind_0003.json

# 2) 查看 35 条 reflection iterations
python3 - <<'PY'
import json
d=json.load(open('rebuttal_plan/case_blind_0003/data/challenge/reflection.json'))
for it in d['iterations']:
    t=it.get('test_result',{})
    print(it['round'], it['hypothesis']['pipeline'], t.get('psnr_rgb'), t.get('ci_pass'), it['decision'])
PY

# 3) 重新计算 E1 规则 vs AutoIR
python3 - <<'PY'
import json
d=json.load(open('rebuttal_plan/case_blind_0003/metrics/E1_rule_vs_AutoIR_blind_0003.json'))
print(json.dumps(d, indent=2, ensure_ascii=False))
PY

# 4) 查看 737 门控
cat rebuttal_plan/case_blind_0003/metrics/gate_737_R2_19.33.json
cat rebuttal_plan/case_blind_0003/metrics/gate_737_R3_24.33.json
```

### 13.3 未复制的内容

- **checkpoint（.pt）没有复制**：文件大，且不是决策证据；训练日志里保留了 checkpoint 路径/step，`metrics/training_summary.json` 里有每轮的 checkpoint 曲线和训练摘要。
- **图像复制到了 `images/`，但 `.gitignore` 忽略 `*.png`**：本地可用于复核；如果要把图像推到远程，需要 `git add -f` 或调整 `.gitignore`。
- **完整评估原始 JSONL**：E1/A0 的 350 行/场景测试行已抽到 `metrics/`；exp41 的 per-image 行没有全部复制，只保留汇总和 per-GPU 结果。

---

## 14. 附录 A：35 条持久化 iteration 的逐条归因

> 说明：R3 没有独立的 iteration 条目（它只提升/重验 R1 的 glass）；R2 的 round 44–50 只在 `r2_summary` 中出现，不在下表。下表是 `reflection.json.iterations` 的完整 35 条。

| # | 轮次 | 假设管线 | 仿真 PSNR | CI | 决策 | 归因类别 | 依据/备注 |
|---:|:--:|---|---:|---:|---|---|---|
| 1 | R0 | `blur_gaussian:2` | 34.83 | 8/10 | ADJUST_SEVERITY | R+L | gaussian 主假设；gm_ratio=0.532；PSNR 模式；30-40dB → UNCERTAIN |
| 2 | R0 | `blur_gaussian:3` | 32.89 | 7/10 | ADJUST_SEVERITY | R | severity 扫描（SKILL 强制） |
| 3 | R0 | `blur_gaussian:1` | 32.91 | 8/10 | ADJUST_SEVERITY | R | severity 扫描（SKILL 强制） |
| 4 | R0 | `blur_gaussian:2,compression_jpeg:3` | 29.76 | 4/10 | TRY_ALTERNATIVE | L | dct_zero_ratio=0.624 触发 JPEG 假设；PSNR 29.76 拒绝 |
| 5 | R0 | `blur_gaussian:4` | 30.55 | 5/10 | ADJUST_SEVERITY | R | severity 扫描；更重 blur 不优 |
| 6 | R0 | `blur_motion:2` | 25.76 | 8/10 | TRY_ALTERNATIVE | R | SKILL 要求至少测试 3 种 blur；motion 仅 25.76 |
| 7 | R0 | `blur_gaussian:2,noise_gaussian_YCrCb:1` | 21.4 | 3/10 | TRY_ALTERNATIVE | L | noise 假设（sigma/spatial_corr）；PSNR 21.40 拒绝 |
| 8 | R0 | `blur_lens:2` | 34.61 | 8/10 | ADJUST_SEVERITY | R | blur 跨族测试（lens）；34.61 接近 gaussian |
| 9 | R0 | `compression_jpeg:3,blur_gaussian:2` | 31.2 | 7/10 | ADJUST_SEVERITY | L | 违反物理顺序的对照（JPEG→blur）；31.20 拒绝 |
| 10 | R0 | `blur_gaussian:2,compression_jpeg:1` | 32.04 | 5/10 | ADJUST_SEVERITY | L | JPEG severity 细化；32.04 拒绝 |
| 11 | R1 | `blur_glass:2` | 36.65 | 10/10 | ADJUST_SEVERITY | L | R1 自主补测未测过的 glass：36.65/10/10 = GT 类型 |
| 12 | R1 | `blur_glass:1` | 35.8 | 8/10 | ADJUST_SEVERITY | R | glass severity 扫描 |
| 13 | R1 | `blur_glass:3` | 33.05 | 8/10 | ADJUST_SEVERITY | R | glass severity 扫描 |
| 14 | R1 | `blur_glass:2` |  | MATCH | ACCEPT | R | verify_signals 强制验证 → MATCH(2✓) |
| 15 | R2 | `brightness_brighten_gamma_RGB:1,blur_gaussian:2` | 22.79 | 7/10 | TRY_ALTERNATIVE | L | gamma 假设（cross_channel_corr=0.731）；22.79 拒绝 |
| 16 | R2 | `blur_gaussian:2,brightness_brighten_gamma_RGB:1` | 22.71 | 7/10 | TRY_ALTERNATIVE | L | gamma 在 blur 后；22.71 拒绝 |
| 17 | R2 | `brightness_brighten_gamma_RGB:1,blur_glass:2` | 22.88 | 9/10 | TRY_ALTERNATIVE | L | gamma+glass；22.88 拒绝 |
| 18 | R2 | `blur_gaussian:2,compression_jpeg:2` | 31.16 | 4/10 | ADJUST_SEVERITY | L | JPEG sev=2 补测；31.16 拒绝 |
| 19 | R2 | `blur_gaussian:2,compression_jpeg:4` | 28.09 | 4/10 | TRY_ALTERNATIVE | L | JPEG sev=4 补测；28.09 拒绝 |
| 20 | R2 | `contrast_weaken_stretch:1,blur_gaussian:2` | 16.84 | 3/10 | REJECT | L | contrast 假设（variance_ratio=0.954）；16.84 拒绝 |
| 21 | R4 | `blur_glass:2` | 36.58 | 10/10 | ADJUST_SEVERITY | L | R4 重新验证 glass 基线；36.58/10/10 |
| 22 | R4 | `blur_glass:2,compression_jpeg:1` | 33.1 | 5/10 | ADJUST_SEVERITY | L | glass+JPEG:1；33.10 拒绝 |
| 23 | R4 | `blur_glass:2,compression_jpeg:2` | 32.02 | 4/10 | ADJUST_SEVERITY | L | glass+JPEG:2；32.02 拒绝 |
| 24 | R4 | `blur_glass:2,compression_jpeg:3` | 30.38 | 4/10 | ADJUST_SEVERITY | L | glass+JPEG:3；30.38 拒绝 |
| 25 | R4 | `blur_glass:2,compression_jpeg:4` | 28.52 | 4/10 | TRY_ALTERNATIVE | L | glass+JPEG:4；28.52 拒绝 |
| 26 | R4 | `blur_glass:2,compression_jpeg:5` | 26.04 | 4/10 | TRY_ALTERNATIVE | L | glass+JPEG:5；26.04 拒绝，JPEG 假阳性封闭 |
| 27 | R4 | `blur_glass:4` | 30.45 | 6/10 | ADJUST_SEVERITY | L | glass severity 4；30.45 拒绝 |
| 28 | R4 | `blur_glass:5` | 28.6 | 6/10 | TRY_ALTERNATIVE | L | glass severity 5；28.60 拒绝 |
| 29 | R4 | `blur_glass:2,noise_gaussian_YCrCb:1` | 21.47 | 3/10 | TRY_ALTERNATIVE | L | glass+YCrCb 噪声；21.47 拒绝 |
| 30 | R4 | `blur_glass:2,noise_gaussian_RGB:1` | 25.72 | 3/10 | TRY_ALTERNATIVE | L | glass+RGB 噪声；25.72 拒绝 |
| 31 | R4 | `blur_lens:3` | 34.6 | 9/10 | ADJUST_SEVERITY | L | lens:3；34.60 低于 glass |
| 32 | R4 | `blur_motion:1` | 26.7 | 6/10 | TRY_ALTERNATIVE | L | motion:1；26.70 拒绝 |
| 33 | R4 | `blur_glass:2,blur_gaussian:1` | 37.3 | 6/10 | ADJUST_SEVERITY | L! | 双 blur glass:2+gaussian:1；37.30 但 CI 6/10；违反每次最多一次 blur 的管线规则 |
| 34 | R4 | `blur_gaussian:1,blur_glass:2` | 36.75 | 7/10 | ADJUST_SEVERITY | L! | 反向双 blur；36.75 CI 7/10；同样违反管线规则 |
| 35 | R4 | `blur_lens:2` | 34.61 | 8/10 | ADJUST_SEVERITY | L | lens:2 复测；34.61 |

> **R3 不在上表：** `reflection.json.iterations` 没有 R3 的独立条目。`r3_summary` 记录了一次重验：`[round 3] blur_glass:2 → 36.59 dB / CI 10/10 / verify_signals=MATCH(2✓/0✗)`。R3 没有新的候选发现，只是把 R1 的 glass:2 从 alternative 提升为主预测；触发依据是 GT PSNR=19.33。
>
> **R2 的不完整账本：** `r2_summary` 还引用了 round 44–50（JPEG-only、lens:1/3、quantization_median 等），这些测试没有写入 `iterations`/`psnr_ranking`；因此真实测试次数至少 50，上表只是持久化的 35 条。


---

## 15. 附录 B：关键日志摘录

见 `logs/key_log_extract.txt`。包含每轮的：

- `EXP_META`（架构、管线、LR、batch、epoch）；
- 4 个 checkpoint 的验证 PSNR；
- `Loaded best checkpoint`；
- 最终训练摘要（best_step、training_seconds、num_params_M 等）。

关键行示例：

```
R0: simple_gate=false, pipeline=[["blur_gaussian",2]], best step 7546 (Set14 26.43)
R1: simple_gate=1,    pipeline=[["blur_gaussian",2]], best step 15092 (Set14 25.24)
R2: simple_gate=1,    pipeline=[["blur_gaussian",2]], best step 11319 (5-support 25.76)
R3: simple_gate=1,    pipeline=[["blur_glass",2]],    best step 11319 (5-support 28.24)
R4: simple_gate=1,    pipeline=[["blur_glass",2],["blur_gaussian",1]], best step 7546 (5-support 27.38)
```

---

## 16. 附录 C：规则控制器候选表（E1）

| 候选 | 仿真 PSNR |
|---|---:|
| `blur_glass:2` | 33.715 |
| `blur_gaussian:2` | 32.339 |
| `blur_lens:2` | 32.079 |
| `blur_gaussian:2 + compression_jpeg:1` | 30.405 |
| `blur_gaussian:3` | 30.298 |
| `blur_gaussian:1` | 29.976 |
| `blur_gaussian:4` | 28.006 |
| `blur_motion:2` | 24.563 |
| `blur_gaussian:2 + noise_gaussian_RGB:1` | 25.178（信号分 0.66） |

规则控制器 R0 选 `gaussian:2`；R1 选 `glass:2`（33.715 最高）；R1 支持集 best 28.40；`trigger_R2=false`。

---

## 17. 附录 D：Qwen 输出

`predicted_params_qwen.json`（节选；完整文件见 `data/qwen/`）：

```json
{
  "pipeline": ["noise_gaussian_RGB:1", "blur_gaussian:2", "compression_jpeg:2"],
  "analysis": {
    "verdict": "UNCERTAIN",
    "tier": 3,
    "reasoning": "5 对图像退化特征存在差异但符合常见管线顺序：图像对 4 显示轻微噪点（noise），图像对 1/2/5 显示明显高斯模糊（blur），图像对 2 显示显著 JPEG 块效应（compression）……综合判断为噪声→模糊→压缩的管线，严重度处于中低水平。",
    "model": "qwen3.7-plus-2026-05-26",
    "qwen_response_time_s": 175.3,
    "qwen_attempts": 1
  }
}
```

`qwen_noise_gate.json`：

```json
{"has_noise": true, "raw": "YES", "time_s": 28.1}
```

**未被采用。** 主 Agent 的 noise prior sigma=1.11 < 2 → NO_NOISE；Qwen 的 noise/JPEG 组合在 PSNR 上也全部下降。

---

## 18. 附录 E：一句话版本（可直接放进 rebuttal）

> `blind_0003` 的 GT 是单步 `blur_glass:2`。LLM 在 R1 主动补测了 R0 漏掉的 `blur_glass:2`（36.65 dB / CI 10/10），这是真实的候选空间扩展；但当前 blur 反思规则（PSNR 改善 ≠ 子类型正确，2% 命中率）不允许因 PSNR 切换，因此 R1/R2 仍保留 `gaussian:2`，直到 R3 用 GT 训练 PSNR=19.33 触发修正才切到 glass。最终测试 27.20/23.52（step7546 口径），但换到 step15092 口径只有 26.25/23.01；R2 的“崩溃”也依赖 checkpoint 选择。规则控制器 E1 在同一场景 R1 直接选 glass，固定 LQ 27.94/24.19，优于 AutoIR 的 25.53/22.33。因此该案例应作为“LLM 候选发现 + 规则过度保守 + GT 反馈修复”的局限性案例，而不是 LLM 独立选择或优于规则的证据。R4 的双 blur 违反“blur 最多一次”的硬约束，且没有带来稳定提升，进一步说明需要硬约束检查和 CI 门控。
