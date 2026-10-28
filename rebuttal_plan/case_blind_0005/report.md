# blind_0005 复盘：impulse-noise 漏识别与 CLR 未能纠错的原因

> 回答审稿人问题 2：**“How does CLR distinguish an incorrect degradation hypothesis from an inadequate restorer, and why does it fail to correct the reported impulse-noise case?”**
>
> 场景：`exp41/challenges/phase4/blind_0005`
> GT（事后标注）：`noise_impulse:5`，单步、极端 impulse noise。
> 本报告只做**事后复盘**；GT 不作为盲识别输入。需要区分：R0–R2 是 support-only；R3/R4 的触发使用了 GT 训练 PSNR（21.84），因此这两轮是 **GT 反馈**，不是完全盲识别。
>
> 一句话结论：**这个案例不是“impulse noise 太难、restorer 不够强”，而是 CLR 的搜索和判据失败。** AutoIR 选择了错误的 `brightness_brighten_shfit_HSV:1 + noise_impulse:3`，之后 R1/R2 没有改变 DC，R3/R4 只是分析、没有重新训练；它把 CI=10/10 当作“理论最优”，把低 GT PSNR 归因于模型容量。**同一套 0.4M Swin+SG、同一 2-epoch/1.5 GPU-h 预算，规则控制器只用 `noise_impulse:5` 就得到 44.92/41.14 dB**，直接证明 restorer 足够、错误在 DC 假设。

---

## 1. 数据、GT 与时间线

| 项目 | 内容 |
|---|---|
| GT | `noise_impulse:5`（`exp41/degradation_gt/blind_0005_params.json`） |
| support | 5 对 `clean_0..4.png` / `degraded_0..4.png` |
| test | DIV2K_valid_HR 100 + LSDIR val1/GTmod4 250，固定 LQ |
| R0 训练 | 2026-06-30 19:18，纯 Swin，`brightness:1 + impulse:3` |
| R1 训练 | 2026-07-01 16:08，Swin+SG，同一管线 |
| R2 训练 | 2026-07-14 09:19，Swin+SG，同一管线 |
| R3 | `exp41/experiments/exp41_R3_blind_0005 -> exp41_R2_blind_0005`，**没有新训练** |
| R4 | `exp41/experiments/exp41_R4_blind_0005 -> exp41_R3_blind_0005`，**没有新训练** |
| 反思记录 | `reflection.json` 最后更新 2026-07-15 06:17；persisted 28 条 iteration + R2/R3/R4 summary-only tests |

**重要：这不是“五轮训练都失败”。** 只有 R0/R1/R2 三次训练，而且三次的 DC 完全相同；R3/R4 只是重新分析并把 R2 的 checkpoint 软链接过去。所谓“五轮零改善”，本质是**同一条错误管线被训练了三次，然后停止更新**。

---

## 2. 逐轮结果

| Round | 预测管线 | 架构 | best step / 选择指标 | 训练时间 | DIV2K PSNR | LSDIR PSNR |
|---|---|---|---:|---:|---:|---:|
| R0 | `brightness_brighten_shfit_HSV:1 + noise_impulse:3` | Swin | step15092 / Set14 41.05 | 5214.3 s | 21.70 | 21.90 |
| R1 | 同上 | Swin+SG | step15092 / Set14 40.27 | 5082.9 s | 21.46 | 21.66 |
| R2 | 同上 | Swin+SG | step7546 / support 21.66 | 5070.8 s | 21.56 | 21.78 |
| R3 | symlink→R2 | — | — | — | 21.56 | 21.78 |
| R4 | symlink→R3 | — | — | — | 21.56 | 21.78 |

说明：

- R0/R1 的 `Set14 41.05/40.27` 是在**预测退化**上验证，不是 GT；它只能说明模型学会了“预测的 LQ”，不能说明预测正确。R2 改用 5 对 support，得到 21.66；这个数字才暴露了问题。
- 表里的 DIV2K/LSDIR 是 `exp41/results/gt_reeval_R*.json`（step7546 口径）。A0 选中的 AutoIR R1 step15092 在固定 LQ 上是 **21.9155/22.1700**；与规则控制器比较时用这个口径（见 §7）。
- R3/R4 没有训练；`gt_reeval_R3/R4` 只是复制 R2 的分数。
- R1/R2 相对 R0 只改了架构（加 SimpleGate）和验证集/checkpoint 选择，**DC 没有变**；所以 R0→R1/R2 的 PSNR 变化不是“修正了退化假设”的结果。

**日志证据：**

```text
R0: ValDataset: 14 images | Loaded best checkpoint: step 15092 (Set14 PSNR_RGB=41.05)
    training_seconds: 5214.3 | num_params_M: 0.5
R1: ValDataset: 14 images | Loaded best checkpoint: step 15092 (Set14 PSNR_RGB=40.27)
    training_seconds: 5082.9 | num_params_M: 0.4
R2: PairedValDataset: 5 image pairs | Loaded best checkpoint: step 7546 (Set14 PSNR_RGB=21.66)
    training_seconds: 5070.8 | num_params_M: 0.4
R3/R4: exp41/experiments/exp41_R3_blind_0005 -> exp41_R2_blind_0005
       exp41/experiments/exp41_R4_blind_0005 -> exp41_R3_blind_0005
```

---

## 3. 初始盲识别：两个 DC 错误

### 3.1 RFA 原始信号

`full_analysis.json`（R0–R4 完全相同，md5 `0a0e4a1a...`）：

| 类别 | verdict | 关键信号 |
|---|---|---|
| compression | `NO_COMPRESSION` | `block_boundary=1.0074`（<1.1）、`unique_G=244`（>200）、`dct_zero_ratio=0.0124` |
| noise | `IMPULSE`（medium） | `noise_prior_sigma=0.0`、`impulse_zero_ratio=0.9138`、`estimated_severity=3` |
| blur | `BLUR_UNLIKELY` | `gm_ratio=2.3562`（远大于 0.85）、`spectral_slope_ratio=0.99` |
| global | `BRIGHTNESS_brighten_sev1; SATURATION_strengthen_sev5` | `brightness_mean_shift=8.0`、`clipped_fraction=0.35`、`heavy_clipping=true`、`variance_ratio=1.4787`、`contrast_spread_ratio=1.273`、`saturation_chroma_ratio=2.5503`、`brightness_pure=false`、`contrast_from_saturation=true` |
| peeled noise | `IMPULSE`（high） | `extreme_pct=95.99`、`residual_std=52.4`、`noise_prior_sigma=18.29`、`impulse_zero_ratio=0.9138` |

RFA 自己给出的 global verdict 是 `BRIGHTNESS_brighten_sev1; SATURATION_strengthen_sev5`，但 `false_positive_checks` 同时说：

```text
brightness_pure = false
contrast_from_saturation = true
```

也就是说，**brightness 不是“纯 brightness”**，方差/色度变化可以归因于 saturation，而 saturation 又可能是 impulse 的副作用。LLM 正确地拒绝了 saturation，但保留了 brightness:1。

### 3.2 错误 1：brightness:1 是 impulse clipping 的补偿性假阳性

- `brightness_mean_shift=8.0`，低于规则控制器的 brightness 阈值 `mean_shift > 15`。
- `clipped_fraction=0.35`、`heavy_clipping=true`：35% 像素被 clip。GT 是 `noise_impulse:5`，salt-and-pepper 噪声本身就会把大量像素推到 0/255，造成“变亮 + clipping + 方差增大 + 色度异常”的全局统计。
- `variance_ratio=1.4787`、`saturation_chroma_ratio=2.5503` 都可以由 impulse + clipping 解释；`contrast_from_saturation=true` 已提示 contrast/variance 是 saturation 的副作用。
- 但 LLM 把 `clipped_fraction=0.35` 解释成“brightness shift + saturation 组合”，在拒绝 saturation 后仍保留了 brightness:1。`predicted_params.json` 原文：

```text
Noise(impulse sev=3)+Brightness(shift sev=1). Both orders give 10/10 CI.
Brightness severity 1 optimal (sev=2 drops CI to 9/10).
```

- 这是一个**补偿性假阳性**：impulse:5 造成高光/暗部 clip，brightness:1 让模拟图像的整体亮度/截断比例更接近目标，于是 CI 更高。CI 把“用 brightness 补偿 severity 不足”误判成“brightness 真实存在”。

**规则控制器为什么没有这个 FP：** `rule_controller.py:dc_from_one` 有一条硬规则：

```python
if not gverdict.startswith("NO_GLOBAL") and not impulse:
    ...  # 只有非 impulse 才解析 global
```

`rules_frozen.json` 里对应 `skip_global_when_impulse=true`。因为 RFA 的 noise verdict 是 `IMPULSE`，规则控制器直接跳过 global，只保留 `noise_impulse:3`。这是它避开 brightness FP 的直接原因。

### 3.3 错误 2：impulse severity 3 而非 5

- RFA 的 `estimated_severity=3` 被 LLM 直接接受。
- 但 peeled residual 给出了更强的信号：`residual_std=52.4`、`noise_prior_sigma=18.29`、`extreme_pct=95.99`、`impulse_zero_ratio=0.9138`。这些统计与“severity 3”并不一致；agent 没有做 severity 校准，只把 `extreme_pct` 用来确认“类型是 impulse”。
- 注意：peeled check 是在剥离 `brightness:1` 之后做的，`noise_prior_sigma=18.29` 可能被剥离误差放大，因此不能单独证明 severity=5；但它至少应该触发一次 severity 校准，而不是直接接受 RFA 的 3。规则控制器的 `noise_impulse:5` signal MATCH 和 44.92 dB 结果是更强的证据。
- 更关键的是：**`noise_impulse:5` 单独从未被测试过**。R0 只测了 impulse:1/2/3 单独；R2 测的是 `impulse:5 + brightness:1`，CI=9/10，被拒绝；R3/R4 也没有测 `impulse:5` 单独。
- 由于 brightness:1 在补偿 severity 不足，`impulse:3 + brightness:1` 的 CI=10/10，而 `impulse:5 + brightness:1` 的 CI=9/10。CI 在这里不是参数的似然，而是“视觉上像不像”；错误的 brightness 把 severity 3 拉到了和 GT 相似的截断比例，反而让正确的 severity 5 看起来“过强”。

**结论：** 两个错误是耦合的：brightness:1 补偿了 impulse:3 的不足；impulse:3 又让 brightness:1 看起来合理。CI 无法拆开这个耦合。

---

## 4. 反思过程：为什么没有纠正

### 4.1 R0：13 个候选，锁定 brightness+impulse:3

R0 的候选测试（persisted iterations 1–13）：

- brightness+saturation:5 → 12.21 / CI 7/10
- brightness:2 → 12.79 / 6/10
- gamma brightness:2 → 13.07 / 5/10
- saturation:2 → 12.33 / 7/10
- contrast stretch:3 → 10.73 / 3/10
- impulse:1/2/3 单独 → 13.54/12.86/12.28，CI 8/8/9
- impulse:3 + brightness:1（两个顺序）→ 12.14/12.13，**CI 10/10**
- brightness:2 + impulse:3 → 11.52 / 9/10

最终选择 `brightness:1 + impulse:3`，理由是 CI=10/10 最高；`brightness:2` 会让 CI 降到 9/10，所以 brightness:1 “最优”。这里没有对 `impulse:5` 单独做任何测试。

### 4.2 R1：只测 gamma/saturation 替代，保持 R0

`r1_summary`：

- `brightness_brighten_gamma_HSV:2 + impulse:3` → verify_signals WEAK（sigma diff 33%、mean_shift diff 0.0623、variance_ratio diff 8%）
- `saturate_strengthen_HSV:2 + impulse:3` → WEAK（sigma diff 67%、variance_ratio diff 86%）
- 重新验证 R0 baseline → WEAK（sigma diff 40%、mean_shift diff 0.0292、variance_ratio diff 15%）
- 决策：`KEEP_R0`，理由：“signal 模式 WEAK 是 impulse 噪声污染 mean_shift/variance_ratio 的预期行为，CI=10/10 是最强可用证据。”

**问题：** WEAK 被解释成“噪声导致的预期伪影”，而不是“这个信号对当前假设没有区分力”。对于 noise，SKILL 的正确做法是使用 §B/噪声专用统计（impulse_zero_ratio、extreme_pct、residual_std、sigma 校准），而不是把全局 mean_shift/variance 的 WEAK 当作可忽略。agent 没有用这些统计去校准 severity。

### 4.3 R2：测了 4 个新候选，仍保持 R1

`reflection_R2`：

- gaussian_RGB → CI 8/10
- speckle → 8/10
- poisson → 5/10
- `brightness:1 + impulse:5` → **CI 9/10**，被拒绝（因为 <10）
- 决策：`KEEP_R1`，认为 impulse:3 是最优 severity。

**问题：** 这是唯一一次接近正确 severity 的测试，但它仍然带着 brightness:1；`impulse:5` 单独从未出现。CI 9/10 被解读为“5 不如 3”，而不是“brightness 与 severity 的耦合”。

### 4.4 R3/R4：不训练，只分析；把低 PSNR 归因于模型容量

`reflection_R3` 原文：

```text
trigger: GT PSNR=21.84 < 25 → must NOT skip.
但 3 轮反思 (R0→R1→R2) 已 22 候选无改善，满足协议终止条件。
...
低 GT PSNR (21.84dB) 是模型对 impulse 噪声恢复能力不足, 非盲识别错误。
```

`reflection_R4` 原文：

```text
GT PSNR=21.84dB 归因于模型对 impulse 噪声的恢复能力不足
(1.5 GPU-h 预算下 Swin 64dim 无法有效处理 35% clipped 像素的 impulse 噪声).
盲识别已是最优。
```

R3 测了 6 个新方向（JPEG、contrast、saturation 顺序、brightness:3、impulse:2 等），R4 测了 7 个新方向（sat:5、3-step、JPEG-only、blur、quantization 等），全部 CI<10；但没有一个测试是 `noise_impulse:5` 单独。R3/R4 也没有触发新训练，`exp41_R3/R4` 是 R2 的软链接。

**这就是 CLR 失败的核心：**

1. **搜索锚定**：所有候选都围绕 `brightness:1 + impulse:3` 增删；没有做“去掉 brightness + 把 impulse 调到 5”的联合编辑。
2. **判据饱和**：CI=10/10 被当作理论最大值；一个错误假设如果已经 CI 饱和，就永远无法被 CI 推翻。
3. **终止条件与错误指标绑定**：连续无改善 → BEYOND_CAPABILITY，而“无改善”是 CI 无改善；所以错误假设越能拟合 CI，越容易终止。
4. **没有 restorer-adequacy 控制**：低 GT PSNR 被直接归因于模型容量，没有训练正确 DC 的对照。
5. **R3/R4 不训练**：即使后来想到新 DC，也没有再训练；闭环实际上在 R2 就停止更新模型。

---

## 5. 规则控制器：同一 restorer、正确 DC，直接反驳“模型容量”解释

E1 规则控制器（`rebuttal_runs/E1/decisions/blind_0005.json`）：

| 项目 | 规则控制器 | AutoIR |
|---|---|---|
| R0 管线 | `noise_impulse:3`（无 brightness；impulse 时跳过 global） | `brightness:1 + impulse:3` |
| R0 support PSNR | 35.13 | —（R0 用 Set14） |
| R1 管线 | **`noise_impulse:5`** | 仍为 `brightness:1 + impulse:3` |
| R1 support PSNR | **43.17** | 21.66（R2 support） |
| 最终 test DIV2K | **44.9185** | 21.9155 |
| 最终 test LSDIR | **41.1387** | 22.1700 |
| 架构/预算 | Swin+SG, 0.4M, 2 epoch, 1.5 GPU-h | Swin+SG, 0.4M, 2 epoch, 1.5 GPU-h |

口径说明：AutoIR 的 21.9155/22.1700 是 A0 选中的 R1 step15092；规则控制器的 44.9185/41.1387 是 R1 step7546。两者都是固定 LQ；23 dB 的差距远大于 checkpoint 选择差异（同一模型不同 checkpoint 通常 <1 dB）。

规则控制器的 R0 候选打分：

```text
noise_impulse:5   signal score 1.00  ← MATCH
noise_impulse:4   0.464
noise_impulse:3   0.33
gaussian_RGB:3    0.33
speckle:3         0.33
poisson:3         0.33
...
```

R0 的 `noise_impulse:3` 得分 0.33（WEAK），`noise_impulse:5` 得分 1.00（MATCH）；规则控制器因此在 R1 直接选 `noise_impulse:5` 并训练，support 从 35.13 升到 43.17，test 44.92/41.14。

**这组数据的含义：**

- 同一个 0.4M Swin+SG、同样的 2 epoch/1.5 GPU-h，正确 DC 可以达到 44.92 dB；所以 21.9 dB **不是模型容量上限**，而是 DC 错误的代价。
- 规则控制器没有 brightness FP（`skip_global_when_impulse`）；它在 R0 也把 severity 低估为 3，但固定候选表里有 `noise_impulse:5`，R1 通过 signal MATCH 修正到 5。LLM 的候选表/搜索没有产生这个单步候选。
- 粗略分解（support PSNR，非严格受控消融）：
  - 去掉 brightness（保持 impulse:3）：21.66 → 35.13，约 **+13.5 dB**
  - impulse severity 3→5：35.13 → 43.17，约 **+8.0 dB**
  - 合计约 **+21.5 dB**；test 上 21.92 → 44.92 是 **+23.0 dB**。
- 需要 caveat：规则控制器是固定候选表（里面本来就有 `noise_impulse:5`），不是盲 CLR；训练 run/seed/support 选择也不同。但 21.5–23 dB 的差距远大于 2-epoch 的 run-to-run 波动，方向是明确的。

---

## 6. CLR 到底能不能区分“错误 DC”与“不够强的 restorer”？

### 6.1 当前 CLR 的实际逻辑

这里 CLR 指 exp41 的 closed-loop reflection：每轮训练后，根据 GT/support PSNR 和信号证据决定是否修正 DC/RS。它可用的信号是：

1. **GT/support PSNR**：低于阈值 → 触发反思；
2. **CI / verify_signals**：判断候选管线是否更像目标；
3. **support PSNR**：决定是否继续下一轮。

这是一个**间接**区分：如果改 DC 后 PSNR 上升，说明原 DC 错；如果改 DC 后仍不升，说明 restorer 不足。问题是：

- 低 PSNR 本身是**二义的**：可能是 DC 错，也可能是 restorer 弱；
- 反射候选如果没有包含正确 DC，就永远看不到“改 DC 后 PSNR 上升”；
- CI 可以在错误 DC 上饱和，给出“DC 已正确”的假证据；
- 没有独立的 restorer-adequacy 控制，最终只能猜。

**blind_0005 正是这个二义性没有被打破：**

- GT PSNR 21.84 低；
- 错误 DC 的 CI=10/10；
- 正确 DC 从未被测试/训练；
- agent 选择“restorer 不足”，而不是“DC 错误”。

### 6.2 正确的区分方法

要在论文/系统中真正回答这个问题，需要至少一个**参数级**或**反事实**证据：

| 方法 | 做法 | 在本案例的预期结果 |
|---|---|---|
| 参数级可辨识性 | 对 noise 用 §B/impulse 统计校准 severity（extreme_pct、impulse_zero_ratio、residual_std、sigma），要求 MATCH | `impulse:5` → MATCH；`impulse:3` → WEAK；规则控制器正是这么做的 |
| 反事实 DC 交换 | 用同一 restorer/预算训练 top-2 DC 候选，比较 GT/support PSNR | 正确 DC 44.92 vs 错误 DC 21.92 → 原 DC 错 |
| Oracle-DC 控制 | 用 GT DC 训练同一 restorer（仅评估侧诊断，不反馈盲决策） | 44.92 → restorer 足够；若 oracle DC 也低，才是 restorer 不足 |
| E5c 4 格 | 不修正 / 只改 DC / 只改架构 / 两者均改，同初始化同预算 | 只改 DC 应大幅上升；只改架构不会 |
| 禁止 CI 单独作终止 | CI 只作辅助，不作为“理论最优”或 BEYOND_CAPABILITY 的唯一依据 | 错误 DC 的 CI=10/10 不再能锁定终止 |

E5c 的任务卡已经写好了（`rebuttal_plan/01_实验执行计划.md` E5c）：对 impulse 分别检验 severity 修正、移除无证据 brightness、二者共同修正；这些是**评估侧 oracle 干预**，不能反馈给盲决策智能体。**blind_0005 targeted 4-cell 已执行**（不是完整 16-cell）：

| Cell | DC | 架构 | DIV2K fixed-LQ 0005 |
|---|---|---|---:|
| no-fix | `brightness:1 + impulse:3` | Swin+SG | 21.915 |
| DC-only | `impulse:5` | Swin+SG | **44.908** |
| RS-only | `brightness:1 + impulse:3` | DualBranch+SG | 21.976 |
| both | `impulse:5` | DualBranch+SG | 41.620 |

主效应：**DC +22.99 dB**；**RS-only +0.06 dB**；**RS at correct DC −3.29 dB**。这直接支持“DC 是主因；只改架构不能救；架构改错还可能伤害正确 DC”。详见 `rebuttal_plan/E5c_blind_0005_4cell.md`。完整 E2-RS/E5b 仍未执行。

---

## 7. 对审稿人问题 2 的直接回答（可直接改写进 rebuttal）

### 中文版

> 当前 CLR 不能可靠地区分“DC 错误”与“restorer 不足”。它用 GT/support PSNR 低于阈值触发反思，但反思的接受/终止主要依赖 CI 和 support PSNR；当错误 DC 已经让 CI 饱和时，低 GT PSNR 会被误归因于 restorer 容量。blind_0005 就是这种情况：系统预测 `brightness_brighten_shfit_HSV:1 + noise_impulse:3`，而 GT 是 `noise_impulse:5`。brightness:1 是 impulse:5 造成 35% clipping 的补偿性假阳性；impulse severity 被低估为 3。R0–R2 三次训练都使用同一条错误管线，R3/R4 只做分析、没有重新训练；agent 把 `brightness:1+impulse:3` 的 CI=10/10 当作理论最优，拒绝 `impulse:5+brightness` 的 CI=9/10，却从未测试 `impulse:5` 单独，最终把 21.84 dB 归因于模型容量。  
> 同一套 0.4M Swin+SG、同一 2-epoch/1.5 GPU-h 预算下，规则控制器只改 DC 为 `noise_impulse:5`，固定 LQ 得到 **44.92/41.14 dB**（AutoIR 21.92/22.17）。这直接证明 restorer 足够，失败在 DC 假设和 CLR 的搜索/判据，而不是 impulse noise 本身。  
> 因此 CLR 需要：对 noise 做参数级 severity 校准（§B/impulse 统计），而不是用 CI；对 top-2 DC 候选做反事实训练（只改 DC）；用 oracle-DC 训练建立 restorer 上限；把“去掉假阳性步骤 + 修正 severity”的联合编辑加入搜索；不要把 CI=10/10 当作正确性或终止的充分证据。E5c targeted 4-cell 已执行：DC-only +22.99 dB，RS-only +0.06 dB，RS at correct DC −3.29 dB；完整 E2-RS/E5b 仍未执行。

### English version

> CLR does not currently distinguish an incorrect degradation hypothesis from an inadequate restorer in a robust way. It triggers reflection when GT/support PSNR is below a threshold, but acceptance and termination are driven mainly by CI and support PSNR. When a wrong DC already saturates CI, low GT PSNR is misattributed to restorer capacity. In blind_0005, AutoIR predicted `brightness_brighten_shfit_HSV:1 + noise_impulse:3` while the GT is `noise_impulse:5`. The brightness step is a compensating false positive caused by the 35% clipping of impulse:5, and the impulse severity is underestimated as 3. R0–R2 trained the same wrong pipeline three times; R3/R4 only re-analyzed and did not retrain. The agent treated CI=10/10 for `brightness:1+impulse:3` as the theoretical optimum, rejected `impulse:5+brightness` at CI=9/10, never tested `impulse:5` alone, and attributed the 21.84 dB GT PSNR to model capacity.  
> With the same 0.4M Swin+SG and the same 2-epoch/1.5 GPU-h budget, the rule controller changed only the DC to `noise_impulse:5` and reached **44.92/41.14 dB** on the fixed LQ test (AutoIR: 21.92/22.17). This shows that the restorer is adequate and the failure is in the DC hypothesis and CLR's search/metric, not in impulse noise per se.  
> CLR therefore needs parameter-level severity calibration for noise (§B/impulse statistics instead of CI), counterfactual training of the top-2 DC candidates (DC-only intervention), an oracle-DC control to establish the restorer ceiling, a joint “remove false-positive step + correct severity” search operator, and must stop treating CI=10/10 as sufficient evidence of correctness or termination. The targeted E5c 4-cell has now been run for blind_0005: DC-only +22.99 dB, RS-only +0.06 dB, and RS at the correct DC −3.29 dB. The full E2-RS/E5b experiment has not been run; the rule-controller comparison remains an evaluation-side proxy, not a blind CLR result.

---

## 8. 为什么“模型容量”解释不成立：关键证据链

| 证据 | 数值 | 说明 |
|---|---:|---|
| AutoIR 错误 DC（brightness:1+impulse:3） | 21.92 / 22.17 | A0 选中 R1 step15092，固定 LQ |
| AutoIR 训练时 Set14 | 41.05 / 40.27 | 在**预测退化**上验证，不能证明 DC 正确 |
| AutoIR support（R2） | 21.66 | 5 对 support，接近 GT PSNR |
| AutoIR GT 门控 | 21.84 | 737 张混合集；R3/R4 的触发值 |
| 规则控制器 R0（impulse:3 only） | support 35.13 | 去掉 brightness，severity 仍错 |
| 规则控制器 R1（impulse:5 only） | support 43.17 | 正确 DC |
| 规则控制器 test | 44.92 / 41.14 | 同架构、同预算、固定 LQ |
| AutoIR test | 21.92 / 22.17 | 同架构、同预算、固定 LQ |

**解释：**

- 如果 restorer 容量是瓶颈，那么正确 DC 的 44.92 不可能出现。
- 如果 Set14 41 dB 能证明 DC 正确，那么 GT PSNR 不会只有 21.7；Set14 是在预测退化上验证，循环论证。
- 如果 severity 3 是对的，规则控制器 R0（impulse:3 only）不会只有 35.13，而 R1（impulse:5）会到 43.17。
- 如果 brightness:1 是对的，去掉它不会让 support 从 21.66 跳到 35.13。

**因此：** 21.9 的低分是“错误 DC”造成的，不是“impulse noise + 1.5 GPU-h”造成的。agent 的 R4 结论“Swin 64dim 无法有效处理 35% clipped 像素”与规则控制器的 44.92 直接矛盾。

---

## 9. 审计与限制

1. **R3/R4 不是新训练**：`exp41_R3_blind_0005 -> exp41_R2_blind_0005`，`exp41_R4_blind_0005 -> exp41_R3_blind_0005`。所以“五轮反思”只有三次训练，且 DC 完全相同。
2. **R0/R1 的 Set14 41 dB 是预测退化验证**：不能用它证明 DC 正确；R2 的 support 21.66 才暴露问题。
3. **checkpoint 口径**：`gt_reeval_R*` 用 step7546（字典序最后，不是数值最后）；A0 AutoIR 用 R1 step15092；规则控制器用 R1 step7546。报告里已分别标注；不要把这些数字混着做逐轮差值。
4. **reflection ledger 不完整**：persisted 28 条 iteration；R2 summary 另有 4 个候选、R3 summary 6 个、R4 summary 7 个，其中部分不在 iteration 数组里；agent 自述总计 35 个候选。真实测试数至少 35，ledger 只保留 28 条。
5. **规则控制器是 proxy，不是盲 CLR**：它的固定候选表里本来就有 `noise_impulse:5`；它没有使用 GT；但它不是 E5c 的受控 4 格。它的 44.92 是评估侧证据，不能直接说成“CLR 自动纠正了”。
6. **E5c 未执行**：没有“只改 DC / 只改架构 / 两者”的受控训练；本报告用规则控制器作为事后对照，建议在 rebuttal 前补 E5c。
7. **Qwen 预测错误且未采用**：Qwen 预测 `noise_gaussian_RGB:3`，与实际 impulse 不符；`qwen_noise_gate` 只说“有噪声”，没有帮助 severity。Qwen 模型字符串是 `qwen3.7-plus-2026-05-26`；它不是主 Skill Agent。
8. **GT PSNR 21.84 是 737 张混合集**；R2 support 21.66 是 5 对 support；DIV2K 100 张的 21.56 又是另一个口径。三者接近但不可互相替代。

---

## 10. 文件索引

```text
rebuttal_plan/case_blind_0005/
├── report.md                         ← 本报告
├── README.md                         ← 一页索引
├── MANIFEST.md                       ← 文件清单 + sha256 + 来源
├── data/
│   ├── challenge/                    ← predicted_params R0/R1、reflection、full_analysis、training_strategy、Qwen
│   ├── rule_reference/               ← E1 规则控制器决策/params/selection/support metrics/rules_frozen
│   └── gt_posthoc/                   ← GT params（仅事后）
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
└── tools/
    ├── collect_extracts.py
    └── make_manifest.py
```

原始图像未复制（约 50MB）；路径为 `exp41/challenges/phase4/blind_0005/{clean,degraded}_*.png`。checkpoint 未复制；训练日志里保留了路径/step。

---

## 11. 复现/检查命令

```bash
# 1) GT
cat rebuttal_plan/case_blind_0005/data/gt_posthoc/blind_0005_params.json

# 2) 原始信号
python3 -m json.tool rebuttal_plan/case_blind_0005/data/challenge/full_analysis_R0.json

# 3) R0/R1 预测
python3 -m json.tool rebuttal_plan/case_blind_0005/data/challenge/predicted_params_R0.json
python3 -m json.tool rebuttal_plan/case_blind_0005/data/challenge/predicted_params_R1.json

# 4) 规则控制器决策
python3 -m json.tool rebuttal_plan/case_blind_0005/data/rule_reference/E1_decision_blind_0005.json

# 5) 训练日志关键行
less rebuttal_plan/case_blind_0005/logs/key_log_extract.txt

# 6) 指标来源
python3 -m json.tool rebuttal_plan/case_blind_0005/metrics/metrics_provenance.json
```

---

## 12. 附录 A：28 条持久化 iteration

| # | 轮次 | 假设管线 | 测试结果 | 决策 | 归因 |
|---:|:--:|---|---|---|---|
| 1 | R0 | `brightness_brighten_shfit_HSV:1,saturate_strengthen_HSV:5` | PSNR 12.21 / CI 7/10 | REJECT | L：初始 global 假设；PSNR 12.21 / CI 7/10 |
| 2 | R0 | `brightness_brighten_shfit_HSV:2` | PSNR 12.79 / CI 6/10 | REJECT | L：brightness alone；12.79 / 6/10 |
| 3 | R0 | `brightness_brighten_gamma_HSV:2` | PSNR 13.07 / CI 5/10 | REJECT | L：gamma brightness；13.07 / 5/10 |
| 4 | R0 | `saturate_strengthen_HSV:2` | PSNR 12.33 / CI 7/10 | REJECT | L：saturation alone；12.33 / 7/10 |
| 5 | R0 | `contrast_strengthen_stretch:3` | PSNR 10.73 / CI 3/10 | REJECT | L：contrast stretch；10.73 / 3/10 |
| 6 | R0 | `noise_impulse:2` | PSNR 12.86 / CI 8/10 | REJECT | R+L：impulse alone；12.86 / 8/10（规则要求扫 noise） |
| 7 | R0 | `saturate_strengthen_YCrCb:3` | PSNR 8.41 / CI 7/10 | REJECT | L：saturation YCrCb；8.41 / 7/10 |
| 8 | R0 | `brightness_brighten_shfit_RGB:2,saturate_strengthen_HSV:3` | PSNR 12.17 / CI 7/10 | REJECT | L：RGB brightness + sat；12.17 / 7/10 |
| 9 | R0 | `noise_impulse:1` | PSNR 13.54 / CI 8/10 | REJECT | R：impulse severity 1（severity 扫描） |
| 10 | R0 | `noise_impulse:3` | PSNR 12.28 / CI 9/10 | REJECT | R：impulse severity 3（severity 扫描） |
| 11 | R0 | `noise_impulse:3,brightness_brighten_shfit_HSV:1` | PSNR 12.14 / CI 10/10 | REJECT | L：impulse:3 + brightness:1 反向组合；12.14 / 10/10 |
| 12 | R0 | `brightness_brighten_shfit_HSV:1,noise_impulse:3` | PSNR 12.13 / CI 10/10 | REJECT | L：brightness:1 + impulse:3；12.13 / 10/10 |
| 13 | R0 | `noise_impulse:3,brightness_brighten_shfit_HSV:2` | PSNR 11.52 / CI 9/10 | REJECT | L：brightness severity 2；11.52 / 9/10 |
| 14 | R1 | `brightness_brighten_gamma_HSV:2,noise_impulse:3` | signal WEAK | TRY_ALTERNATIVE | L：gamma+impulse；WEAK |
| 15 | R1 | `saturate_strengthen_HSV:2,noise_impulse:3` | signal WEAK | TRY_ALTERNATIVE | L：saturation+impulse；WEAK |
| 16 | R1 | `brightness_brighten_shfit_HSV:1,noise_impulse:3` | signal WEAK | TRY_ALTERNATIVE | L：baseline signal re-verify；WEAK |
| 17 | R2 | `brightness_brighten_shfit_HSV:1,saturate_strengthen_HSV:3,noise_impulse:3` | signal WEAK | TRY_ALTERNATIVE | L：brightness+sat3+impulse；WEAK |
| 18 | R2 | `brightness_brighten_shfit_HSV:1,noise_impulse:4` | signal WEAK | TRY_ALTERNATIVE | L：impulse:4+brightness；WEAK |
| 19 | R2 | `noise_impulse:4,brightness_brighten_shfit_HSV:1` | signal WEAK | TRY_ALTERNATIVE | L：impulse:4+brightness 反向；WEAK |
| 20 | R2 | `noise_impulse:3,brightness_brighten_shfit_HSV:1` | signal WEAK | TRY_ALTERNATIVE | L：impulse:3+brightness 反向；WEAK |
| 21 | R2 | `brightness_brighten_shfit_HSV:1,saturate_strengthen_HSV:2,noise_impulse:3` | signal WEAK | TRY_ALTERNATIVE | L：brightness+sat2+impulse；WEAK |
| 22 | R2 | `brightness_brighten_shfit_HSV:1,noise_impulse:3` | signal WEAK | TRY_ALTERNATIVE | L：baseline 复核；WEAK |
| 23 | R4 | `saturate_strengthen_HSV:5,noise_impulse:3` | PSNR 11.11 / CI 7/10 | REJECT | L：sat5+impulse；11.11 / 7/10 |
| 24 | R4 | `brightness_brighten_shfit_HSV:1,saturate_strengthen_HSV:5` | PSNR 12.21 / CI 7/10 | REJECT | L：brightness+sat5；12.21 / 7/10 |
| 25 | R4 | `brightness_brighten_shfit_HSV:1,saturate_strengthen_HSV:5,noise_impulse:3` | PSNR 11.09 / CI 8/10 | REJECT | L：brightness+sat5+impulse；11.09 / 8/10 |
| 26 | R4 | `compression_jpeg:1` | PSNR 13.55 / CI 5/10 | REJECT | L：JPEG-only；13.55 / 5/10 |
| 27 | R4 | `blur_gaussian:1,noise_impulse:3,brightness_brighten_shfit_HSV:1` | PSNR 11.71 / CI 6/10 | REJECT | L：blur+impulse+brightness；11.71 / 6/10 |
| 28 | R4 | `quantization_hist:1,brightness_brighten_shfit_HSV:1,noise_impulse:3` | PSNR 8.42 / CI 3/10 | REJECT | L：quantization+brightness+impulse；8.42 / 3/10 |

> 说明：上表是 `reflection.json.iterations` 的 28 条持久化记录；R2 summary 另有 4 个候选、R3 summary 6 个、R4 summary 7 个，其中部分未写入 iteration 数组。agent 自述总候选数为 35，因此上表不是完整账本。**在 AutoIR 的 28 条 iteration 和 R2/R3/R4 summary 中，正确候选 `noise_impulse:5` 单独都没有出现；它只出现在规则控制器的候选表（附录 C）里。**


---

## 13. 附录 B：R2/R3/R4 summary-only 候选

### R2（`reflection_R2`）

| 候选 | PSNR | CI | 结果 |
|---|---:|---:|---|
| `brightness:1 + noise_gaussian_RGB:3` | 12.50 | 8/10 | REJECT |
| `brightness:1 + noise_speckle:3` | 13.11 | 8/10 | REJECT |
| `brightness:1 + noise_poisson:3` | 13.27 | 5/10 | REJECT |
| `brightness:1 + noise_impulse:5` | 10.98 | 9/10 | REJECT |

### R3（`reflection_R3`）

| 候选 | CI | 结果 |
|---|---:|---|
| `compression_jpeg:1 + brightness:1 + impulse:3` | 7/10 | REJECT |
| `brightness:1 + compression_jpeg:2 + impulse:3` | 6/10 | REJECT |
| `brightness:3 + impulse:3` | 6/10 | REJECT |
| `brightness:1 + impulse:2` | 8/10 | REJECT |
| `brightness:1 + contrast_strengthen_scale:1 + impulse:3` | 3/10 | REJECT |
| `saturate_strengthen_HSV:1 + brightness:1 + impulse:3` | 9/10 | REJECT |

### R4（`reflection_R4`）

| 候选 | CI | 结果 |
|---|---:|---|
| `saturate_strengthen_HSV:5 + impulse:3` | 7/10 | REJECT |
| `brightness:1 + saturate_strengthen_HSV:5`（无 impulse） | 7/10 | REJECT |
| `brightness:1 + saturate:5 + impulse:3` | 8/10 | REJECT |
| `compression_jpeg:1` alone | 5/10 | REJECT |
| `blur_gaussian:1 + impulse:3 + brightness:1` | 6/10 | REJECT |
| `quantization_hist:1 + brightness:1 + impulse:3` | 3/10 | REJECT |
| `contrast_strengthen_HSV` | — | 函数不存在，SKIPPED |

**注意：** 这三个表里没有 `noise_impulse:5` 单独。这正是失败的核心——正确候选不在搜索空间里。

---

## 14. 附录 C：规则控制器候选打分

| 候选 | signal score | PSNR（仿真） |
|---|---:|---:|
| **`noise_impulse:5`** | **1.00** | 11.79 |
| `noise_impulse:4` | 0.464 | 12.47 |
| `noise_impulse:3`（R0） | 0.33 | 13.00 |
| `noise_impulse:3 + JPEG:1` | 0.33 | 14.09 |
| `noise_impulse:3 + contrast:1` | 0.33 | 12.35 |
| `noise_impulse:3 + sat:1` | 0.33 | 11.50 |
| `noise_impulse:2` | 0.33 | 13.60 |
| `noise_impulse:1` | 0.33 | 14.29 |
| `noise_gaussian_RGB:3` | 0.33 | 12.88 |
| `noise_gaussian_YCrCb:3` | 0.33 | 12.08 |
| `noise_speckle:3` | 0.33 | 13.09 |
| `noise_poisson:3` | 0.33 | 13.66 |

规则控制器 R0 选 `noise_impulse:3`（0.33）；R1 选 `noise_impulse:5`（1.00）。R0 support 35.13；R1 support 43.17；R1 test 44.92/41.14。

---

## 15. 附录 D：一句话版本

> blind_0005 的 GT 是 `noise_impulse:5`；AutoIR 预测成 `brightness:1 + noise_impulse:3`。brightness 是 impulse 造成 35% clipping 的补偿性假阳性，severity 3 是低估；R0–R2 三次训练都用同一条错误 DC，R3/R4 只分析不训练。CLR 把错误管线的 CI=10/10 当作理论最优，把 GT PSNR=21.84 归因于模型容量，却从未测试 `noise_impulse:5` 单独。规则控制器在同一架构/预算下只改 DC 为 `noise_impulse:5`，得到 44.92/41.14 dB，AutoIR 为 21.92/22.17 dB。因此失败在 DC 搜索和 CLR 判据，不在 impulse noise 或 restorer 容量。E5c 的“只改 DC / 只改架构 / 两者”受控实验尚未执行；本案例的规则控制器对照是评估侧证据，不是盲 CLR 结果。
