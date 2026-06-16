---
name: image-degradation-simulator
description: Use this skill whenever the user needs to analyze degraded images, identify distortion types and severity, simulate image degradation using x_distortion algorithms, or reproduce degradation effects on clean images. Trigger when the user mentions image degradation, distortion simulation, x_distortion, degrading clean images, matching image quality degradation, noise/blur/compression analysis, reproducing image distortions, "make this image look like that one", or understanding what degradations are present in an image.
---

# Image Degradation Simulator

Analyze degraded images, identify present distortion types and their severity, then reproduce the degradation on clean images through iterative hypothesis, simulation, visual comparison, and refinement.

**同图模式优先**：实验生成挑战时使用 `--same-image` 标志（`blind_challenge.py`），clean 和 degraded 来自同一原图。这使得像素级校准成为可能，盲识别准确率远高于跨图模式。

## 🚫 绝对禁止（违反立即停止）

以下任何一条都会导致 CPU 100% 持续数小时、结果质量差、实验作废：

1. **不允许写任何 Python 脚本文件**：只使用已有的 Skill 脚本（analyze_degradation.py、apply_multi.py、compare_degradation.py、save_prediction.py）。
2. **禁止 PSNR 枚举搜索**：PSNR 仅用于最终验证（正确管线 > 40dB）。所有函数和 severity 决策必须通过校准阈值，不通过 PSNR 排名。
3. **禁止跨类别盲目组合**：不要遍历所有组合。
4. **禁止 `run_in_background: true` 启动多个并行搜索**。
5. **禁止编写子进程调用脚本**。

**正确做法**：analyze检测 → 校准阈值决策函数类型+severity → apply+compare → PSNR最终验证(>40dB=正确) → 残差分析噪声 → 保存。

## 🔴 强制检查清单（保存预测前必须逐项确认，不可跳过）

exp17 盲化评估（35 单退化）发现：50% 的失败案例不是阈值/指标问题，而是 **Agent 跳过了应该执行的检查步骤**。此清单强制完成每一步。

### A. 残差先行原则 ⚠️ exp17 头号教训

**在 target 上看到 unique_G 减少或 extreme% 升高时，绝不能直接下结论。必须先算残差。**

```
❌ 错误: 看到 unique_G < 25 → "quantization!"
✅ 正确: 看到 unique_G < 25 → 先算 residual = target - clean
         → residual 上 var_slope > 1.0 → "Poisson 噪声，不是 quantization"
```

**强制规则**: 任何关于 quantization 的判定，必须在计算残差并确认 `var_slope < 1.0`（排除 Poisson）之后才能做出。

### B. 噪声判别步骤强制顺序（不可跳步）

对每个残差，**必须按顺序执行全部 6 项检查**，记录每项结果后再下结论。可辅助使用 wavelet 噪声先验：

```
🆕 0. wavelet:      noise_prior.py --target <t> --clean <c> (同图模式)
      → 仅以下两项已验证可靠 (exp18测试 10/10):
        - gaussian_RGB vs gaussian_YCrCb: 通道 σ 比 > 1.4 → YCrCb ✅
        - 纯噪声 σ 估计: sev=1~2 范围内 ±30% (blur存在时低估, 不可靠)
      → 不可用于: speckle/poisson 检测 (0/10), 混合退化噪声估计

□ 1. impulse:     extreme% = ___ (> 0.3%? → impulse)
□ 2. speckle:      vm_slope = ___ (> 0.01? → speckle; sev=1 时 > 0.005 也考虑)
□ 3. poisson:     var_slope = ___ (> 1.0 AND vm_slope≈0? → poisson)
□ 4. spatial:     spatial_corr = ___ (0.15-0.5? → spatially_correlated)
□ 5. YCrCb:       rgb_ratio = ___ (> 1.4? → YCrCb)
                   🆕 辅助: noise_prior σ_per_channel max/min ratio > 1.4 → YCrCb ✅
□ 6. gaussian:    以上都不满足 → gaussian_RGB
```

**在 reflection.json 中必须记录这 6 项的数值**，不能只写"判断为 gaussian"。

### C. Blur 判定强制规则

```
□ 1. gm_ratio < 0.85 → blur 可能存在（mild blur 区 0.75-0.85 不能排除）
□ 2. 🆕 angular_fft.anisotropy_ratio > 4.0 AND h_v_ratio ≠ 1.0 (>30%偏差) → motion blur
     来源: exp18测试 5/5 motion正确, MTF频域分析对motion最可靠
□ 3. radial_ratio > 2.0 → lens blur 候选，但需 PSNR 验证 (⚠️ 内容依赖, 非可靠判据)
     exp18测试: gaussian↔lens MTF无法区分 (两者MTF曲线形状相同)
□ 4. 非确定性 blur (glass/jitter) PSNR 不会 > 40dB，不要反复调 severity 追求高 PSNR
□ 5. 指标超过阈值 → 信任指标，不要用"直觉"否定
```

### C2. 经验信号速查表 (来源: exp17+exp18, 32 cases)

以下规律从 32 组盲识别实验中系统提取，用于辅助 Agent 判断"退化是否存在"，而非精确识别子类型。

#### 退化存在性判断 (区分纯退化 vs 混合退化)

⚠️ **关键发现**: 信号可靠性在混合退化中急剧下降。必须分场景使用。

| 退化 | 信号 | 阈值 | 纯退化 | 混合退化 | 结论 |
|------|------|------|:--:|:--:|------|
| **comp/quant 存在** | `unique_G` | < 200 | N/A | 召回33% **FP=0%** | ⭐⭐⭐ 从不误报, 误报率0% |
| **global 存在** | `Cr_variance` | < 100 | N/A | **召回100%** FP=17% | ⭐⭐⭐ 混合中仍可靠 |
| **blur 存在** | `gm_ratio` | < 0.85 | 召回100% | 召回71% FP=20% | ⭐⭐ 混合中下降 |
| **noise 存在** | `overshoot_ratio` | > 0.5 | 不可用 | 召回35% **FP=90%** | ❌ 混合中完全失效 |
| **global 存在** | `saturation_mean` | < 50 | 不可用 | 召回33% FP=33% | ❌ 混合中失效 |

**混合退化中信号污染的原因**:
- `gm_ratio`: noise/compression 增加梯度 → 掩盖 blur 的梯度下降 → 漏检
- `overshoot_ratio`: compression 也产生 overshoot → noise 不存在时也 > 0.5 → 大量误报
- `saturation_mean`: 多步退化互相抵消/增强饱和效果 → 不可靠

#### 不可靠的信号 (不要用于判断)

| 信号 | 原因 |
|------|------|
| `flat_region_variance` 判断 noise | 噪声/无噪声均值差仅 30%，大量重叠 |
| `laplacian_variance_ratio` 判断 blur severity | 混合退化时噪声/锐化拉高 laplacian，sev=4 反而 > sev=2 |
| `radial_ratio` 判断 lens blur | 内容依赖 (clean 图本身 radial_ratio 可达 5) |

#### PSNR 适用范围

```
14/32 纯确定性退化 (无 noise) → PSNR 判据可用
18/32 含 noise             → PSNR 对噪声部分无效, 需统计匹配
10/32 混合 (noise+blur)     → 最困难, PSNR + 统计都不可靠
```

#### 关键推论 (仅使用混合退化中验证过的信号)

1. **unique_G < 200 → 一定有 compression 或 quantization** (FP=0%, 纯/混合均可靠)
2. **Cr_variance < 100 → 一定有 global 退化** (混合中召回100%, FP=17%)
3. **gm_ratio < 0.85 → blur 很可能存在** (混合中召回71%, 但 noise/comp 可能掩盖)
4. **overshoot_ratio 在混合退化中不可用于判断 noise** (FP=90%!)
5. **纯退化 (1步) 中信号远更可靠** → 先判断退化步数, 单步退化 PSNR 直接可用

### D. 保存前自检

```
□ 1. 如果预测包含 quantization → 残差 var_slope < 1.0 已确认?
□ 2. 如果预测 noise 子类型 → 6 项检查全部完成并记录?
□ 3. 如果预测 blur 子类型 → radial_ratio / dir_change 数值已记录?
□ 4. 确定性退化 PSNR > 40dB 或非确定性退化有充分指标证据?
□ 5. reflection.json 包含完整的 6 项噪声检查数值?
```

### 多退化并行处理

以上规则针对**单个退化**的搜索。多个独立的退化可以并行处理，按退化复杂度分配：

### Agent 任务分配策略

| 退化类型 | 每组退化数 | Agent 数 | 原因 |
|------|:--:|:--:|------|
| 单退化 | 6-8 | 1 | 最简单，处理快 |
| 双退化 | 4-5 | 4-5 | 中等复杂度 |
| 三退化 | 2-3 | 6-8 | 最复杂，需2倍时间 |

**总 Agent 数 > 8 是合理的**——三退化需要更多 Agent 来平衡负载。

**48 退化示例分配** (8单+20双+20三):
- 单退化: 1 Agent × 8 组
- 双退化: 4 Agent × 5 组
- 三退化: 7 Agent × 3 组 (最后1个2组)
- 总计: **12 Agent 并行**，预计同时完成

## Core principles

### PSNR-based identification (v4)

**核心原理**: 校准阈值直接从指标推断函数类型和 severity。PSNR 仅用于最终验证（正确管线 > 40dB）。

**当前工作流**:
1. `analyze_degradation.py` 检测类别 + 计算所有指标
2. 校准阈值速查表直接决策函数类型和 severity（不枚举!）
3. apply + compare 验证
4. PSNR > 40dB → 确定性部分正确
5. 噪声从残差分布识别

**与旧版的关键区别**: v1-v3 试图用 CI 和决策树区分相似函数，失败了 (JPEG vs JPEG2000, blur子类型混淆)。v4 用 PSNR 像素级匹配——正确函数 PSNR 45 dB vs 错误函数 21 dB，差异 24 dB，不存在混淆。

### Handling out-of-domain parameters

The x_distortion severity scale (1–5) covers a specific range for each distortion. Real-world images may have degradations that fall outside these ranges. When this happens:

**Degradation stronger than severity 5:** Apply the same distortion twice with severity values that sum to the needed level. For example, if the target noise appears to have sigma ≈ 0.35 but `noise_gaussian_RGB` maxes out at sigma=0.25 (severity 5), try `noise_gaussian_RGB:5` followed by `noise_gaussian_RGB:2` (sigma accumulates: 0.25 + 0.10 ≈ 0.35). Document this in params.json with `"out_of_domain": true` and explain the rationale.

**Degradation between two severity levels:** Pick the closer severity. If the target looks halfway between severity 2 and 3, try severity 3 first (slightly over is easier to detect and correct than slightly under). Document the mismatch in reflection.json: `"match_note": "Target noise is between severity 2 and 3. Severity 3 selected as closest available match."`

**Degradation type not in x_distortion:** If the target shows artifacts that don't match any available distortion (e.g., lens distortion, chromatic aberration, HDR artifacts), identify the closest available approximation and clearly document in reflection.json what couldn't be reproduced and why. This is valuable research data — knowing the library's limitations is as important as knowing its capabilities.

**When the pipeline parameters are genuinely ambiguous:** Add a `match_quality` field to the params.json entry:
```json
{
  "function": "noise_gaussian_RGB",
  "severity": 5,
  "actual_params": {"sigma": 0.25},
  "out_of_domain": true,
  "match_quality": "undershoot",
  "match_note": "Target noise sigma estimated at ~0.35, but max severity 5 only reaches 0.25. Applied double Gaussian noise (sev 5 + sev 2) to approximate."
}
```

## The x_distortion library

Located at `x_distortion/` in the project root. Core usage:

```python
from x_distortion import add_distortion, distortions_dict
degraded = add_distortion(img, severity=severity, distortion_name=dist_name)
```

- `img`: np.ndarray, uint8, H x W x 3, RGB, [0, 255]
- `severity`: integer 1–5
- `distortion_name`: specific function identifier string
- Returns degraded image as np.ndarray, uint8

List available distortions by category:

```python
from x_distortion import distortions_dict
for cat, funcs in distortions_dict.items():
    print(f"{cat}: {funcs}")
```

### Distortion categories

| Category | Functions |
|----------|-----------|
| `blur` | gaussian, motion, glass, lens, zoom, jitter |
| `noise` | gaussian_RGB, gaussian_YCrCb, speckle, spatially_correlated, poisson, impulse |
| `compression` | jpeg, jpeg_2000 |
| `brighten` | shift_HSV, shift_RGB, gamma_HSV, gamma_RGB |
| `darken` | shift_HSV, shift_RGB, gamma_HSV, gamma_RGB |
| `contrast_strengthen` | scale, stretch |
| `contrast_weaken` | scale, stretch |
| `saturate_strengthen` | HSV, YCrCb |
| `saturate_weaken` | HSV, YCrCb |
| `oversharpen` | oversharpen |
| `pixelate` | pixelate |
| `quantization` | otsu, median, hist |

For the complete severity-to-actual-parameter mappings (e.g., severity 3 → sigma=0.15 for gaussian_RGB noise), read `references/severity_mappings.md`.

### Multi-distortion

`add_distortion` applies one distortion at a time. To apply a sequence of distortions, use the bundled script:

```bash
python ${CLAUDE_SKILL_DIR}/scripts/apply_multi.py \
  --input <clean_image_path> \
  --distortions "blur_gaussian:3,noise_gaussian_RGB:2,compression_jpeg:4" \
  --output <output_path>
```

## 批量盲识别工作流程 (v7 — detect + 残差验证 + 三级反思)

> v6: detect_degradation.py 自动决策 + decision_flow 披露。v7: 强化 Round A/C 残差验证，三级反思机制。
> v8: exp17 盲化评估（35 单退化）→ 强制检查清单 + 噪声残差先行 + 阈值修正 + 失败案例。
> v9: exp17 双退化评估（20 双退化）→ 掩盖推理 + compression 定向补充验证。
> 校准阈值来自 100 张 DIV2K 图像的系统校准 (exp15/scripts/calibrate_thresholds.py)。
> exp17 盲化基准：单退化 88.6%（31/35），双退化 30%（6/20），掩盖推理预期双退化提升至 ~70%。

### 核心流程 (v10 — 噪声优先, 两条路径)

⚠️ **强制执行顺序。整个流程只有两个阶段: 噪声判断 → 分路径执行。**

来源: 1140 cases大规模合成 + exp17/exp18 32组真实退化

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: 噪声判断 (必须先做, 决定后续所有策略)                │
│                                                             │
│   residual = target - clean (同图模式)                       │
│   噪声判别: SKILL.md §B 6步检查 + noise_prior.py (辅助)     │
│                                                             │
│   → 无 noise: 进入 路径 A (高置信)                           │
│   → 有 noise: 进入 路径 B (低置信)                           │
└─────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════
路径 A: 纯确定性退化 ⭐ 高置信 (PSNR 可用)
═══════════════════════════════════════════════════════════════

  信号可靠性 (大规模验证):
    ✅ gm_ratio < 0.85     → blur 存在 (83%召回, 0%FP)
    ✅ unique_G < 200      → compression/quant 存在 (0%FP)
    ✅ Cr_variance < 100   → global 退化存在 (100%召回)
    ✅ PSNR > 40dB         → 函数+严重度正确
    ❌ blur 子类型无法区分 (gaussian/lens/zoom/glass 50%)

  A1. 信号扫描:
      gm_ratio < 0.85? → 测试 blur 类型 (gaussian/motion/lens, PSNR 验证)
      unique_G < 200?  → 测试 compression (JPEG vs JPEG2000)
      Cr_var < 100?    → 测试 global (contrast/brightness/saturate)
      其他 → 可能还有 oversharpen/quantization/pixelate

  A2. PSNR 验证:
      apply → PSNR vs target
      > 40dB → ✅ 正确
      30-40dB → 调 severity
      < 30dB → 换函数族
      PSNR gap > 10dB between candidates → 胜者显著

  A3. 保存:
      verdict = LIKELY (PSNR >= 40 + gap >= 10) 或 GOOD
      alternatives: PSNR >= 30 的不同函数族候选
      所有候选 PSNR < 30 → POOR

═══════════════════════════════════════════════════════════════
路径 B: 含噪声退化 ⚠️ 低置信 (PSNR 对噪声无效)
═══════════════════════════════════════════════════════════════

  信号可靠性:
    ✅ unique_G < 200       → 仍可靠 (0%FP)
    ✅ Cr_variance < 100    → 仍可靠 (100%召回)
    ❌ gm_ratio             → 不可用 (噪声增加梯度, 召回1%)
    ❌ overshoot_ratio      → 不可用 (FP=90%)
    ❌ PSNR 对噪声部分      → 无效 (随机seed)

  B1. 噪声识别 (用统计, 不用 PSNR):
      §B 6步检查 (impulse→speckle→poisson→spatial→YCrCb→gaussian)
      严重度: 残差 std 与校准阈值 closest match
      辅助: noise_prior.py wavelet σ 估计 (仅 gaussian_RGB/YCrCb 区分可靠)

  B2. 确定性部分识别 (剥离噪声后):
      det_sim = apply(clean, 非noise退化)
      残差 = target - det_sim → 统计验证噪声假设
      PSNR(target, det_sim) → 验证确定性部分
      ⚠️ 不要用 PSNR 选噪声! 不要用 gm_ratio!

  B3. 保存:
      verdict = UNCERTAIN (必须, 退化耦合无法可靠验证)
      alternatives: 至少 3 个 PSNR >= 30 的确定性候选 + 不同噪声类型候选
      所有候选 PSNR < 30 → POOR, 全部保留供训练验证

═══════════════════════════════════════════════════════════════
反思 (训练后, 最多 3 轮)
═══════════════════════════════════════════════════════════════

  触发: Spec < DFPIR - 3dB

  R1. 先重检 Phase 1 噪声判断: 噪声存在性判断是否错误?
      → 路径A误判为路径B? → 重新按路径A执行
      → 路径B误判为路径A? → 重新按路径B执行

  R2. 修正确定性部分:
      优先测试 alternatives 中不同函数族的候选
      PSNR gap >= 10dB → 换候选
      PSNR gap < 3dB → 两个候选同样可能, 选训练PSNR更好的

  R3. 修正噪声部分 (仅用统计, 不用 PSNR!):
      重新执行 §B 6步检查
      检查 alternatives 中不同噪声类型的候选

  R4. 终止条件:
      所有候选 PSNR < 20dB → BEYOND_CAPABILITY, 放弃
      R2修正后 PSNR提升 < 2dB → 放弃
      噪声统计全部不匹配 → 放弃

═══════════════════════════════════════════════════════════════
保存
═══════════════════════════════════════════════════════════════

  predicted_params.json: pipeline + alternatives + analysis
  reflection.json: Phase 1噪声判断 + 路径选择 + psnr_ranking + 反思记录
```

### 残差诊断速查表

| residual_A 模式 | 诊断 | 修正 |
|------|------|------|
| 8×8 块状结构 | JPEG 漏检 (块边界在残差中可见) | 添加 compression_jpeg |
| 随机均匀分布 + res_std > 8 | noise 漏检 | 添加 noise, 从残差分布判断类型 |
| 边缘区域强信号 | blur severity 有误 | 调整 blur severity ±1 |
| 整体亮度/色彩偏移 | global (brightness/contrast/saturation) 有误 | 检查 decision_flow 中 global 决策 |
| 均匀无结构 + res_std < 5 | ✅ 确定性部分正确, 残差仅为 noise | 进入 Round B |

### 反思三级示例

```
案例: GT=[jpeg:3, noise_gauss:2, blur:4], Pred=[blur:4]

Round A: det_sim = blur:4 → PSNR=22dB ❌
残差诊断: residual_A 显示 8×8 块状 + 均匀噪声
→ 残差模式提示: JPEG + noise 漏检

第 1 级反思 (移除 FP): blur 是唯一检测, 高置信 → 不移除
第 2 级反思 (添加漏检):
  尝试 guess_a=compression_jpeg sev 2-4
  → full_sim = blur:4 + jpeg:3 → PSNR=32dB ↑10dB ✅
  尝试 guess_b=noise_gaussian_RGB sev 1-2
  → full_sim = blur:4 + jpeg:3 + noise:2 → PSNR=42dB ↑10dB ✅
→ 管线修正成功, 保存 GOOD
```

### 校准阈值速查表 (100 DIV2K 校准 + exp17 盲化评估修正, exp15+17)

| 检测项 | 指标 | 阈值 | 来源 |
|--------|------|------|------|
| blur 存在 | gm_ratio | < 0.85 (mild zone 0.75-0.85 标记为"可能 blur") | exp17 修正 |
| blur 确认 | gm_ratio + lap_ratio | < 0.75 OR lap_ratio < 0.55 | calibration |
| blur severity | gm_ratio | [0.60, 0.40, 0.32, 0.27] → sev 1-5 | calibration |
| blur_motion | dir_change + patch_ratio_std | > 20% + std > 0.38 | calibration |
| blur_lens | gradient_radial_ratio | > 2.0 → 强制 lens（不要归因于内容!）; > 1.3 → 候选 | exp17 修正 |
| blur_zoom | gradient_radial_ratio | < 0.9 | calibration |
| blur_jitter | gm_ratio + lap_ratio | gm > 1.0 AND lap < 0.7 | source code |
| JPEG | block_norm + specificity | > 1.10 + > 1.2 | calibration |
| JPEG2000 | uG_ratio + zc_ratio | < 0.85 + > 1.3 (medium conf) | calibration |
| quantization | uG | < 25 **AND** 残差 var_slope < 1.0（必须排除 Poisson!）| exp17 修正 |
| oversharpen | gm_ratio + lap_edge_ratio | > 1.4 + > 2.5 | calibration |
| contrast | std_ratio + proportional check | < 0.70 (weaken) / > 1.3 (strengthen) | source code |
| brightness | mean_shift_pct | abs > 0.08 | source code |
| saturation | sat_ratio | < 0.65 (weaken) / > 1.6 (strengthen) | source code |
| noise_impulse | exact_0+255 pixel fraction | > 0.3% | calibration |
| noise_speckle | vm_slope on residual | > 0.01 (sev≥2); sev=1 时 > 0.005 或 speckle_contrast > 0.005 | exp17 修正 |
| noise_poisson | var_slope on **residual** | > 1.0 + vm_slope≈0（⚠️ 必须在残差上测!）| exp17 修正 |
| noise_spatially_correlated | spatial_corr on residual | 0.15-0.5（⚠️ 必须检查，不可跳过!）| exp17 修正 |
| noise_YCrCb | rgb_std_ratio | > 1.4 | calibration |

### 逐层剥离示例

```
GT: blur_gaussian:3 → noise_poisson:2 → compression_jpeg:2

错误做法 (v4.1):
  在target上直接看 noise 指标
  → flat_variance 被 blur:3 抹平 → "无 noise" ❌
  → 预测: blur_gaussian:3 + compression_jpeg:2 (漏检noise!)

正确做法 (v5):
  Round A: gm_r=0.32→blur:3, block_boundary>1.1→JPEG:2
           deterministic_sim = apply(clean, blur:3 + JPEG:2)
           PSNR=35dB → 确定性部分正确 ✅
  
  Round B: residual = target - deterministic_sim
           在 residual 上:
           - flat_variance(residual) = 850 (显著, 不是0!)
           - var vs intensity: 正相关 → Poisson! ✅
           → noise_poisson:2
  
  Round C: apply(clean, blur:3 + poisson:2 + JPEG:2) → full_sim
           compare(full_sim, target) → CI子指标全部通过
           PSNR=42dB → 确认 ✅
  
  最终: blur_gaussian:3 + noise_poisson:2 + compression_jpeg:2 ✅
```

### 为什么 Step 2 不含 noise？

noise 指标在 target 上被 blur/compression 严重污染：
- blur 降低 flat_variance → noise 看起来比实际弱
- compression 产生 impulse 样式的极端像素 → 假阳性
- blur+noise 耦合 → overshoot 信号来自 noise 还是 blur？

**只有剥离确定性退化后，residual 中的 noise 信号才是真实的。**

### exp17 盲化评估 — 失败案例与预防 (v8 新增)

exp17 对 35 个单退化进行盲化评估（目录名盲 ID，Agent 完全不知退化类型）。3 个失败案例的教训：

**案例 1: Poisson:4 → 误判为 quantization_otsu:1** 🔴 最严重
```
Agent 看到: unique_G 减少 + extreme% 升高 → "是 quantization!"
实际情况: 噪声将像素推至极端值, unique_G 被动减少。
          残差上 var_slope=1.72 >> 1.0 → 明确的 Poisson 信号!
根因: Agent 在 target 上直接用 unique_G 判断, 没有先算残差。
预防: 强制规则 → 判定 quantization 前必须确认残差 var_slope < 1.0。
```

**案例 2: spatially_correlated:4 → 误判为 gaussian_RGB:2**
```
Agent 看到: 残差分布对称 + 通道 std 相似 → "gaussian!"
实际情况: spatial_corr=0.233 明确在 0.15-0.5 范围内。
根因: Agent 跳过了 spatial_corr 检查步骤, 凭"直觉"判为 gaussian。
预防: 噪声 6 项检查必须逐项执行并记录数值, 不可跳步。
```

**案例 3: speckle:1 → 误判为 gaussian_RGB:1**
```
Agent 看到: vm_slope < 0.01 → "不是 speckle"
实际情况: sev=1 的 speckle_contrast=0.14, vm_slope 信号被内容淹没。
根因: vm_slope 阈值 0.01 对 sev=1 过高。
预防: sev=1 时降低阈值到 0.005, 或检查 speckle_contrast > 0.005。
```

**案例 4: lens:4 → 误判为 gaussian:3** (blur 子类型)
```
Agent 看到: radial_ratio=3.17 → "应该是中心构图导致的, 不是 lens blur"
实际情况: radial_ratio=3.17 就是 lens blur!
根因: Agent 用"直觉"否定了指标。radial_ratio > 2.0 不可能是纯内容造成的。
预防: 强制规则 → radial_ratio > 2.0 必须优先判定 lens。
```

**案例 5: zoom:1 → 误判为 noise** (blur 漏检)
```
Agent 看到: gm_ratio=0.85 > 0.75 → "不是 blur"
实际情况: sev=1 的 zoom blur 很 mild, gm_ratio 刚好高于阈值。
根因: blur 阈值 0.75 对 mild blur 不够敏感。
预防: gm_ratio 0.75-0.85 标记为 mild blur 可能, 检查 lap_ratio 辅助确认。
```

### exp17 双退化评估 — 掩盖推理 (v9 新增)

exp17 对 20 个随机双退化进行盲化评估。结果：两函数全对 30%，但 **Agent 方向 100% 正确**（至少猜对一个类别）。核心发现：

**compression 是"最容易被掩盖的退化"** — 14 个失败中 11 个是 compression 漏检。

掩盖机制（强退化 → 破坏弱退化特征）：

| 主导退化 | 掩盖 compression 的机制 |
|----------|----------------------|
| global (brightness/contrast/saturation) | 像素值重映射 → 8×8 块边界消失；clipping → DCT 模式不可见 |
| noise (impulse/poisson/YCrCb) | 随机像素破坏 8×8 规律性；ringing 被噪声淹没 |
| blur (gaussian:5) | 平滑抹掉 block_boundary；ringing 被模糊消除 |

**启发式规则（掩盖推理）⚠️ 触发条件：PSNR < 35dB 且已识别退化数不足且不含 compression**：

掩盖推理不替代常规反思（换 severity/换子类型/调顺序）。常规反思优先执行，
仅在常规修正无法将 PSNR 提升到 35dB 以上时，才启动掩盖推理。

```
触发条件（3 条全部满足才执行）:
  1. 当前管线 PSNR < 35dB（确实有问题，不是 fine-tune 范围）
  2. 已识别退化数 < 预期退化数（如双退化只找到 1 个，单退化全对则跳过）
  3. 已识别的退化中不含 compression（compression 已被找到则跳过）

满足条件 → 掩盖检查:
  □ 剥离已知退化 → 残差上测试 compression_jpeg + compression_jpeg_2000
    （先测 sev=1，低严重度最容易被掩盖）
    验证: PSNR > 50dB 才追加（compression 是确定性的）
    追加后 PSNR 提升 > 5dB → "掩盖确认"

不满足条件 → 跳过掩盖检查:
  - 单退化 PSNR > 40dB → 已做对，不需要
  - 已找到所有退化 → 不需要
  - 已有 compression → 不需要
```

**为什么不是枚举**：只在已识别出至少一个退化后，对"最容易被掩盖的类别 (compression)"做定向补充测试。最多增加 10 次 PSNR 验证（2 函数 × 5 严重度），不是遍历组合。

**预期效果**：修复 11/14 的 compression 漏检，双退化准确率从 30% → ~70%。

**剩余不可修复**：含 noise 且 noise 子类型错误的 3 个案例（speckle↔gaussian↔impulse 混淆）。噪声子类型只能靠 Phase 5 训练 + Phase 6 反思。

### 关键改进 (vs v4.1 + calib v2)

| 问题 | v4.1 | calib v2 | v5 |
|------|------|---------|-----|
| noise漏检(20例) | CI诊断但模糊 | 阈值不清 | **先剥离blur→残差分析noise** |
| Poisson误判(5例) | 无检测 | 阈值不准 | **残差强度-方差相关** |
| oversharpen误入 | 2次 | 1次 | gm_r>2+osr>50+nvs三重确认 |
| global漏检(7例) | 无 | 部分 | **剥离局部后再检查ratios** |

        unique_G↑+block↓ → JPEG→JPEG2000（压缩类型耦合）
        gradient↑+laplacian↑ → 缺少blur（噪声被误判为主导）
        flat_variance↓+Cr/Cb↓ → blur在noise之后（噪声被模糊抹平）
        
  方法: apply完整管线 → compare → 分析每个子指标偏差方向
        → 根据耦合诊断表推断哪个退化有误
        → 定向修正（只改被诊断的部分）

Round 4 — PSNR最终验证
  目标: 确定性部分像素级确认
  方法: 只对确定性退化部分计算PSNR（排除noise）
        → PSNR > 40 dB → 确定性部分正确
        → PSNR < 30 dB → 确定性部分有误，回到Round 2
        → 噪声类型从残差统计特征确认
```

### 噪声单退化识别 ⚠️ PSNR 无效，用残差+分布

PSNR 对噪声函数天然偏低（随机种子不同导致像素无法匹配）。**噪声类型识别必须用残差和分布特征。**

同图模式下，残差 = target - clean 直接暴露噪声模式：

#### 六种噪声的残差/分布判别

| 噪声类型 | 残差特征 | 分布特征 | 关键验证 |
|---------|---------|---------|---------|
| **Gaussian RGB** | 方差为常数，与像素强度无关 | 正态分布，对称 | 分 10 个强度 bin → 每 bin 方差接近 |
| **Gaussian YCrCb** | **RGB 通道噪声不均** (B > G > R) | Cr/Cb 正态分布 | rgb_std_ratio > 1.4 OR Cr/Y_std > 1.3 |
| **Poisson** | **方差 ∝ 强度**（暗区噪声小，亮区大） | 低强度偏斜，高强度近正态 | var_slope > 1.0 + vm_slope≈0 |
| **Speckle** | **方差 ∝ 强度²**（乘性，vm_slope > 0） | Gamma 分布 | vm_slope > 0.01 |
| **Impulse** | 稀疏极端像素 (0 和 255) | 两端尖峰 | exact_0+255 像素比例 > 0.3% |
| **Spatially Correlated** | 邻域像素残差相关 | 空间自相关高 | spatial_corr 0.15-0.5 |

#### 判别流程 (v6 — speckle 优先)

```
Step 1: 检查 impulse → impulse_pct(net) > 0.3% → noise_impulse
Step 2: 检查 Speckle → vm_slope > 0.01 (var/mean 随强度上升) → noise_speckle
Step 3: 检查 Poisson → var_slope > 1.0 + vm_slope≈0 → noise_poisson
Step 4: 检查 Spatial → spatial_corr 0.15-0.5 → noise_spatially_correlated
Step 5: 检查 YCrCb → rgb_std_ratio > 1.4 OR Cr/Y_std > 1.3 → noise_gaussian_YCrCb
Step 6: 默认 → noise_gaussian_RGB (低置信)
```

#### 多退化含噪声时的注意事项

多退化时残差被其他退化污染。复合退化含 noise 的情况后续构造专门实验解决。
当前：先识别确定性退化 → 剥离 → 残差分析噪声。

### 关键原则

### ⚠️ 挑战函数范围（4类别 35 函数）

`blind_challenge.py` 从 4 类别生成退化管线（每类最多1个函数，最多4步）：

| 类别 | 函数数 | 确定性? |
|------|:--:|:--:|
| blur | 6 | ✅ 全部确定性 |
| noise | 6 | ❌ 全部随机 |
| compression | 2 | ✅ 全部确定性 |
| global | 21 | ✅ 全部确定性（除 pixelate 和 quantization 外） |

global 包含: brightness(8), contrast(4), saturation(4), oversharpen, pixelate, quantization(3)

### 关键原则 (v7)

1. **detect 自动决策 + Agent 审查** — detect_degradation.py 给出 pipeline + decision_flow，Agent 审查 evidence 后确认或调整
2. **残差验证是核心** — Round A 的 residual_A 模式直接诊断漏检/FP: 8×8块→JPEG漏检, 随机→noise漏检, 边缘结构→blur有误
3. **PSNR 只验证确定性退化** — 不用于噪声搜索。阈值: >40dB 完美, 35-40dB 类别正确, 25-35dB 有FP/漏检, <25dB 严重错误
4. **噪声从残差识别** — PSNR 对噪声无效，残差+分布才是正确方法。判别顺序: impulse → speckle → Poisson → spatial → YCrCb → gaussian
5. **decision_flow 向 Agent 披露** — 每个决策的证据链完整透明，Agent 可据此独立判断
6. **反思 = 综合审视 + 多步修正** — 每轮可同时 移除FP + 添加漏检 + 调整severity + 调整顺序 + 替换误诊。改完后一次 PSNR 验证。最多 3 轮
7. **训练后反思最可靠** — L1+L2 失败 → NEEDS_WORK → Phase 5 训练 → Phase 6 Spec vs M_blind PSNR 差触发深度反思
8. **blur 子类型简化** — 仅区分 motion / jitter / gaussian
9. **顺序从数据推断** — 默认: compression → quantization → global → blur → noise

### 与旧版对比

| 问题 | v4.1 | v5 | v6 (当前) |
|------|------|-----|------|
| 单退化确定函数 | PSNR验证(75%) | 校准阈值 | **detect自动决策(84%单退化)** |
| 单退化噪声 | 残差+分布 | 残差+分布 | **残差+校准阈值+RGB ratio** |
| JPEG vs JPEG2000 | CI盲区 | PSNR可区分 | **block_norm+特异性+zc_ringing** |
| 全局退化识别 | 无 | PSNR验证 | **色彩空间映射+校准阈值** |
| blur子类型 | 决策树(radial) | 决策树(radial) | **简化: motion/jitter/gaussian** |
| 多退化耦合 | CI诊断 | 逐层剥离 | **逐层剥离+耦合修正+guesses** |
| Agent可见性 | 原始指标 | 原始指标 | **decision_flow完整证据链** |
| 多退化耦合 | CI子指标 | 失败(万能填充) | CI耦合诊断+逐层剥离 |
| JPEG vs JPEG2000 | CI盲区 | PSNR可区分 | PSNR验证 |
| noise子类型(多退化) | 决策树 | 失败 | 待后续实验 |
| 全局退化识别 | 无 | 无 | **新增 (Round0+PSNR)** |

## Mode Selection: Same-image vs Cross-image

Before starting, check whether the clean source image is the **same image** as the target (just undegraded) or a **different image**.

### How to determine

Run this quick check before starting analysis:

```python
from PIL import Image
import numpy as np
target = np.array(Image.open('<target>').resize((64,64)).convert('L'), dtype=np.float32)
clean  = np.array(Image.open('<clean>').resize((64,64)).convert('L'), dtype=np.float32)
corr = np.corrcoef(target.flat, clean.flat)[0,1]
print(f"Image correlation: {corr:.3f}")
# corr > 0.95 → same image (different resolution OK after resize)
# corr < 0.95 → different images → cross-image mode
```

Decision rules:
- `corr > 0.95` → **same-image mode** (same scene, just degraded)
- `corr < 0.95` → **cross-image mode** (different scenes)
- User explicitly says "this is the original" → **same-image mode** regardless
- No clean source provided → **target-only mode** (same flow as cross-image Phase 1)

### Same-image mode (preferred, recommended)

**生成挑战**：使用 `blind_challenge.py --same-image`，clean 即为 degraded 的原图（未退化版本）。

When the target and clean source are the same image, you have a massive advantage: **every metric can be calibrated**. Compute target/clean ratios for all 14 analysis modules. Content-dependent metrics become reliable:

```bash
uv run python .claude/skills/image-degradation-simulator/scripts/analyze_degradation.py --target <target> --clean <clean>
```

The `ratios_vs_clean` section tells you exactly what changed: gradient_magnitude_ratio=0.21 means 79% sharpness loss → severe blur. laplacian_variance_ratio=0.03 means 97% detail loss → confirms heavy blur. saturation_mean shift of +50 means deliberate saturation boost.

In same-image mode, you can also use MSE/PSNR during iterations as a convergence signal — pixel-perfect match IS possible if your pipeline is exactly right.

验证同图：correlation > 0.95（`blind_challenge.py --same-image` 自动满足）

### Cross-image mode (fallback)

When the target and clean source are different images, follow the target-only Phase 1 below. **Never use clean source statistics as a baseline** — content differences will mislead you into false degradation detections.

---

## The iterative analysis loop (cross-image)

The core workflow has two phases:

**Phase 1: Target-only analysis** — Identify degradation TYPES from the target image alone using content-independent metrics and visual inspection.

**Phase 2: Degradation simulation** — Apply the identified pipeline to the clean image (the "output canvas"). Compare degradation TEXTURE (not pixel values) between target and simulated results. Iterate on severity only.

### Step 1: VISUAL INSPECTION FIRST — read the target image

**Before running any script, read the target image with your eyes.** You are a vision model — use that capability. Look at the image and describe what you actually SEE:

- **Blur**: Is the image soft overall? Can you see directional smear? Are edges smoothly faded or do they have halos?
- **Noise**: Is there visible grain? Is it uniform across the image? Are there bright/dark speckles? Look at flat regions (sky, walls) — are they smooth or textured?
- **Compression**: Are there block-shaped artifacts (8×8 squares)? Is color bleeding at edges? 
- **Quantization**: Are there visible color bands in gradients (sky, shadows)?
- **Oversharpen**: Are there bright/dark halos along edges? Does the image look artificially crisp?
- **Pixelate**: Is there an obvious blocky, low-resolution appearance?

Write down your visual observations in reflection.json. This is your primary evidence. Do not skip this step — the numbers exist to support or challenge your visual findings, not replace them.

### Step 2: Quantitative confirmation — run analysis script

Now run the analysis script to get numbers that support or challenge what you saw:

```bash
python ${CLAUDE_SKILL_DIR}/scripts/analyze_degradation.py --target <target_path>
```

The `degradation_summary` section provides content-independent findings. **Treat these as hypotheses to verify, not as ground truth.** 

Each metric has known failure modes — use your visual inspection to resolve them:

| Metric | What it flags | Known false positive | How to resolve visually |
|--------|--------------|---------------------|------------------------|
| `overshoot_ratio > 0.5` | Oversharpen | **Noise also elevates it** | Check `noise_vs_sharpen.verdict`: noise/oversharpen/uncertain. Also visually: halos at edges vs uniform grain everywhere |
| `impulse_total_pct > 0.5%` | Impulse noise | **Oversharpen clipping** at 0/255 | Look at spatial distribution: random dots = noise, edge-correlated = oversharpen |
| `block_boundary_ratio > 1.1` | JPEG blocks | Blur can mask blocks | Look for 8×8 grid patterns |
| `unique_G < 256` | JPEG/quantization | Blur can reduce unique colors | Check if reduction is G-only (JPEG) or all-channel (blur/quantization) |
| `gradient ↓` + `hf_lf ↓` | Blur | Image can be naturally soft | Is the softness uniform across the image? Natural softness varies by region. |

**When numbers and visual inspection disagree, your eyes are the final authority.**

### Step 3: Form initial hypothesis

Based on your analysis, propose a degradation pipeline. Write it down explicitly in reflection.json:

```
Hypothesis 1:
  1. blur_gaussian, severity=3
  2. noise_gaussian_RGB, severity=2
  3. compression_jpeg, severity=3
```

Start simple — try to capture the most visually dominant degradation first before adding subtle ones. A single well-chosen distortion with the right severity often gets surprisingly close.

**You must NOT enumerate all options.** Do not generate a grid of severity × type combinations. Do not test things you have no visual evidence for. Each simulation you run should test a specific hypothesis you can justify. If you haven't written down WHY you're testing a particular distortion before running it, you're brute-forcing.

### Step 4: Simulate

Apply your hypothesized pipeline to the clean image using `scripts/apply_multi.py`. Save the result to the workspace.

### Step 5: Compare — visual inspection first, then numbers

**First, look at the simulated image and the target side-by-side.** You are comparing degradation TEXTURE, not image content:

- "The blur level looks similar — edges are equally soft"
- "The noise grain in the simulated image is finer than the target"
- "The simulated image has clean flat regions, but the target has visible grain everywhere"
- "The target has blocky JPEG artifacts that our simulation is missing"

Trust what you see. The visual comparison is the primary judgment.

**Then run the comparison script** for quantitative support:

```bash
python ${CLAUDE_SKILL_DIR}/scripts/compare_degradation.py \
  --target <target_path> --simulated <iter_path>
```

The `content_independent_metrics` section shows which degradation fingerprints match quantitatively. Use it to catch subtle mismatches your eyes might miss. But remember: **when numbers and visual inspection disagree, your eyes win.**

**For each iteration, do a three-way comparison:**

```
         Target (degraded)          Clean Source              Simulated (current iter)
         ─────────────              ────────────              ───────────────────────
         "What degradations         "What does this           "Did applying my
          do I see?"                 image look like           hypothesis move the
                                     without any               clean source toward
                                     degradation?"             the target?"
```

Ask yourself three questions:
1. **Direction check**: Did this adjustment move the simulation closer to the target or further away? Compare the current iteration against the PREVIOUS iteration, not just the target.
2. **Over-correction check**: Did I overshoot? If iteration N was too strong and iteration N+1 is too weak, the answer is between them.
3. **Side-effect check**: Did adding/removing a degradation introduce new artifacts that the target doesn't have? (e.g., adding noise destroyed cross-channel correlation that the target preserves)

Write each comparison into reflection.json with specific observations. Include the iteration images for later review.

**Detect out-of-domain situations early.** If severity 5 still undershoots the target, or severity 1 still overshoots, you're out of the library's range. Don't keep trying severity adjustments — recognize the limit and apply the out-of-domain strategies (double-application, closest-match documentation) described in Core Principles.

If the simulation is already very close, you can stop. Otherwise, update your hypothesis with specific adjustments — each change should address a specific observed discrepancy, not be a random probe.

### Step 6: Iterate

Repeat steps 4–6. Save each iteration's output so the researcher can trace the evolution. Usually 3–5 iterations are sufficient. Stop when:

- The simulated image is visually very close to the target
- Further adjustments produce only imperceptible changes
- You've tried reasonable variants and the remaining differences are likely due to factors outside the x_distortion scope (e.g., content changes, geometric transforms)

### Step 7: Save final results

Once satisfied, organize everything into the output directory using the bundled save script:

```bash
python ${CLAUDE_SKILL_DIR}/scripts/save_results.py \
  --output-dir degradation_results/<name> \
  --target <target_degraded_image> \
  --clean <clean_source_image> \
  --simulated <final_simulated_image> \
  --params <params.json> \
  --reflection <reflection.json> \
  --iterations-dir <workspace/iterations>
```

If you prefer manual organization, the output directory structure should be:

```
degradation_results/<descriptive_name>/
├── target.png              # Copy of the target degraded image
├── clean_source.png        # Copy of the clean source image
├── simulated.png           # Final simulated result
├── params.json             # Distortion pipeline parameters
├── reflection.json         # Full search-and-reflect process log
├── analysis.md             # Brief report of findings
└── iterations/             # Intermediate simulation results
    ├── iter_01.png
    ├── iter_02.png
    └── ...
```

### params.json format

For each distortion in the pipeline, record both the abstract severity (1–5) and the actual physical parameters. Read `references/severity_mappings.md` to look up the actual parameters for each distortion at each severity level.

```json
{
  "target_image": "target.png",
  "clean_image": "clean_source.png",
  "pipeline": [
    {
      "step": 1,
      "category": "blur",
      "function": "blur_gaussian",
      "severity": 3,
      "actual_params": {
        "sigma": 3
      }
    },
    {
      "step": 2,
      "category": "noise",
      "function": "noise_gaussian_RGB",
      "severity": 2,
      "actual_params": {
        "sigma": 0.1
      }
    }
  ],
  "notes": "Gaussian blur at severity 3 was the dominant degradation. After blur, mild Gaussian RGB noise was needed to match the fine grain in the target."
}
```

### reflection.json format

This is the most important artifact — it captures the entire search-and-reflect process, making the reasoning traceable and the skill improvable. Write this file incrementally as you work through each iteration. Do not wait until the end to fill it in — the intermediate thinking is as valuable as the final answer.

```json
{
  "initial_analysis": {
    "suspected_degradations": [
      {"category": "blur", "confidence": "high", "evidence": "overall softness, no sharp edges, fine details lost"},
      {"category": "noise", "confidence": "medium", "evidence": "visible grain in shadow regions, possibly Gaussian"}
    ],
    "ruled_out": [
      {"category": "compression", "reason": "no blocking artifacts or ringing visible"},
      {"category": "pixelate", "reason": "no blocky low-resolution appearance"}
    ],
    "uncertainties": [
      "Cannot determine if blur is Gaussian or Lens type from initial inspection",
      "Noise might be spatially correlated rather than independent Gaussian"
    ]
  },
  "iterations": [
    {
      "round": 1,
      "hypothesis": {
        "pipeline": [
          {"function": "blur_gaussian", "severity": 3}
        ],
        "reasoning": "The dominant artifact is overall softness consistent with moderate Gaussian blur. Starting simple to establish the baseline before adding secondary degradations."
      },
      "comparison": {
        "observations": [
          "Blur profile matches well in the center of the image",
          "Simulated image lacks the fine noise grain visible in the target's dark areas",
          "Blur strength is slightly too aggressive — target retains more edge definition"
        ],
        "discrepancies": [
          "Missing noise component",
          "Blur severity overestimated by ~1 level"
        ],
        "similarity_rating": "moderate"
      },
      "adjustment": "Reduce blur to severity 2, add Gaussian RGB noise starting at severity 1"
    },
    {
      "round": 2,
      "hypothesis": {
        "pipeline": [
          {"function": "blur_gaussian", "severity": 2},
          {"function": "noise_gaussian_RGB", "severity": 1}
        ],
        "reasoning": "Round 1 showed blur was too strong and noise was missing. Reducing blur and adding mild Gaussian noise."
      },
      "comparison": {
        "observations": [
          "Blur level now closely matches the target",
          "Noise grain is present but finer than the target — need higher severity",
          "Noise pattern looks correct (uniform, no spatial correlation)"
        ],
        "discrepancies": [
          "Noise severity too low"
        ],
        "similarity_rating": "good"
      },
      "adjustment": "Keep blur at severity 2, increase noise to severity 2"
    }
  ],
  "final_assessment": {
    "pipeline": [
      {"function": "blur_gaussian", "severity": 2, "actual_params": {"sigma": 2}},
      {"function": "noise_gaussian_RGB", "severity": 2, "actual_params": {"sigma": 0.1}}
    ],
    "confidence_per_degradation": [
      {"function": "blur_gaussian", "confidence": "high", "rationale": "Blur profile matches across all regions"},
      {"function": "noise_gaussian_RGB", "confidence": "medium", "rationale": "Noise type seems right but spatial correlation pattern subtly different"}
    ],
    "remaining_discrepancies": [
      "Slight color cast in target not reproduced — may be from a brightness or saturation shift not yet identified"
    ],
    "alternatives_considered": [
      "blur_lens was considered but ruled out — target blur is uniform across the frame, not radial"
    ]
  }
}
```

The key fields to populate carefully:
- **initial_analysis.suspected_degradations**: List every degradation you suspect, with visual evidence. Be specific about what you see.
- **initial_analysis.ruled_out**: Equally important — what did you look for but not find? This shows thoroughness.
- **iterations[].comparison.observations**: Be concrete and visual. "Missing noise" is better than "looks different." Include spatial context (shadows, edges, flat regions).
- **iterations[].adjustment**: Explain WHY you're changing specific parameters, not just what you're changing.
- **final_assessment.confidence_per_degradation**: Honest self-assessment. Low confidence on a degradation is valuable information for future research.

### analysis.md

Write a brief report covering:
1. What degradations were found and their likely physical causes
2. The iteration history — what changed between each attempt and why
3. Remaining discrepancies, if any, and hypotheses for what might explain them
4. Confidence assessment for each identified distortion

## Important considerations

### Order sensitivity

The order of degradations in the pipeline changes the result. For example:
- `blur → noise` produces clean noise grain on a soft image
- `noise → blur` smears the noise grain, producing a different look
- `noise → compression` loses fine noise to DCT blocking
- `compression → noise` adds noise on top of blocking artifacts

When the target shows both blur and noise, try both orders and compare. The one that looks more natural is usually correct — real camera pipelines tend to add noise before any blur (denoising) step, while display/reproduction pipelines may blur before adding noise.

### Severity beyond the 1–5 range

The x_distortion library only supports severity 1–5. If the target appears to have a degradation stronger than severity 5, you can apply the same distortion twice (e.g., `blur_gaussian:5, blur_gaussian:3` for an effective sigma beyond 5). Note this in params.json.

### Images that combine content differences with degradation

When the target degraded image and the clean source image have **different content** (different scenes, resolutions, or crops), you cannot use pixel-level comparison. This is the hardest case and requires strict discipline about which analysis metrics to trust.

#### Content-independent metrics: Trust these for degradation TYPE

These metrics detect structural anomalies that no natural image exhibits, regardless of content. Use them to **identify which degradations are present**:

| Metric | Detects | Trust Reason |
|--------|---------|--------------|
| `noise.impulse_total_pct` > 0.5% | Impulse noise | Natural images don't have clusters of 0/255 pixels |
| `compression.block_boundary_ratio` > 1.1 | JPEG compression | Natural images don't have 8×8 DCT block boundaries |
| `compression.unique_G` selectively < 256 (R/B full) | JPEG chroma subsampling | G-only reduction = YCrCb JPEG signature |
| `file_info.bytes_per_pixel` < 1.0 | Heavy compression | Uncompressed images always > 1.0 BPP |
| `sharpening.overshoot_ratio` > 0.5 | Oversharpen | Natural edges don't have Laplacian overshoot |
| `multiscale` abrupt spike (>5x jump) | Pixelate | Natural images have smooth multi-scale variance roll-off |
| `frequency` 8-pixel FFT peaks | JPEG compression | DCT quantization creates periodic frequency peaks |

**Note: `h_v_ratio` is NOT in this list.** Natural images have h_v_ratios from 0.5 to 1.5 due to scene content (horizons, buildings). Motion blur detection in cross-image scenarios requires visual inspection of directional smear, not metric thresholds.

#### Content-dependent metrics: DO NOT trust for degradation TYPE

These metrics measure absolute values that vary naturally with image content. **Never conclude a degradation is present based solely on these metrics.** Use them only for severity calibration when the degradation type is already confirmed by content-independent metrics.

| Metric | Why Unreliable |
|--------|---------------|
| `gradient.gradient_magnitude_mean` | Sharp landscape >> soft portrait, regardless of blur |
| `gradient.laplacian_variance` | Textured image >> smooth image, regardless of blur |
| `basic_stats.mean` (per channel) | Scene lighting varies naturally |
| `basic_stats.std` (per channel) | High-contrast scene >> low-contrast scene |
| `noise.flat_region_variance` | **The most dangerous metric.** Textured images have high "flat region" variance without any noise. Never use this alone to claim Gaussian noise. |
| `ycrcb.Y_local_std_median` | Same as above — natural texture ≠ noise |
| `saturation.saturation_mean` | Colorful photo >> grayscale document |
| `radial.gradient_radial_ratio` | Center-composed image >> edge-composed image |
| `gradient.directional_h_v_ratio` | Natural images range 0.5–1.5. Only useful for same-image target vs clean comparison |
| `cross_channel` absolute values | Depends on scene color palette |
| `frequency.hf_lf_ratio` absolute value | Different images have different frequency distributions |

#### The cross-image decision rule

```
For each suspected degradation:
  1. Check content-INDEPENDENT metrics → determine TYPE
  2. If no content-independent metric supports it → RULE IT OUT
  3. Only after type is confirmed → use content-dependent metrics for SEVERITY calibration
  4. Still uncertain? → Simulate and visually compare degradation TEXTURE (not image content)
```

**Example of correct reasoning:**
- "h_v_ratio=0.45 → motion blur confirmed (content-independent). Gradient magnitude is 15.5 — can't determine severity from this alone (content-dependent). Let me try severity 2 and visually compare the blur texture."
- "flat_region_variance=2700 → could be noise OR natural texture. impulse_pct=0.1% → no impulse. block_boundary_ratio=0.64 → no JPEG. Without a content-independent noise metric, I cannot confirm Gaussian noise from numbers alone. Let me visually inspect for grain."

**Example of wrong reasoning (the most common failure):**
- "flat_region_variance is elevated → Gaussian noise present." ← WRONG. The variance comes from image texture, not degradation.

### Handling uncertainty

Some degradations are inherently ambiguous. High-severity JPEG compression can look similar to quantization. Heavy Gaussian blur can mask other degradations. When uncertain between competing hypotheses, try both and let the visual comparison decide. Document the alternatives you considered in params.json notes.
