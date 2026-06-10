---
name: image-degradation-simulator
description: Use this skill whenever the user needs to analyze degraded images, identify distortion types and severity, simulate image degradation using x_distortion algorithms, or reproduce degradation effects on clean images. Trigger when the user mentions image degradation, distortion simulation, x_distortion, degrading clean images, matching image quality degradation, noise/blur/compression analysis, reproducing image distortions, "make this image look like that one", or understanding what degradations are present in an image.
---

# Image Degradation Simulator

Analyze degraded images, identify present distortion types and their severity, then reproduce the degradation on clean images through iterative hypothesis, simulation, visual comparison, and refinement.

**同图模式优先**：实验生成挑战时使用 `--same-image` 标志（`blind_challenge.py`），clean 和 degraded 来自同一原图。这使得像素级校准成为可能，盲识别准确率远高于跨图模式。

## 🚫 绝对禁止（违反立即停止）

以下任何一条都会导致 CPU 100% 持续数小时、结果质量差、实验作废：

1. **不允许写任何 Python 脚本文件**：不要用 Write/Edit 创建 `.py` 文件来做盲识别搜索。只使用已有的 Skill 脚本（`analyze_degradation.py`、`apply_multi.py`、`compare_degradation.py`）。如果需要循环，在 Bash 中手动逐条执行，每次只测 1 个假设。
2. **禁止 `for` 循环遍历退化类型**：看到 `for blur in blurs: for noise in noises: for comp in comps` 立即停止。
3. **禁止 `itertools.permutations` / `itertools.product`**：组合爆炸。
4. **禁止一次性测试 > 5 个假设**：每轮 1 个，基于结果调整下一轮。总共最多 5 轮。
5. **禁止 `run_in_background: true` 启动多个并行搜索**：一次只跑一个模拟，观察结果后再决定下一步。
6. **禁止编写子进程调用脚本**：不要写 Bash 脚本调用 `blind_search2.py 2009 &` 然后 `blind_search2.py 2010 &`。
7. **禁止嵌套循环**：`for perm in itertools.permutations([...])` 绝对禁止。

**正确做法**：每个退化 3-5 轮，每轮 1 个假设。先 `analyze`，形成假设，`apply_multi` 模拟，`compare` 比较，根据结果调整下一轮假设。

## Core principles

### Visual reasoning, not brute-force enumeration

This is the single most important rule. **Never brute-force by testing all distortion types or all severity levels.** The x_distortion library has 35 functions × 5 severity levels = 175+ possible single-distortion combinations, and the combinatorial space explodes for multi-distortion pipelines. Exhaustive search wastes tokens, time, and proves nothing about understanding.

Instead, treat each test as a hypothesis to validate or refute:
- **Look first, simulate second.** Read the target image and describe what you see before writing any code.
- **Test one hypothesis at a time.** Don't generate 30 images in a grid search. Generate ONE image that tests your current best guess, then compare and refine.
- **Rule out categories by visual inspection, not by running them.** You can see that an image has no blocking artifacts without running JPEG compression. You can see it has no salt-and-pepper noise without running impulse noise. Only simulate what you genuinely suspect.
- **Each iteration should be a deliberate adjustment**, not a random probe. **If you find yourself writing a for-loop over all distortion types, stop — you're brute-forcing.**

The difference between a good analysis and a bad one:
- Bad: "Let me test all 35 functions and see which one has the lowest MSE."
- Good: "The image shows uniform isotropic softness with no directional smear, no radial falloff, and no pixel jitter. This rules out motion, lens, zoom, jitter, and glass blur. The edge profiles show smooth Gaussian-like falloff. Hypothesis: `blur_gaussian`. Let me test severity 3 first since the blur is moderate."

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

The `--distortions` argument is a comma-separated list of `name:severity` pairs, applied in order.

## 批量盲识别工作流程

当需要对多个退化图做盲识别时，逐个处理，不要写脚本批量：

```
对每个退化：
  1. Read 目标图 + 参考图（视觉检查）
  2. uv run python analyze_degradation.py --target <d> --clean <c>
  3. ⚠️ 从 ratios_vs_clean 计算类别证据评分 (blur/noise/compression/oversharpen)
  4. 基于 evidence + 视觉观察形成假设（1个管线）
  5. uv run python apply_multi.py --input <c> --distortions "f1:s1,..." --output /tmp/t.png
  6. PYTHONPATH=<project> uv run python compare_degradation.py --target <d> --simulated /tmp/t.png --clean <c>
  7. ⚠️ 保存前质量校验（5项检查，必须执行）
     - 检查管线结构一致性
     - 类别证据评分 + 子类型指纹验证
     - 管线-证据对齐判定
     - 已知误识别模式检查
     - 综合置信度评估 (A+B+C 三维度)
  8. 通过 → 保存到 predicted_params/{id}.json + reflection.json
     不通过 → 标记 NEEDS_WORK 或调整假设回到步骤4
```

⚠️ 步骤 3 和 7 是新增的。**证据评分在假设形成前和保存前各执行一次**，确保假设有数据支撑、保存前再次确认。

### 保存前质量校验 ⚠️ 必须执行

CI pass（compare_degradation.py 的 verdict=GOOD）**不等于盲识别正确**。exp13 数据证明：CI=8/10 时函数正确率仅 56%。保存前必须执行以下检查：

#### 检查 1: 管线结构一致性

```
规则 1.1: blur/noise/compression 三大类别各最多出现一次
         如果同类别出现两次 → 标记 NEEDS_WORK，说明识别结果不合理

规则 1.2: oversharpen 与 blur 不能同时存在
         两者梯度方向相反 (锐化 vs 模糊)，如同时出现 → 识别矛盾

规则 1.3: 管线最多 3 步，最少 1 步
```

#### 检查 2: 类别级证据评分 (Category Evidence Score) ⚠️ 替代简单 CI

**核心思想**：CI 把 10 个指标平等对待是错误的。应该按退化类别分组评估——blur 指标只对 blur 有判断力，noise 指标只对 noise 有判断力。

**证据评分规则**（同图模式下使用 ratios_vs_clean）：

##### Blur 证据 (来自 analyze_degradation.py --clean)

| 指标 | 强证据阈值 | 中等证据 | 无证据 |
|------|:--:|:--:|:--:|
| `gradient_magnitude_ratio` | < 0.5 (丢失>50%) | 0.5-0.8 | > 0.8 |
| `laplacian_variance_ratio` | < 0.2 (丢失>80%) | 0.2-0.5 | > 0.5 |
| `hf_lf_ratio` (target/clean) | < 0.3 | 0.3-0.6 | > 0.6 |

```
blur_evidence = 强 (2+ 指标在强证据区) | 中 (1 指标在强证据区) | 弱 (无指标在强证据区)
```

**如果管线含 blur 但 blur_evidence = 弱** → 误识别，标记 NEEDS_WORK

##### Blur 子类型指纹（验证具体的 blur 函数）

| 子类型 | 必须满足的指标特征 |
|--------|------------------|
| `blur_gaussian` | gradient_radial_ratio ≈ 1.0 (均匀); directional_h_v_ratio ≈ 1.0 (无方向性) |
| `blur_motion` | directional_h_v_ratio 明显偏离 1.0 (>1.3 或 <0.7) |
| `blur_lens` | gradient_radial_ratio 明显偏离 1.0 (>1.2，边缘衰减快于中心) |
| `blur_glass` | gradient_radial_ratio 略偏离 1.0 (1.1-1.3) + hf_lf 中度下降 |
| `blur_zoom` | gradient_radial_ratio 明显偏离 + directional 接近 1.0 |
| `blur_jitter` | gradient_magnitude 局部方差大 (不规则) |

**验证方法**：在 compare_degradation.py 输出中检查对应的 content-dependent 指标。如果 predicted 的 blur 子类型与指标指纹不匹配 → 子类型可能错误。

##### Noise 证据

| 指标 | 强证据阈值 | 中等证据 | 无证据 |
|------|:--:|:--:|:--:|
| `flat_region_variance_ratio` | > 3.0 | 1.5-3.0 | < 1.5 |
| `Cr_local_std_ratio` | > 2.0 | 1.3-2.0 | < 1.3 |
| `Cb_local_std_ratio` | > 2.0 | 1.3-2.0 | < 1.3 |

```
noise_evidence = 强 | 中 | 弱
```

**Noise 子类型指纹**：

| 子类型 | 必须满足的指标特征 |
|--------|------------------|
| `noise_gaussian_RGB` | Cr/Cb_local_std ≈ Y_local_std (三通道均匀) |
| `noise_gaussian_YCrCb` | Cr/Cb_local_std >> Y_local_std (色度噪声远大于亮度) |
| `noise_impulse` | impulse_total_pct > 0.5% (必须); impulse_pct_0 ≈ impulse_pct_255 (椒盐对称) |
| `noise_speckle` | flat_region_variance 中等 + 视觉上的斑点模式 (非均匀分布) |
| `noise_spatially_correlated` | flat_region_variance 高 + 空间上非独立 (邻域像素相关) |
| `noise_poisson` | flat_region_variance 与亮度正相关 (暗区噪声大) |

##### Compression 证据

| 指标 | 强证据阈值 | 中等证据 | 无证据 |
|------|:--:|:--:|:--:|
| `block_boundary_ratio` | > 1.3 | 1.1-1.3 | < 1.1 |
| `unique_G` | < 100 | 100-200 | > 200 |

```
compression_evidence = 强 | 中 | 弱
```

**Compression 子类型指纹**：

| 子类型 | 必须满足的指标特征 |
|--------|------------------|
| `compression_jpeg` | block_boundary_ratio > 1.1 (8×8 块必须); unique_G 显著 < 256 |
| `compression_jpeg_2000` | block_boundary_ratio < 1.1 (无块效应); unique_G 中等下降 + ringing 伪影 |

##### Oversharpen 证据（高风险类别）

| 指标 | 必须满足 | 否则 |
|------|:--:|------|
| `overshoot_ratio` | **> 0.5** | oversharpen 不存在 → **误识别** |
| `zero_crossing_density` (target/clean) | **> 1.5** | 锐化程度不足 |
| 视觉确认 | **边缘光晕肉眼可见** | 无光晕 → 几乎肯定误识别 |

**oversharpen 的严格判定**：3 个条件**全部满足**才接受。
exp13 数据：oversharpen 在 Pred 中出现 4 次，GT 中 0 次。这是最常见的误识别模式。

##### 类别-管线对齐判定

```
1. 统计 evidence 显示哪些类别存在：
   blur_evidence ≥ 中 → blur 类别应出现在管线中
   noise_evidence ≥ 中 → noise 类别应出现在管线中
   compression_evidence ≥ 中 → compression 类别应出现在管线中

2. 对比管线：
   证据显示存在但管线没有 → **漏检**，标记 NEEDS_WORK
   管线有但证据显示不存在 → **误引入**，标记 NEEDS_WORK
   证据和管线一致 → **通过**

3. 特别检查：
   oversharpen 若在管线中 → 必须通过严格判定
   blur_glass 若在管线中 → 验证 gradient_radial_ratio > 1.1
```

#### 检查 3: 已知误识别模式 (来源: exp13 24组数据)

```
模式 1: oversharpen 误引入
  exp13 中 Pred 出现 4 次但 GT 出现 0 次
  → 如果管线含 oversharpen，检查原始图片是否真的有边缘光晕
  → 如果没有肉眼可见的光晕 → 可能是 compression_jpeg 或 blur 被误识别
  → 重新检查 compression 和 blur 类别是否被漏检

模式 2: blur_glass → blur_gaussian / blur_lens 混淆
  exp13 中 blur_glass (GT) 5次，Pred 中全部被替换
  → 如果管线含 blur_gaussian 或 blur_lens，检查图像是否有径向模糊特征
  → radial falloff → blur_lens; 均匀柔化 → blur_gaussian; 不规则扭曲 → blur_glass

模式 3: noise_speckle → noise_gaussian 混淆
  exp13 中 noise_speckle (GT) 2次，Pred 中全部替换
  → 检查噪声是否有空间相关性或斑点模式
  → 如果噪声不是纯随机均匀分布 → 考虑 spatially_correlated 或 speckle

模式 4: noise 与 compression 混淆
  noise_spatially_correlated 在跨图模式常被误判为 compression
  → 检查 block_boundary_ratio: > 1.1 → 有 JPEG 块效应
  → 仅 entropy 高但无块效应 → 可能是噪声
```

#### 检查 4: 已验证的子指标排除规则 ⚠️ 实验验证

以下规则经过 exp13 24 组数据验证，可靠性已知：

##### ✅ 高可靠性：oversharpen 排除

```
oversharpen 在 exp13 中 4 次 Pred 出现，GT 中 0 次（100% 误引入）

排除方法（简单 laplacian 检查）:
  laplacian_neg_pct = 拉普拉斯算子中 < -50 的像素占比
  if laplacian_neg_pct < 2%:
      → oversharpen 几乎不存在（4/4 误识别满足此条件）
      → 如果管线含 oversharpen，标记 NEEDS_WORK，重新检查

  验证: 4/4 误识别 laplacian_neg_pct < 0.5%，GT 中无 oversharpen
```

##### ⚠️ 已知不可靠的指标

```
impulse_total_pct: JPEG2000 和 brightness_darken 产生大量 0/255 像素
                   → 不能可靠区分 impulse noise
                   → 仅当 > 5% 时才有参考价值

block_boundary_ratio: 多种退化 (blur/noise/compression) 都产生类似块效应
                      → 不能可靠区分 JPEG
                      → 仅作辅助参考

PSNR (simulated vs target): add_distortion 使用随机种子
                            → 即使函数正确，像素也无法匹配
                            → 不可用于质量评估！
                            正确替代方案: CI 的纹理级对比
```

#### 检查 5: 视觉确认（同图模式必须）

```
对每一类退化必须有视觉证据:
  blur:  边缘平滑? 有方向性拖影? 有径向衰减? → 确定 blur 子类型
  noise: 噪声均匀? 有色度噪声? 有空间模式? → 确定 noise 子类型  
  compression: 8×8 块边界? ringing? → 确认是 jpeg 还是 jpeg_2000
  oversharpen: 边缘光晕 (halo)? → 否则可能是误识别
  brightness/contrast/saturation: 全局统计量明显偏移? → 量化确认
```

#### 检查 6: 综合置信度评估

**三维度加权判定**（PSNR 已实验验证不可用，已移除）：

```
维度 A: 类别-管线对齐 (检查 2)
维度 B: 子类型指纹验证 (检查 2 子项)
维度 C: 结构一致性 (检查 1 + 检查 3 + 检查 4)

综合判定:
  A+B+C 全部 ✅ → GOOD (高置信)
  A ✅ 但 B 或 C ⚠️ → GOOD (注明可疑点)
  A ⚠️ 或 B ❌ → NEEDS_WORK
  A ❌ → POOR (必须重识别)
  
  ⚠️ 若检查4触发 oversharpen 排除规则 → 至少 NEEDS_WORK，oversharpen 需重新验证
```

#### 校验记录格式

校验结果写入 `reflection.json`：

```json
{
  "validation": {
    "category_alignment": {
      "blur_evidence": "强",
      "noise_evidence": "强", 
      "compression_evidence": "无",
      "pipeline_claims": ["blur", "noise"],
      "missing": [],
      "extra": [],
      "verdict": "passed"
    },
    "subtype_fingerprints": {
      "blur_gaussian": {"gradient_radial": 1.02, "directional": 0.98, "verdict": "passed"},
      "noise_impulse": {"impulse_pct": 3.2, "verdict": "passed"}
    },
    "structure_check": {
      "category_duplicate": false,
      "oversharpen_blur_conflict": false,
      "known_patterns_triggered": [],
      "verdict": "passed"
    },
    "final_verdict": "GOOD",
    "confidence": "high",
    "notes": "PSNR=37.5, 所有检查通过。blur_gaussian 指纹匹配。"
  }
}
```

**同图模式输出格式**：
```json
{"pipeline": [{"function":"...","severity":N}], "analysis": {"verdict":"GOOD|NEEDS_WORK|POOR", "ci_pass_rate":"N/10"}}
```

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
