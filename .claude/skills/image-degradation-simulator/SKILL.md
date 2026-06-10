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

### 多退化并行处理

以上规则针对**单个退化**的搜索。多个独立的退化可以并行处理（每个退化一个 Agent），但需控制并行数：

| 并行数 | CPU 负载 | Token 消耗 | 推荐场景 |
|:--:|:--:|:--:|------|
| ≤ 4 | 低 | 可控 | 小批量 |
| 6-8 | 中 | 中等 | **推荐**（匹配 GPU 数量） |
| 12+ | 高 | 大 | 可能 CPU 饱和 |
| 24+ | 极高 | 爆炸 | ❌ 不推荐 |

**批量盲识别建议**：每轮 8 个 Agent 并行，每个负责 1 个退化。48 退化 = 6 轮。

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


## 批量盲识别工作流程 (v2 — 逻辑收敛框架)

> exp14 验证: CI 优化的 v1 流程导致 83% GOOD 误报率。CI=10/10 时函数可以完全错误。
> v2 核心: **分阶段逻辑收敛，CI 降级为辅助，类别对齐强制验证。**

### 核心思想

不从组合空间中盲目搜索。分 6 阶段逐步缩小范围：

```
Phase 1: 类别检测 → 哪些类别存在？
Phase 2: 函数识别 → 每个类别具体是哪个函数？
Phase 3: 顺序确定 → 退化顺序？
Phase 4: 严重度校准 → severity 值？
Phase 5: 交叉验证 → 完整管线能否解释所有指标？
Phase 6: 接受决策 → 多维度判定
```

### Phase 1: 类别检测

**目标**: 确定 blur/noise/compression/global 哪些存在。不涉及具体函数。

#### 1a: 全局退化 (最简单，优先)

ratios_vs_clean 直接反映全局统计量变化：

| 类型 | 检测方法 | 阈值 |
|------|---------|------|
| brightness | per-channel mean ratio | 任一通道偏离 1.0 超过 ±0.1 |
| contrast | std_ratio | < 0.85 或 > 1.15 |
| saturation | saturation_mean_ratio | < 0.7 或 > 1.3 |
| gamma | mean 变但 std 不变 | gamma_HSV/RGB |

注意: 饱和度变化也可能是 JPEG 色度子采样的副作用。如有 compression 证据需交叉验证。

#### 1b: Blur

| 指标 | 强证据 | 中证据 |
|------|:--:|:--:|
| gradient_magnitude_ratio | < 0.5 | 0.5-0.8 |
| laplacian_variance_ratio | < 0.2 | 0.2-0.5 |
| hf_lf_ratio | < 0.3 | 0.3-0.6 |

`blur_present = (强证据≥2) OR (强证据=1 AND 中证据≥2)`

⚠️ **jitter blur 可能增加 gradient** (多重边缘叠加)。motion blur 看 h_v_ratio 与 clean 的变化(>15%)，不是绝对值。

#### 1c: Noise

| 指标 | 证据 |
|------|------|
| flat_region_variance_ratio | > 1.5 |
| impulse_pct net increase (target-clean) | > 0.5% |
| Cr 或 Cb local_std_ratio | > 2.0 |

`noise_present = 任一满足`

⚠️ **必须用 target vs clean 的差值/比值**。clean 本身可能有高 impulse 或 flat_variance。

#### 1d: Compression (区分 JPEG vs JPEG2000)

| 指标 | JPEG | JPEG2000 |
|------|:--:|:--:|
| block_boundary_ratio | **> 1.1** | < 1.1 |
| bytes_per_pixel | < 1.0 | **< 0.5** |
| unique_G vs clean | 显著下降 | 下降或不变 |
| 视觉 ringing | 轻微 | **明显** |

`compression_present = (block_boundary>1.1) OR (bytes_per_pixel<0.8)`

⚠️ **JPEG2000 无 8×8 块**。block_boundary 检查通过 ≠ 无压缩。**bytes_per_pixel 是最可靠指标**。

---

### Phase 2: 函数识别

**目标**: 对每类确定具体函数。用决策树，不盲目测试。

#### 2a: Blur 决策树

```
1. directional_h_v_ratio 与 clean 对比变化 > 15%? → blur_motion
2. gradient_radial_ratio > 1.3? → blur_lens (中心清晰边缘模糊)
3. gradient_magnitude_ratio > 1.2 (梯度增加!)? → blur_jitter (多重边缘)
4. gradient_radial_ratio 1.1-1.3?
     laplacian_ratio < 0.1 → blur_glass
     laplacian_ratio > 0.3 → blur_zoom
5. gradient_radial_ratio < 0.9? → blur_zoom (中心模糊)
6. 默认 → blur_gaussian
```

#### 2b: Noise 决策树

```
1. impulse_pct(net increase) > 0.5%? → noise_impulse
2. Cr_local_std_ratio > 2.0 AND Cb > 2.0?
     Y 也高 → noise_gaussian_RGB
     Y 不高 → noise_gaussian_YCrCb
3. flat_variance_ratio > 3.0?
     局部方差大 → noise_spatially_correlated
     视觉斑点 → noise_speckle
     暗区>亮区 → noise_poisson
     均匀 → noise_gaussian_RGB
```

#### 2c: Compression 决策树

```
block_boundary > 1.1 → compression_jpeg
bytes_per_pixel < 0.8 AND block_boundary < 1.1 → compression_jpeg_2000
否则 → 标记类型不确定
```

#### 2d: 全局退化决策树

```
saturation_ratio < 0.5 → saturate_weaken (HSV vs YCrCb 看 Cr/Cb 变化)
saturation_ratio > 1.5 → saturate_strengthen
mean偏移 + std不变 → brightness_*_gamma_*
mean偏移 + std变化 → brightness_*_shift_*
std变化 + mean不变 → contrast_strengthen/weaken
```

---

### Phase 3: 顺序确定

**退化管线有物理逻辑**。不要测试 N! 种排列。

```
默认顺序: 全局退化 → blur → noise → compression → oversharpen

规则:
  noise + blur 都在: 测试 noise→blur 和 blur→noise
    flat_variance 低于预期 → blur 在 noise 之后(抹平了噪声)
  
  compression 永远在最后 (除非有证据表明不是)
  
  oversharpen + noise: 检查 noise_vs_sharpen
    如果 verdict=oversharpen → oversharpen 确实存在
    如果 verdict=noise → impulse 增加来自噪声,非 oversharpen

候选顺序 ≤ 3 种。apply → compare，选 CI 更高的。
CI 差异 < 2 → 选更符合物理逻辑的。
```

---

### Phase 4: 严重度校准

Severity 只有 5 个离散值。2 轮二分足够：

| 函数 | 校准指标 |
|------|---------|
| blur_* | gradient_magnitude_ratio |
| noise_gaussian | flat_region_variance_ratio |
| noise_impulse | impulse_pct (net increase) |
| compression_jpeg | unique_G |
| oversharpen | overshoot_ratio |

```
1. 初始 severity = 3
2. apply + compare, 检查校准指标
3. 偏离 > 20%: 调整 ±1
4. 最多 2 轮 (不要为 severity 跑 5 轮)
```

---

### Phase 5: 交叉验证

**在 apply+compare 之后，接受之前，必须验证**:

```
□ 每类 evidence 是否被管线中的函数解释?
   blur_evidence=强 但 severity=1 → 不匹配
   noise_evidence=弱 但有 noise 函数 → 可能误引入

□ 是否有指标异常无法被管线解释?
   unique_RGB 显著变化但无 compression → 遗漏
   saturation 显著变化但无全局退化 → 遗漏

□ 管线是否存在逻辑矛盾?
   oversharpen + blur → 矛盾
   同类别出现两次 → 违规

□ 退化顺序是否物理合理?
   compression 是否在最后?
   noise→blur vs blur→noise 是否符合物理逻辑?
```

---

### Phase 6: 接受决策

**CI 降级为辅助参考** (exp14: CI=10/10 函数可完全错误)。

```
维度 A: 类别完整性 (权重最高)
  ✅ Phase1 检测到的类别全在管线中, 无多余
  ⚠️ 缺/多 1 个
  ❌ 缺/多 2+

维度 B: 指标一致性
  ✅ Phase5 交叉验证全部通过
  ⚠️ 1-2 项不通过但可解释
  ❌ 3+ 项不通过

维度 C: CI (辅助, 无否决权)
  CI≥8 + A+B✅ → 支持 GOOD
  CI<7 + A+B✅ → 仍可 GOOD

最终:
  A✅+B✅ → GOOD (不论 CI)
  A✅+B⚠️ → GOOD (注明可疑)
  A⚠️ 或 B❌ → NEEDS_WORK
  A❌ → POOR
```

### 输出格式

predicted_params.json: `{"pipeline":[{function,severity}], "analysis":{"verdict":"GOOD|NEEDS_WORK|POOR","ci_pass_rate":"N/10"}}`

reflection.json 必须记录每阶段推理:
```json
{
  "phase1_category_detection": {"blur": {"present": true, "evidence": [...]}, ...},
  "phase2_function_identification": {"blur": {"decision_tree_path": "...", "selected": "..."}, ...},
  "phase3_order_determination": {"candidates_tested": [...], "selected": "...", "reasoning": "..."},
  "phase4_severity_calibration": [{"function": "...", "rounds": [...]}],
  "phase5_cross_validation": {"checks": [...], "all_passed": true/false},
  "phase6_acceptance_decision": {"dimension_a": "passed", "dimension_b": "passed", "dimension_c": "8/10", "final": "GOOD"}
}
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
