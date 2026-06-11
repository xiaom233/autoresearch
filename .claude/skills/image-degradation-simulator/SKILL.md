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

The `--distortions` argument is a comma-separated list of `name:severity` pairs, applied in order.



## 批量盲识别工作流程 (v4.1 — 逐层剥离 + PSNR验证)

> v1-v3.1: CI子指标可捕获耦合效应，但函数区分力不足。
> v4: PSNR像素匹配区分力强，但无法处理多退化耦合。
> v4.1: **CI诊断类型+耦合 → PSNR验证确定性 → 逐层剥离分解多退化。**

### 核心洞察

1. **类型和顺序耦合**：noise→blur 时噪声特征被模糊掩盖，blur→noise 时噪声保持。固定流程失败。
2. **PSNR 只验证确定性退化**：噪声函数有随机性，PSNR 在噪声上天然偏低。**不能用 PSNR 识别噪声类型。**
3. **噪声类型从残差和分布识别**：同图模式下 target-clean 的残差直接暴露噪声特征（见下方噪声分类表）。
4. **逐层剥离**：不是一轮解决所有退化，而是每轮找主导退化→剥离→分析残差。

### 校准阈值速查表 (100图校准, Layer 1)

以下阈值来自 100 张 DIV2K 图像的系统校准。`_r` 后缀表示 target/clean 比值。

#### Blur 子类型识别

| 函数 | gm_r(sev3) | gm_r(sev5) | 唯一信号 |
|------|:--:|:--:|------|
| gaussian | 0.32 | 0.23 | — |
| lens | 0.40 | 0.26 | radial_ratio > 1.3 |
| motion | 0.47 | 0.39 | dir_change > 15% |
| glass | 0.40 | 0.29 | radial 1.1-1.3 |
| zoom | 0.53 | 0.40 | radial < 0.9 |
| **jitter** | **1.29** | **1.60** | **gm_r > 1.0!** (唯一梯度增加的blur) |

检测: blur_present = gm_r < 0.6 OR lap_r < 0.5 (例外: jitter 是 gm_r > 1.0 AND lap_r > 2.0)

#### Noise 子类型识别

| 函数 | imp(sev3) | osr(sev3) | 唯一信号 |
|------|:--:|:--:|------|
| gaussian_RGB | 1799 | 462 | 通道独立, Cr/Y≈1.0 |
| gaussian_YCrCb | 2303 | 710 | Cr/Y_std > 2.0 |
| impulse | 743 | 941 | imp > 500 AND osr > 500 |
| poisson | 534 | 62 | var_mean_ratio > 5 |
| speckle | 255 | 695 | speckle_contrast > 0.01 |
| spatially_corr | 505 | 0 | osr ≈ 0 (唯一!) |

检测: noise_present = fv_r > 1.5 OR imp_net > 0.5% OR Cr/Cb_std_r > 2.0

#### Compression

| 函数 | gm_r(sev3) | uG(sev3) | 唯一信号 |
|------|:--:|:--:|------|
| JPEG | 0.84 | 1× | block_boundary > 1.1 |
| JPEG2000 | 0.44 | 1× | ringing_ratio > 2.0, osc 0.3-0.5 |

检测: JPEG = block > 1.1. JPEG2000 = ringing > 2.0 AND osc 0.3-0.5 (注意: jitter blur 也会触发 ringing > 2.0, 用 osc > 0.5 区分)

#### 全局退化 (全部从 ratios_vs_clean 直接推断)

**brightness**: 8 函数. gamma vs shift: gamma 保持 min=0,max=255; shift 改变极值. RGB vs HSV: per_channel 一致→RGB, 不一致→HSV
**contrast**: scale 精确: std_r = 1 - sev×0.2 (weaken) 或 1 + sev×0.2 (strengthen). stretch 更可变
**saturation**: weaken(sev5) → sat_r = 0.00. strengthen_HSV(sev5) → sat_r = 4.55×. YCrCb → R_mean 不变
**oversharpen**: gm_r > 2.0 AND osr > 50 (校准阈值). 必须同时满足 noise_vs_sharpen=oversharpen
**pixelate**: multiscale 残差 > 5× 相邻 scale. gm_r = 0.5-0.8
**quantization**: uG < 50 → 量化确认

#### 耦合分析 ⚠️ 全局退化信号被局部退化污染

局部退化会改变全局统计量，导致误判：

| 局部退化 | 虚假的全局信号 | 机制 |
|---------|-------------|------|
| blur | std↓ → 像 contrast_weaken | 模糊抹平像素差异 |
| noise | std↑, Cr/Cb↑ → 像 contrast+saturation | 噪声增加方差 |
| compression JPEG | Cr/Cb↓ → 像 saturation_weaken | 色度子采样 |
| oversharpen | mean 偏移, std↑ → 像 brightness+contrast | 锐化改变分布 |

**正确流程**：先剥离局部退化 → 再检查全局信号。

```
Round N-1: 识别并剥离 blur/noise/compression（用 PSNR 验证）
Round N:   对残差检查全局信号
           → 如果剥离后 ratios_vs_clean 恢复正常 → 之前的"全局信号"是耦合假象
           → 如果剥离后仍有明显偏移 → 真正的全局退化
```

**判断方法**：不直接看 target/clean 的 ratios。而是：
1. simulate(clean, 局部退化管线) → 得到 partial
2. 检查 partial vs target 的差异 → 这才是真正的全局退化信号
3. 从 partial→target 的残差中推断全局退化类型和 severity

### v4.1 四轮剥离流程

```
Round 0 — 全局退化优先识别（如果存在）
  全局退化最容易检测且全部确定性 → 优先处理
  方法: ratios_vs_clean 检查 mean/std/saturation 偏移
        → 校准阈值直接推断函数+severity（不枚举），PSNR 最终验证
        → 从 target 中剥离全局退化 → 继续识别剩余

Round 1 — 主导退化识别
  目标: 找到信号最强的退化（通常是效果最明显的那个）
  方法: analyze → 类别evidence → 决策树选函数
        → apply单函数 → CI子指标检查耦合信号
        → 如果是确定性函数 → PSNR验证
        → 确定主导退化的类型和severity

Round 2 — 剥离主导退化，分析残差
  目标: 去除主导退化后，在残差中找次要退化
  方法: simulate(clean, 主导退化) → 与target对比
        → 残差 = target - simulated
        → 对残差重新 analyze → 检测剩余类别
        → 决策树选次要退化函数
        → PSNR验证（确定性部分）

Round 3 — 耦合诊断（最关键！）
  目标: 处理类型和顺序的耦合效应
  方法: CI子指标偏差分析:
        
        耦合信号诊断表:
        overshoot↑+impulse↓+hf_lf↓ → 可能是compression+noise, 不是oversharpen
        impulse↑+hf_lf↑+zero_cross↑ → noise类型/severity错（耦合干扰）
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
| **Gaussian YCrCb** | **Cr/Cb 通道方差 >> Y 通道** | Cr/Cb 正态分布 | Cr_local_std / Y_local_std > 2.0 |
| **Poisson** | **方差 ∝ 强度**（暗区噪声小，亮区大） | 低强度偏斜，高强度近正态 | 强度-方差 Spearman 相关 > 0.5 |
| **Speckle** | **方差 ∝ 强度²**（乘性） | Gamma 分布 | speckle_contrast > 0.01 |
| **Impulse** | 稀疏极端像素 (0 和 255) | 两端尖峰 | impulse_pct(net) > 0.5% |
| **Spatially Correlated** | 邻域像素残差相关 | 空间自相关高 | spatial_cluster_ratio > 0.3 |

#### 判别流程

```
Step 1: 检查 impulse → impulse_pct(net) > 0.5% → noise_impulse
Step 2: 检查 YCrCb → Cr/Y_std > 2.0 → noise_gaussian_YCrCb
Step 3: 检查 Poisson → 强度-方差 正相关 → noise_poisson
Step 4: 检查 Speckle → speckle_contrast > 0.01 → noise_speckle
Step 5: 检查 Spatial → spatial_cluster > 0.3 → noise_spatially_correlated
Step 6: 默认 → noise_gaussian_RGB
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

### 关键原则

1. **PSNR 只验证确定性退化** — 不用于噪声搜索
2. **噪声从残差识别** — PSNR 对噪声无效，残差+分布才是正确方法
3. **CI子指标诊断耦合** — 耦合信号诊断表是处理多退化的核心工具
4. **逐层剥离** — 不在一轮解决所有问题，每轮专注一个退化
5. **顺序从数据推断** — 不预设顺序；耦合诊断表告诉你哪个退化被另一个掩盖

### 与旧版对比

| 问题 | v3.1 | v4 | v4.1 |
|------|------|-----|------|
| 单退化确定函数 | 决策树(~60%) | PSNR(75%) | PSNR验证(75%) |
| **单退化噪声** | 决策树(失败) | PSNR(失败) | **残差+分布判别** |
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
