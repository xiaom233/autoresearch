# Phase 2 结果：退化数学分类验证

## 实验矩阵与结果

| ID | 退化 | 数学类型 | 颜色空间 | 架构 | PSNR | Δ vs Swin |
|----|------|:--:|:--:|------|:--:|:--:|
| B1 | noise+gamma_RGB | 曲线 | RGB | Swin | 29.33 | — |
| B2 | noise+gamma_RGB | 曲线 | RGB | **ChannelCurve** | 29.28 | **-0.05** |
| B3 | noise+stretch | 曲线 | RGB | Swin | 28.33 | — |
| B4 | noise+stretch | 曲线 | RGB | **ChannelCurve** | 28.71 | **+0.38** |
| C1 | L1: blur+brightness_HSV | 颜色空间 | HSV | **ColorMLP** | 23.49 | **+0.28** |
| C2 | L6: impulse+lens+saturate_HSV | 颜色空间 | HSV | **ColorMLP** | 26.46 | **-0.52** |
| C3 | L1: blur+brightness_HSV | 颜色空间 | HSV | ChannelCurve (错) | 23.19 | -0.02 |
| C4 | noise+gamma_RGB | 曲线 | RGB | ColorMLP (错) | 29.42 | +0.09 |
| Y1 | noise+saturate_YCrCb | 颜色空间 | YCrCb | Swin | 29.45 | — |
| Y2 | noise+saturate_YCrCb | 颜色空间 | YCrCb | **ColorMLP** | 29.19 | **-0.26** |
| D1 | S5: motion(5)+noise(1)+jpeg(1) | 局部 | — | ChannelCurve | 21.94 | +1.43* |
| D2 | S5: motion(5)+noise(1)+jpeg(1) | 局部 | — | ColorMLP | 21.76 | +1.25* |

*D1/D2 对比的是 NAS R1 的 S5 Swin (Ft, LR=1e-3) = 20.51，不是同条件下的 Direct 基线。LR 和策略都不同，Δ 不可直接解读。

## 逐组分析

### 组 B：非线性曲线（gamma vs stretch）

**gamma (B1 vs B2)**：ChannelCurve **无帮助** (-0.05 dB)。

逐数据集对比：

| 数据集 | Swin | ChannelCurve | Δ |
|------|:--:|:--:|:--:|
| Set5 | 29.80 | 29.76 | -0.04 |
| Set14 | 28.61 | 28.62 | +0.01 |
| B100 | 28.58 | 28.52 | -0.06 |
| Urban100 | 28.58 | 28.56 | -0.02 |
| Manga109 | 30.27 | 30.23 | -0.04 |
| DIV2K | 30.35 | 30.23 | -0.12 |

全部数据集上差异 ≤ 0.12 dB，属于噪声级别。

**原因分析**：gamma 校正 `y = x^γ` 是简单的逐通道幂函数。网络的 MLP 层（Linear→GELU→Linear）本身就是通用函数逼近器，学逆幂函数 `y^(1/γ)` 绰绰有余。ChannelCurve 的逐通道 1×1 Conv 曲线是冗余的。

**stretch (B3 vs B4)**：ChannelCurve **有帮助** (+0.38 dB)。

stretch = sigmoid-like 对比度压缩 `y = 1/(1+(μ/x)^c)`。这是比 power law 更复杂的非线性——网络内部的 MLP 需要更多容量才能逼近。显式的逐通道曲线提供了额外的建模能力。

**关键洞察**：不是"数学形式=曲线→需要 ChannelCurve"，而是"**当修正复杂度超过网络隐式容量时，显式组件才有价值**"。gamma 在 MLP 的容量内，stretch 超出了。

### 组 C：颜色空间（HSV）

**C1 (L1 + ColorMLP)**：**成功修复 NaN + 改善** (+0.28 dB)

L1 = blur_gaussian(3) + brightness_darken_HSV(3)。之前 FiLM-GCM 在这个退化上两次触发 NaN（LR=1e-3 和 5e-4 均失败）。ColorMLP 不仅没有 NaN，还比 Swin 基线高 0.28 dB。

逐数据集（vs G1r Swin 基线 23.21）：

| 数据集 | Swin | ColorMLP | Δ |
|------|:--:|:--:|:--:|
| Set5 | 24.87 | 25.27 | +0.40 |
| Set14 | 23.52 | 23.78 | +0.26 |
| B100 | 23.83 | 24.00 | +0.17 |
| Urban100 | 21.38 | 21.62 | +0.24 |
| Manga109 | 23.63 | 24.01 | +0.38 |
| DIV2K | 24.74 | 24.97 | +0.23 |

全部数据集正收益。跨通道逐像素 MLP 可以学 RGB↔HSV 的逆映射。

**C2 (L6 + ColorMLP)**：**反而有害** (-0.52 dB)

L6 = noise_impulse(3) + blur_lens(3) + saturate_weaken_HSV(3)。ColorMLP 比 Swin 基线低 0.52 dB。

逐数据集（vs G9r Swin 基线 26.98）：

| 数据集 | Swin | ColorMLP | Δ |
|------|:--:|:--:|:--:|
| Set5 | 28.61 | 28.37 | -0.24 |
| Set14 | 26.94 | 26.32 | -0.62 |
| B100 | 26.82 | 26.58 | -0.24 |
| Urban100 | 25.32 | 24.78 | -0.54 |
| Manga109 | 27.52 | 26.97 | -0.55 |
| DIV2K | 28.45 | 28.40 | -0.05 |

**原因分析**：L6 的 impulse_noise 和 lens_blur 是复杂的**空间**退化。ColorMLP 在像素空间做颜色校正，可能干扰了后续的空间特征提取。当全局退化和复杂空间退化混合时，输入端逐像素颜色处理反而有害。

**C3 vs C1（错误架构交叉验证）**：
- ColorMLP (正确，跨通道): +0.28
- ChannelCurve (错误，逐通道): -0.02
- 跨通道 > 逐通道，验证了 HSV 需要跨通道处理的原则

**C4 (gamma + ColorMLP，错误架构)**：意外表现最好 (+0.09)，但差异在噪声范围内。可能只是随机波动。

### 组 Y：YCrCb

**Y1-Y2**：ColorMLP 在 YCrCb 上**有害** (-0.26 dB)。

YCrCb↔RGB 是**线性**变换（矩阵乘法）。saturate_YCrCb 的逆操作是一个 3×3 线性矩阵——1×1 Conv（无激活函数）可以精确学习。ColorMLP 的非线性（ReLU）反而引入了不必要的复杂度。

**启示**：线性颜色空间（YCrCb）只需要线性通道混合，非线性 MLP 是过度设计。

### 组 D：负对照

D1/D2 与基线比较有提升，但这可能来自 Direct vs Ft 的策略差异和 LR 差异，不能归因于组件本身。**需要同条件的 Swin Direct 基线才能正确判断。**

## 原则修正

Phase 2 的结果不支持最初简单的"数学形式→组件"映射。修正后的原则：

### 原则 A（修正）：退化难度决定显式组件的边际价值

不是"数学形式=曲线→需要 ChannelCurve"，而是：

| 退化难度 | 示例 | 网络隐式容量 | 显式组件价值 |
|:--:|------|:--:|:--:|
| 简单 | gamma (power law) | 足够 | 无 (0 dB) |
| 中等 | stretch (sigmoid) | 不足 | 有 (+0.38) |
| 中等 | contrast_scale (affine) | 不足 | 有 (+0.79 ~ +3.52) |
| 复杂 | HSV + 简单局部退化 | 不足 | 有 (+0.28) |
| 复杂 | HSV + 复杂局部退化 | — | **有害** (-0.52) |

**核心洞察**：显式组件只有在网络的隐式容量不够时才产生价值。如果 MLP 层已经能学到修正，显式组件是冗余的（甚至可能干扰优化）。

### 原则 C（修正）：颜色空间 × 局部退化复杂度决定跨通道需求

- HSV + 简单局部退化 → ColorMLP 有帮助 (+0.28)
- HSV + 复杂局部退化 → ColorMLP 有害 (-0.52)
- YCrCb (线性颜色空间) → ColorMLP 有害 (-0.26)，线性通道混合更合适

**核心洞察**：跨通道逐像素处理与空间特征提取之间存在**干扰**。当局部退化复杂时，输入端颜色处理会破坏对空间修复有用的特征。

### 新发现的原则 G：组件与空间退化的兼容性

全局修正组件（输入端逐像素）可能与空间退化（blur/noise）的处理产生负交互：
- FiLM-GCM（per-block）：深层特征调制，与空间处理共存良好（L5 +3.52）
- ColorMLP（input）：像素级颜色修正，干扰空间特征提取（L6 -0.52）
- ChannelCurve（input）：像素级曲线，中性影响

**假说**：组件放置位置（input vs per-block）是一个关键设计维度。Input-level 组件与空间退化的兼容性更差。

## 下一步实验方向

基于 Phase 2 的发现，后续实验应聚焦：

1. **原则 B**：严重度效应 — 组件价值是否在高严重度时更大？
2. **原则 G**：组件放置位置 — per-block vs input-level 的系统对比
3. **退化复杂度量化**：什么构成"简单"vs"复杂"退化？需要系统验证
4. **修复 L6 的 ColorMLP 退化**：是否可以通过调整放置位置来修复？
5. **YCrCb 专用组件**：纯线性 1×1 Conv（无激活）替代 ColorMLP
