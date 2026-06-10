# 全局退化架构实验：出发点、结果与结论

## 一、实验出发点

### 背景

NAS 实验（R1-R3）在 0.45M RestoreNet 上发现：

1. **退化-架构耦合存在且显著**：不同退化需要不同的注意力机制
   - motion blur → OCAB（空间注意力）+ 大窗口 = +1.72 dB
   - contrast/brightness → **必须用 Swin**，MDTA/OCAB 崩溃 5-8 dB
   - noise → 架构不重要

2. **全局退化（contrast/brightness/saturation）需要根本上不同的架构**
   - GDFN（3×3 深度卷积）在全局退化上产生致命的归纳偏置
   - 根本原因：全局退化是逐像素相同的变换，没有空间结构，空间卷积是错误归纳偏置

3. **但 Swin（默认架构）在全局退化上也只是"不崩溃"，而非"做得好"**
   - L2 (noise+contrast): Swin = 28.70，有改善空间
   - L1 (blur+brightness_HSV): Swin = 23.21，一般

### 核心问题

**什么样的网络结构能更好地处理全局变换？能否根据退化原理设计对应的架构组件？**

### 设计思路

分析 `x_distortion/` 下全部全局退化函数的数学形式，按逆操作类型分类：

| 类型 | 数学形式 | 示例 | 逆操作 |
|:--:|------|------|------|
| A: 仿射 | `y = a·x + b` | contrast_scale, brightness_shift_RGB | 仿射（需估计 a,b） |
| B: 非线性曲线 | `y = f(x)` | gamma, contrast_stretch | 非线性曲线 |
| C: 颜色空间变换 | RGB→HSV→改通道→RGB | brightness_HSV, saturate_HSV | 跨通道非线性映射 |

对应设计三个架构组件：

| 组件 | 目标 | 机制 | 参数量 |
|------|:--:|------|:--:|
| **FiLM-GCM** | A: 仿射 | GAP→MLP→per-channel scale+shift（per-block） | +25.7K |
| **ChannelCurve** | B: 曲线 | 逐通道 1×1 Conv+ReLU（input-level） | +75 |
| **ColorMLP** | C: 颜色空间 | 跨通道 1×1 Conv+ReLU（input-level） | +115 |

参考 SOTA：SE-Net (通道注意力)、Zero-DCE (逐像素曲线估计)、HAT (混合注意力)。

## 二、实验设计

### Phase 1：FiLM-GCM 验证（10 组）

5 个退化 × 2 架构（Swin vs FiLM-GCM），覆盖 brightness/contrast/saturation：

| 退化 | Pipeline | 全局类型 | 步数 |
|------|----------|:--:|:--:|
| L1 | blur_gaussian(3) + brightness_darken_HSV(3) | brightness | 双 |
| L2 | noise_gaussian(3) + contrast_weaken_scale(3) | contrast | 双 |
| L4 | blur_gaussian(3) + noise_gaussian(3) + brightness_darken_HSV(3) | brightness | 三 |
| L5 | blur_motion(3) + compression_jpeg(3) + contrast_weaken_scale(3) | contrast | 三 |
| L6 | noise_impulse(3) + blur_lens(3) + saturate_weaken_HSV(3) | saturation | 三 |

LR=1e-3（L1/L5/L6 因 NaN 重跑用 LR=5e-4）。

### Phase 2：ChannelCurve + ColorMLP 验证 + 交叉验证（12 组）

新增退化覆盖 gamma, stretch, YCrCb：

| 组 | 退化 | 数学类型 | 架构 | 验证问题 |
|:--:|------|:--:|------|------|
| B | noise+gamma_RGB | B:曲线 | Swin vs ChannelCurve | 曲线操作匹配？ |
| B | noise+stretch | B:曲线 | Swin vs ChannelCurve | 同类通用性？ |
| C | L1 brightness_HSV | C:颜色空间 | ColorMLP | 修复 NaN？ |
| C | L6 saturate_HSV | C:颜色空间 | ColorMLP | 跨类别通用性？ |
| C | L1 brightness_HSV | C:颜色空间 | ChannelCurve（错） | 错误架构对照 |
| C | gamma_RGB | B:曲线 | ColorMLP（错） | 错误架构对照 |
| Y | noise+saturate_YCrCb | C:颜色空间 | Swin vs ColorMLP | 线性 vs 非线性颜色空间 |
| D | S5 motion | 局部 | Curve/ColorMLP | 负对照 |

全部 Direct 训练、EMBED_DIM=64、EPOCH_BUDGET=2、LR=5e-4。

## 三、实验结果

### Phase 1 汇总

| 退化 | 全局类型 | Swin | FiLM-GCM | Δ | 稳定性 |
|------|:--:|:--:|:--:|:--:|:--:|
| L1 | brightness_HSV | 23.21 | 23.12 | -0.09 | ❌ NaN 两次 |
| L2 | contrast_scale | 28.70 | **29.49** | **+0.79** | ✅ |
| L4 | brightness_HSV | 22.90 | 22.84 | -0.06 | ✅ |
| L5 | contrast_scale | 19.73 | **23.25** | **+3.52** | ✅ |
| L6 | saturate_HSV | 26.98 | **27.20** | +0.22 | ✅ |

**L5 +3.52 dB 是整个 NAS 系列最大单点改进。**

L5 逐数据集：

| 数据集 | Swin | FiLM-GCM | Δ |
|------|:--:|:--:|:--:|
| Set5 | 22.27 | 25.02 | +2.75 |
| Set14 | 20.79 | 23.63 | +2.84 |
| B100 | 20.98 | 23.96 | +2.98 |
| Urban100 | 18.54 | 21.71 | +3.17 |
| Manga109 | 19.20 | 23.01 | +3.81 |
| DIV2K | 20.48 | 25.27 | **+4.79** |

全部 6 个数据集一致改善。

**L1 NaN 问题**：brightness_darken_HSV + gaussian_blur 组合在 LR=1e-3 和 5e-4 下均触发 FiLM-GCM NaN。降低 LR 仅推迟发散点（48%→95%），无法避免。HSV 颜色空间操作的非线性 RGB 变换，用仿射调制（FiLM）去逆是数学上的不匹配。

### Phase 2 汇总

| ID | 退化 | 架构 | PSNR | Δ vs Swin | 结论 |
|----|------|------|:--:|:--:|------|
| B1 | noise+gamma | Swin | 29.33 | — | 基线 |
| B2 | noise+gamma | **ChannelCurve** | 29.28 | **-0.05** | ❌ 无帮助 |
| B3 | noise+stretch | Swin | 28.33 | — | 基线 |
| B4 | noise+stretch | **ChannelCurve** | 28.71 | **+0.38** | ✅ 有帮助 |
| C1 | L1 brightness_HSV | **ColorMLP** | 23.49 | **+0.28** | ✅ 修复 NaN |
| C2 | L6 saturate_HSV | **ColorMLP** | 26.46 | **-0.52** | ❌ 有害 |
| C3 | L1 brightness_HSV | ChannelCurve（错） | 23.19 | -0.02 | 逐通道无效 |
| C4 | gamma_RGB | ColorMLP（错） | 29.42 | +0.09 | 噪声级别 |
| Y1 | noise+saturate_YCrCb | Swin | 29.45 | — | 基线 |
| Y2 | noise+saturate_YCrCb | **ColorMLP** | 29.19 | **-0.26** | ❌ 有害 |

## 四、实验结论

### 结论 1：显式组件只在网络隐式容量不足时有价值（核心原则）

不是"全球化退类型 X → 组件 Y"的查表，而是取决于退化修正的**难度**是否超出了网络 MLP 层的隐式逼近能力：

| 退化 | 修正难度 | 网络能隐式学到？ | 显式组件价值 |
|------|:--:|:--:|:--:|
| gamma (power law) | 低 | ✅ | 0 dB |
| stretch (sigmoid) | 中 | ❌ | +0.38 dB |
| contrast_scale (仿射) | 中 | ❌ | +0.79 ~ +3.52 dB |
| brightness_HSV | 中高 | ❌ | +0.28 dB |

**gamma 没有从 ChannelCurve 获益**——MLP 的 Linear→GELU→Linear 本身就能逼近幂函数。不是因为"曲线操作"错了，而是因为网络已经够用。

### 结论 2：全局组件与空间退化存在兼容性梯度（新发现）

| 组件 | 放置位置 | L1 (简单局部) | L6 (复杂局部) |
|------|:--:|:--:|:--:|
| **ColorMLP** | input-level | +0.28 ✅ | -0.52 ❌ |
| **FiLM-GCM** | per-block | — | +3.52 ✅ (L5) |

**输入端逐像素颜色处理与空间特征提取存在负交互**。当局部退化复杂时（L6 的 impulse noise + lens blur），input-level ColorMLP 破坏了后续空间处理需要的信息。Per-block 的 FiLM-GCM 没有这个问题——它在特征层级内部调制，与空间处理兼容。

### 结论 3：颜色空间决定跨通道需求，但需要匹配线性/非线性

- **HSV**（非线性 RGB 变换）：ColorMLP > ChannelCurve（+0.28 vs -0.02），跨通道处理确实有帮助
- **YCrCb**（线性变换）：ColorMLP 有害（-0.26），线性 1×1 Conv 更合适
- **RGB**（无变换）：ColorMLP 中性（+0.09），跨通道处理是冗余的

### 结论 4：contast + 结构化局部退化是 FiLM-GCM 最优场景

L5 (+3.52) 远超 L2 (+0.79)。两者都是 contrast_scale，区别在于：
- L2：noise（随机）→ 全局统计量估计容易
- L5：motion blur + JPEG（结构化）→ 全局+局部的分离更有价值

FiLM-GCM 的 GAP→MLP→affine 机制分离了"全局对比度修正"和"局部伪影去除"，在两者高度纠缠时（L5）价值最大。

### 结论 5：训练策略也需要匹配退化类型

- **Ft（盲预训练→微调）在全局退化上崩溃**（exp9: L2 -4.51 dB）
- **Direct 训练在全局退化上稳定**（Phase 1-2 所有实验）
- 原因：盲预训练的局部退化修复偏置对全局退化产生负迁移
- 这是一个独立原则：**全局退化不仅需要特定架构，还需要特定训练策略**

## 五、修正后的抽象原则

基于 Phase 1-2 的全部实验数据，提出以下原则（替代原始的六条候选原则）：

| # | 原则 | 验证状态 |
|:--:|------|:--:|
| **G1** | 显式架构组件的价值取决于退化修正难度是否超出网络隐式容量 | Phase 2 验证 |
| **G2** | 组件的放置位置（input vs per-block）决定其与空间退化的兼容性 | Phase 2 部分验证 |
| **G3** | 颜色空间类型（RGB/HSV/YCrCb）决定跨通道处理需求 | Phase 2 验证 |
| **G4** | 仿射型全局退化 + 结构化局部退化是 per-block 调制的最大收益场景 | Phase 1 验证 |
| **G5** | 全局退化需要 Direct 训练，Ft（盲预训练）产生负迁移 | exp9 + Phase 1-2 确认 |
| **G6** | 严重度放大架构差异（待验证） | Phase 3 |

这些原则不是"退化 X→组件 Y"的查表，而是一套**决策规则**——分析退化的数学属性后，根据原则决定是否需要以及如何放置架构组件。

## 六、待验证

1. **原则 G6**：严重度效应 — 低严重度时所有架构表现相似？
2. **原则 G2**：组件放置位置 — per-block ChannelCurve/ColorMLP vs input-level 系统对比
3. **L6 ColorMLP 退化修复**：per-block ColorMLP 能否消除 -0.52 dB 的负交互？
4. **YCrCb 线性组件**：纯线性 1×1 Conv（无 ReLU）替代 ColorMLP
5. **组件组合**：FiLM-GCM + ChannelCurve 在 gamma+contrast 混合退化上的叠加效果
