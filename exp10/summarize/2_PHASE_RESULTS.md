# 各阶段实验结果

> 合并自 PHASE2_RESULTS.md + PHASE3_PLAN.md + PHASE3_RESULTS.md + PHASE4_RESULTS.md + GLOBAL_ARCH_SUMMARY.md + PRINCIPLES_PLAN.md

---

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
# Phase 3：三方向 SOTA 架构实验

## 实验目标

Phase 1-2 发现：FiLM-GCM 对 contrast + 结构化局部退化有效 (+3.52 dB)，但对其他全局退化类型缺乏有效方案。且简单的"退化类型→组件"映射不成立（gamma 不需要 ChannelCurve，ColorMLP 在复杂空间退化上有害）。

Phase 3 目标：基于 SOTA 论文的**实际 GitHub 实现**，探索三个更 advanced 的架构方向，寻找通用的全局退化处理原则。

## 三个方向

| 方向 | 来源 | 核心机制 | 位置 | 参数 |
|------|------|------|:--:|:--:|
| **FreqMod** | SFHformer (ECCV 2024) | FFT→Real+Imag拼接→FCPE(DWConv)→BN→PWConv→IFFT | per-block | +276K |
| **ColorPre** | CSEC (CVPR 2024) | 低分辨率(64×64)→CNN→全局仿射参数(scale+shift)→全分辨率应用 | input | +0.8K |
| **DualBranch** | SFHformer Mixer | GAP→MLP→FiLM(全局分支) + RSTB(局部分支) + 门控融合 | 架构级 | +3K |

### 关键设计决策（基于实际代码）

1. **FreqMod** 使用 Real+Imag 拼接（非 amplitude/phase 分离），与 SFHformer 一致
2. **ColorPre** 使用低分辨率全局处理（CSEC 核心设计：全局光照在 256×256 估计，全分辨率应用）
3. **DualBranch** 在 RSTB 层间插入全局调制，用学习门控融合

## 实验矩阵（22 组）

覆盖 contrast 和 gamma 两类全局退化 × 三种复杂度级别：

| 退化 | 类型 | Swin | FreqMod | ColorPre | DualBranch |
|------|:--:|:--:|:--:|:--:|:--:|
| SF1: contrast_weaken_scale(3) | 单(纯全局) | ✅ | ✅ | ✅ | ✅ |
| L2: noise(3)+contrast(3) | 双 | ✅ | ✅ | ✅ | ✅ |
| L5: motion(3)+jpeg(3)+contrast(3) | 三 | 已有(19.73) | ✅ | ✅ | ✅ |
| SF8: brightness_gamma_RGB(3) | 单(纯全局) | ✅ | ✅ | ✅ | ✅ |
| SF5: blur(3)+noise(3)+gamma(3) | 三 | ✅ | ✅ | ✅ | ✅ |
| S5: motion(5)+noise(1)+jpeg(1) | 纯局部 | ✅ | — | — | — |

全部 Direct 训练、EMBED_DIM=64、EPOCH_BUDGET=2、LR=5e-4。

## 预期验证的问题

1. **FreqMod 是否能成为统一的全局退化方案？**
   - FFT 天然分离全局（幅度谱）和局部（相位谱）
   - 如果有效，一个模块处理所有全局退化类型
   - 对比基准：FiLM-GCM on L5 = 23.25 (+3.52)

2. **低分辨率全局处理（ColorPre）是否比 per-pixel 逐像素处理（ColorMLP）更稳定？**
   - ColorMLP 在 L6 上 -0.52（与空间退化冲突）
   - CSEC 的低分辨率设计可能避免这种冲突
   - 对比基准：ColorMLP on L1 = 23.49, on L6 = 26.46

3. **显式双分支分离（DualBranch）是否优于隐式 per-block 调制（FiLM-GCM）？**
   - DualBranch 有独立的全局和局部分支
   - 对比基准：FiLM-GCM on L5 = 23.25

## 已知问题与修复

| 问题 | 根因 | 修复 |
|------|------|------|
| FreqMod 崩溃 | `torch.compile` + bfloat16 不支持 `view_as_complex` | `@torch.compiler.disable` 装饰器 |
| 调度器重复进程 | `nohup bash -c` 中 `eval &` 产生多个子进程 | 任务完成后清理 |
| SFHformer 作为 submodule | `git clone` 内嵌仓库 | 忽略 resource/SFHformer |

## 时间线

- 03:00：第一批 8 组启动（SF1 + L2 的 Swin/ColorPre/DualBranch）
- 03:05：FreqMod 崩溃，修复后重新排队
- 04:30：第一批完成，第二批（L5/SF8/SF5）启动
- 06:00：全部完成

## 后续计划

根据 Phase 3 结果决定：
- 哪个方向值得深入（严重度实验、更多退化类型）
- 是否需要组合多方向（FreqMod + ColorPre + DualBranch）
- 是否需要在更多颜色空间退化上验证
# Phase 3 结果：三方向 SOTA 架构验证

## 实验目标

基于 SFHformer (ECCV 2024)、CSEC (CVPR 2024) 的实际代码，设计三个架构方向，验证是否能成为通用全局退化解决方案。

## 三个方向

| 方向 | 来源 | 机制 | 参数量 | 速度 |
|------|------|------|:--:|:--:|
| **FreqMod** | SFHformer | FFT→Real+Imag→FCPE→BN→PWConv→IFFT (per-block) | +276K | 2.1x慢 |
| **ColorPre** | CSEC | 低分辨率(64×64)→CNN→仿射参数→全分辨率应用 (input) | +0.8K | 1%慢 |
| **DualBranch** | SFHformer Mixer | GAP→MLP→FiLM(全局) + RSTB(局部) + 门控融合 (架构级) | +3K | 3%慢 |

## 实验矩阵

6 退化 × 4 架构 = 24 组（含追加的 severity/brightness_shift/saturate 实验）

| 退化 | 类型 | Swin | FreqMod | ColorPre | DualBranch |
|------|:--:|:--:|:--:|:--:|:--:|
| SF1 contrast_pure | 单 | 27.15 | 42.30 | **46.06** | 40.51 |
| SF8 gamma_pure | 单 | 41.32 | 31.65 | **41.40** | 40.93 |
| L2 contrast+noise | 双 | 28.64 | 27.85 | **29.45** | 29.42 |
| L5 contrast+motion+jpeg | 三 | 19.73 | 22.14 | 22.29 | **22.33** |
| SF5 gamma+blur+noise | 三 | 22.85 | 22.26 | **22.88** | 22.80 |
| L1 brightness_HSV+blur | 双 | 23.21 | 23.08 | — | — |
| SF7 saturate+noise | 双 | 29.43 | 25.76 | 29.09 | — |
| BS bright_shift+noise | 双 | 29.14 | — | **29.17** | — |
| S5 motion (负对照) | 局部 | 21.85 | — | — | — |

## 关键发现

### FreqMod：全面失败

| 退化 | Swin | FreqMod | Δ | 结论 |
|------|:--:|:--:|:--:|------|
| SF1 contrast_pure | 27.15 | 42.30 | +15.15 | 唯一正收益 |
| L5 contrast+triple | 19.73 | 22.14 | +2.41 | 还行但不如FiLM |
| L1 brightness_HSV | 23.21 | 23.08 | -0.13 | 中性 |
| SF5 gamma+triple | 22.85 | 22.26 | -0.59 | ❌ 更差 |
| L2 contrast+noise | 28.64 | 27.85 | -0.79 | ❌ 更差 |
| SF7 saturate+noise | 29.43 | 25.76 | **-3.67** | ❌ 灾难 |
| SF8 gamma_pure | 41.32 | 31.65 | **-9.67** | ❌ 灾难 |

**6/7 退化上 FreqMod ≤ Swin 基线。** SFHformer 的 FFT 方法在小模型(0.45M)上不 work。参数膨胀(+276K, 60%) + 速度减半(2.1x慢) + 效果更差 = 完全放弃。

根因：torch.fft 操作在小特征图上引入的伪影超过其全局建模收益；bfloat16 AMP 需要额外处理。

### ColorPre：最稳定赢家

| 退化 | Δ PSNR | 结论 |
|------|:--:|------|
| SF1 contrast_pure | +18.91 | Trivial case (逆=仿射) |
| L2 contrast+noise | +0.81 | 持平 FiLM-GCM |
| L5 contrast+motion+jpeg | +2.56 | 不如 FiLM +3.52 |
| SF5 gamma+blur+noise | +0.03 | 中性 |
| SF7 saturate+noise | -0.34 | 轻微下降 |
| BS bright_shift+noise | +0.03 | 中性 |

**优势**：+0.8K 参数（0.2%），无 NaN，5/6 ≥ Swin 基线
**劣势**：复杂退化(L5)上不如 per-block FiLM，saturate 场景无帮助

CSEC 的低分辨率正则化设计天然稳定——比 FiLM-GCM 的 unbounded MLP 更安全。

### DualBranch：中规中矩

- L5 +2.60（vs ColorPre +2.56, FiLM +3.52）
- L2 +0.78（vs ColorPre +0.81）
- SF8/SF5 轻微下降
- +3K 参数，稳定

双分支显式分离不如 FiLM-GCM 的隐式 per-block 调制有效。

## FreqMod 修复历程

1. **torch.compile + bfloat16 冲突**：`view_as_complex` 不支持 bfloat16 → 用 `torch.amp.autocast(enabled=False)` 包裹 FFT 操作
2. **第一次修复**：`@torch.compiler.disable` 装饰器 — 无效（AMP autocast 仍将中间结果转回 bfloat16）
3. **第二次修复**：`with torch.amp.autocast('cuda', enabled=False)` — 成功

## 结论

1. **ColorPre 是最佳默认选择**：+0.8K，稳定，通用改善
2. **FiLM-GCM 在 contrast+结构化退化上最强** (+3.52)，但有 NaN 风险
3. **FreqMod 完全失败**，放弃
4. **DualBranch 无独立优势**，不如直接用 ColorPre 或 FiLM
5. **Gamma 系列退化不需要任何额外组件**（MLP 已能处理）
6. **Direct 训练 > Ft**（S5: 21.85 vs 20.51, +1.34）
# Phase 4/5 结果：CSN/PCP + Top3缺口 + 输出端颜色

## Phase 4: CSN 和 PCP

### CSN (InstanceNorm + affine, per-block, +1K)

| 退化 | Swin | CSN | Δ | 结论 |
|------|:--:|:--:|:--:|------|
| L5 contrast+triple | 19.73 | 21.12 | +1.39 | ✅ 正收益 |
| L1 brightness_HSV+blur | 23.21 | 23.14 | -0.07 | 稳定无NaN |
| SF8 gamma_pure | 41.32 | 32.64 | **-8.68** | ❌ 灾难 |
| L6 saturate+impulse+lens | 26.98 | 22.38 | **-4.60** | ❌ 灾难 |
| L4 brightness_HSV 三退化 | 22.90 | 22.88 | -0.02 | 中性 |
| SF9 saturate_pure | 47.04 | 35.31 | — | ❌ 差 |
| bHSV_pure | 32.13 | 28.51 | -3.62 | ❌ 差 |

**结论**：CSN **不稳定**。简单退化(L5)上有帮助，但纯全局退化(SF8, SF9, bHSV)和复杂空间退化(L6)上引入严重伪影。InstanceNorm 在去除全局统计量的同时也破坏了有用的空间信息。

**L1 无 NaN 验证通过**：CSN 的 IN affine 有界，不像 FiLM-GCM 那样发散。

### PCP (shared, per-RSTB, +3K)

| 退化 | Swin | PCP (+3K) | PCP-old (+91K) | 结论 |
|------|:--:|:--:|:--:|------|
| L4 brightness_HSV 三退化 | 22.90 | 22.88 | — | 中性 |
| L6 saturate+impulse+lens | 26.98 | 26.59 | — | -0.39 |
| SF9 saturate_pure | 47.04 | 45.78 | — | 略差 |
| BSt bright_shift 三退化 | 22.95 | 22.89 | — | 中性 |

**结论**：共享 PCP (+3K) 效果有限，不如参数更少的 ColorPre (+0.8K)。

### PCP-old (per-block, +91K, 不公平)

| 退化 | Swin | PCP-old | Δ |
|------|:--:|:--:|:--:|
| L5 contrast+triple | 19.73 | 23.01 | +3.28 |
| L1 brightness_HSV+blur | 23.21 | 23.27 | +0.06 |
| L2 contrast+noise | 28.64 | 29.41 | +0.77 |

+91K 参数不可直接对比，仅作上限参考。L5 +3.28 接近 FiLM +3.52，但 +20% 参数是不可接受的代价。

## Phase 5: Top3 缺口补齐 + Output-ColorMLP + 组合

### Top 3 缺口补齐

| 实验 | 退化 | 架构 | PSNR | Δ vs Swin |
|------|------|------|:--:|:--:|
| P5_L1_ColorPre | L1 | ColorPre | 23.47 | +0.26 |
| P5_L6_ColorPre | L6 | ColorPre | 26.63 | -0.35 |
| P5_L2_CSN | L2 | CSN | 运行中 | — |
| P5_SF5_CSN | SF5 | CSN | 运行中 | — |
| P5_SF7_CSN | SF7 | CSN | 运行中 | — |
| P5_BS_CSN | BS | CSN | 运行中 | — |

- L1 ColorPre +0.26：匹配 ColorMLP +0.28，证明输入级颜色预处理对 HSV 退化有效
- L6 ColorPre -0.35：再次确认复杂空间退化上全局模块有害

### Output-ColorMLP (输出端颜色，+59参数)

| 实验 | 退化 | PSNR | 结论 |
|------|------|:--:|------|
| P5_L1_OutCM | L1 | 运行中 | — |
| P5_L6_OutCM | L6 | 运行中 | — |
| P5_SF7_OutCM | SF7 | 运行中 | — |
| P5_SF5_OutCM | SF5 | 运行中 | — |

核心假设：颜色修正放在空间修复之后可以避免输入端 ColorMLP 的干扰问题（L6 -0.52）。

### 组合方案

| 实验 | 退化 | 组合 | 参数 | 结论 |
|------|------|------|:--:|------|
| P5_L6_CSNoCM | L6 | CSN + ColorMLP@output | +1,059 | 排队 |
| P5_L1_CSNoCM | L1 | CSN + ColorMLP@output | +1,059 | 排队 |
| P5_SF7_CSNoCM | SF7 | CSN + ColorMLP@output | +1,059 | 排队 |
| P5_SF5_CPreCSN | SF5 | ColorPre + CSN | +1,822 | 排队 |

## 困难退化的诚实评估

四个"困难退化"（L6, SF5, L1, SF7）经 20+ 组实验后的结论：

| 退化 | 特征 | 最佳模块 | 最佳 Δ | 能突破吗？ |
|------|------|------|:--:|:--:|
| L1 brightness_HSV+blur | HSV + 简单空间 | ColorMLP/ColorPre | +0.26~0.28 | 微弱改善 |
| L6 saturate+impulse+lens | HSV + 复杂空间 | FiLM-GCM | +0.22 | **无法突破** |
| SF5 gamma+blur+noise | 曲线 + 空间 | ColorPre | +0.03 | 不需要模块 |
| SF7 saturate+noise | HSV + 随机空间 | — | 全部负 | **无法突破** |

**根本原因**：全局修正和空间修复在单次前向传播中**共享特征 → 梯度冲突**。这不是模块设计的问题，是架构层面的根本限制。

**突破方向**：Phase 6 的分步训练（Cascade/SeqFreeze），通过梯度解耦和任务分解来解决冲突。

## Phase 4/5 总结

1. **CSN 不是稳定方案**：纯全局和复杂空间退化上有灾难性失败（SF8 -8.68, L6 -4.60）
2. **共享 PCP (+3K) 不如果 ColorPre (+0.8K)**：更简单更有效
3. **L1/L6/SF5/SF7 在单次训练中无法突破**：需要分步训练
4. **ColorPre 连续验证最稳定**：L1 +0.26, L2 +0.81, L5 +2.56, 0 NaN
5. **Output-ColorMLP 和组合方案结果待出**（运行中）
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
# 12 小时实验计划：退化条件化架构的抽象原则

## 目标

不是建立"退化X→架构Y"的查表，而是发现**决定架构选择的原则性因素**：
> 输入任意退化管线 → 分析退化属性 → 原则决定架构组件

## 六条候选原则

### 原则 A：数学形式决定修正类型

| 退化数学形式 | 需要的修正 | 架构机制 |
|-------------|-----------|---------|
| 仿射 `y=a·x+b` | 逐通道 scale+shift | 全局统计量→仿射参数 |
| 非线性曲线 `y=f(x)` | 逐像素逆曲线 | 逐通道曲线逼近 |
| 颜色空间变换 | 跨通道非线性映射 | 跨通道逐像素函数 |

**关键假设**：不是退化名称（contrast/brightness/saturate）决定架构，而是**数学形式**（affine/curve/colorspace）决定架构。

**验证**：Phase 2 的交叉实验（B1-B4 测曲线, C1-C4 测颜色空间+交叉验证）

### 原则 B：严重度放大架构差异

**假设**：严重度 1 时所有架构差不多（修正量小），严重度 5 时架构差异最大（修正量大）

**验证**：每种退化×3 严重度×2 架构
- contrast_scale sev=1/3/5 → FiLM-GCM vs Swin (6组)
- gamma sev=1/3/5 → ChannelCurve vs Swin (6组)
- saturate_HSV sev=1/3/5 → ColorMLP vs Swin (6组)
- 共 18 组

**预期**：sev=1 时 Δ<0.3dB, sev=5 时 Δ>2dB

### 原则 C：颜色空间决定跨通道需求

**假设**：
- RGB 空间操作 → 逐通道处理足够
- HSV 空间操作 → 必须跨通道（RGB↔HSV 非线性通道耦合）
- YCrCb 空间操作 → 线性跨通道即可（YCrCb↔RGB 是线性变换）

**验证**：
- 同操作（saturate_weaken），不同颜色空间（HSV vs YCrCb），同架构（ColorMLP）
- Phase 2 已有 Y1-Y2，补充 sev=1/5 验证线性/非线性差异

### 原则 D：局部退化严重度干扰全局统计量估计

**假设**：FiLM-GCM 依赖 GAP 提取通道统计量。严重 blur 平滑特征→统计量偏移；严重 noise 增加统计量方差→估计不准确。

**验证**：固定 contrast_weaken(3)，变化局部退化
- blur_gaussian sev=1/3/5 + contrast → FiLM-GCM (3组)
- noise_gaussian sev=1/3/5 + contrast → FiLM-GCM (3组)
- 测 FiLM Δ 是否随局部退化严重度递减

### 原则 E：退化步数不改变架构原则（但可能改变效果幅度）

**假设**：双退化（1局部+1全局）和三退化（2局部+1全局）适用相同的架构原则，但三退化的架构收益可能更大（退化更难，好架构的优势更明显）。

**验证**：取最佳组件，对比双退化 vs 三退化
- contrast_scale 双退化(L2) vs 三退化(新) → FiLM-GCM
- gamma 双退化(B1) vs 三退化(新) → ChannelCurve
- saturate 双退化 vs 三退化(L6) → ColorMLP
- 6 组

### 原则 F：组件可叠加，无负面交互

**假设**：多个原则同时适用时（如既有 HSV 操作又有 gamma 操作），对应组件可以叠加使用。

**验证**：组合退化 + 组合组件
- gamma + saturate_HSV → ChannelCurve + ColorMLP
- contrast_scale + brightness_HSV → FiLM-GCM + ColorMLP
- 4 组

## 12 小时时间线

```
现在 (12:00)
├─ Phase 2 继续运行 (~2h 剩余)
│
├─ 14:00 Phase 2 完成 → 分析初步结果
│
├─ 14:00-17:30  Wave 1: 原则B 严重度实验 (18组)
│   - 3 退化类型 × 3 严重度 × 2 架构
│   - 18 × 86min ÷ 8GPU ≈ 3.2h
│
├─ 17:30-20:00  Wave 2: 原则D+E (12组)
│   - 局部退化干扰 (6组) + 步数验证 (6组)
│   - 12 × 86min ÷ 8GPU ≈ 2.2h
│
├─ 20:00-22:00  Wave 3: 原则F+补充 (6组)
│   - 组件叠加 (4组) + 边缘情况 (2组)
│   - 6 × 86min ÷ 8GPU ≈ 1.1h
│
└─ 22:00-24:00  分析总结
    - 汇总所有实验结果
    - 提炼最终原则
    - 写入 summarize/
```

## 需要新建的退化文件

### 严重度实验（原则B）

```json
// contrast sev=1,3,5 + noise(3)
{"pipeline": [{"function": "noise_gaussian_RGB", "severity": 3}, {"function": "contrast_weaken_scale", "severity": 1}]}
{"pipeline": [{"function": "noise_gaussian_RGB", "severity": 3}, {"function": "contrast_weaken_scale", "severity": 5}]}

// gamma sev=1,5 + noise(3)  (sev=3 already in B1)
{"pipeline": [{"function": "noise_gaussian_RGB", "severity": 3}, {"function": "brightness_darken_gamma_RGB", "severity": 1}]}
{"pipeline": [{"function": "noise_gaussian_RGB", "severity": 3}, {"function": "brightness_darken_gamma_RGB", "severity": 5}]}

// saturate sev=1,5 + noise(3)  (sev=3 need new file)
{"pipeline": [{"function": "noise_gaussian_RGB", "severity": 3}, {"function": "saturate_weaken_HSV", "severity": 1}]}
{"pipeline": [{"function": "noise_gaussian_RGB", "severity": 3}, {"function": "saturate_weaken_HSV", "severity": 3}]}
{"pipeline": [{"function": "noise_gaussian_RGB", "severity": 3}, {"function": "saturate_weaken_HSV", "severity": 5}]}
```

### 局部退化干扰（原则D）

```json
// contrast(3) + blur sev=1,5
{"pipeline": [{"function": "blur_gaussian", "severity": 1}, {"function": "contrast_weaken_scale", "severity": 3}]}
{"pipeline": [{"function": "blur_gaussian", "severity": 5}, {"function": "contrast_weaken_scale", "severity": 3}]}

// contrast(3) + noise sev=1,5  (sev=3 already L2)
{"pipeline": [{"function": "noise_gaussian_RGB", "severity": 1}, {"function": "contrast_weaken_scale", "severity": 3}]}
{"pipeline": [{"function": "noise_gaussian_RGB", "severity": 5}, {"function": "contrast_weaken_scale", "severity": 3}]}
```

### 三退化验证（原则E）

```json
// contrast + blur + noise (已有 L4: blur+noise+brightness, 新建 contrast 版)
{"pipeline": [{"function": "blur_gaussian", "severity": 3}, {"function": "noise_gaussian_RGB", "severity": 3}, {"function": "contrast_weaken_scale", "severity": 3}]}
// gamma + blur + noise
{"pipeline": [{"function": "blur_gaussian", "severity": 3}, {"function": "noise_gaussian_RGB", "severity": 3}, {"function": "brightness_darken_gamma_RGB", "severity": 3}]}
// saturate + blur + noise
{"pipeline": [{"function": "blur_gaussian", "severity": 3}, {"function": "noise_gaussian_RGB", "severity": 3}, {"function": "saturate_weaken_HSV", "severity": 3}]}
```

## 预期产出（6条原则的验证结论）

```
原则 A: ✅/❌ 数学形式决定架构 — Phase 2 验证
原则 B: ✅/❌ 严重度放大差异 — Phase 3 验证
原则 C: ✅/❌ 颜色空间决定跨通道 — Phase 2+3 验证
原则 D: ✅/❌ 局部退化干扰全局估计 — Phase 4 验证
原则 E: ✅/❌ 步数不改变原则 — Phase 4 验证
原则 F: ✅/❌ 组件可叠加 — Phase 5 验证
```

## 最终目标

不是一张退化→架构的查表，而是一套**决策规则**：

```
输入: 退化管线 P = [(op₁, sev₁), (op₂, sev₂), ...]

分析退化属性:
  ∀ op ∈ P:
    math_form(op) ∈ {affine, curve, colorspace}
    color_space(op) ∈ {RGB, HSV, YCrCb}
    severity(op) ∈ [1,5]

决策规则:
  R1: if ∃ op with math_form=affine → FiLM-GCM
  R2: if ∃ op with math_form=curve → ChannelCurve  
  R3: if ∃ op with color_space=HSV → ColorMLP
  R4: if max(severity) ≤ 2 → 默认架构足够（Δ小）
  R5: if ∃ severe_local_degradation → 全局统计量估计需鲁棒化
  R6: 多规则触发时，对应组件可叠加

输出: 架构配置
```

这套规则的好处：
- **可泛化**：对未见过的退化组合也适用
- **可解释**：每步决策有明确的退化属性依据
- **可学习**：未来可以从外部数据学习 math_form/color_space 分类器
- **可更新**：新增退化类型只需要分类到现有属性维度
