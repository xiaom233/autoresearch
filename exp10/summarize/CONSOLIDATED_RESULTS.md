# 全局退化架构实验：合并结果

## 退化覆盖矩阵

所有退化 × 所有组件的 PSNR 差异（Δ vs Swin 基线）

| 退化 | 类型 | ColorPre | CSN | FiLM-GCM | ColorMLP | FreqMod | 最佳 |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| SF1 contrast_pure | 单 | **+18.91** | — | — | — | +15.15 | ColorPre |
| SF8 gamma_pure | 单 | +0.08 | -8.68 | — | — | -9.67 | ≈Swin |
| SF9 saturate_pure | 单 | — | -11.73* | — | — | — | Swin |
| bHSV_pure | 单 | — | -3.62 | — | — | — | Swin |
| L2 contrast+noise | 双 | +0.81 | 运行中 | +0.79 | — | -0.79 | ColorPre/FiLM |
| L1 brightness_HSV+blur | 双 | +0.26 | -0.07 | NaN | **+0.28** | -0.13 | ColorMLP/ColorPre |
| SF7 saturate+noise | 双 | -0.34 | 运行中 | — | — | -3.67 | ≈Swin |
| BS bright_shift+noise | 双 | +0.03 | 运行中 | — | — | — | ≈Swin |
| B1 noise+gamma | 双 | — | — | — | — | — | ≈Swin |
| B3 noise+stretch | 双 | — | — | — | — | — | Curve +0.38 |
| L5 contrast+motion+jpeg | 三 | +2.56 | +1.39 | **+3.52** | — | +2.41 | FiLM-GCM |
| L4 brightness_HSV 三退化 | 三 | — | -0.02 | -0.06 | — | — | ≈Swin |
| L6 saturate+impulse+lens | 三 | -0.35 | -4.60 | +0.22 | -0.52 | — | FiLM |
| SF5 gamma+blur+noise | 三 | +0.03 | 运行中 | — | — | -0.59 | ≈Swin |

*SF9_CSN = 35.31 vs Swin 47.04

## 按退化类型总结

### Contrast 系列（仿射型）

| 退化 | 有效组件 | 无效组件 |
|------|------|------|
| SF1 pure | ColorPre +18.91 | 过于简单，任何仿射预测都行 |
| L2 + noise | ColorPre +0.81, FiLM +0.79 | FreqMod -0.79 |
| L5 + motion+jpeg | **FiLM +3.52** | — |

**原则**：contrast 是仿射变换，FiLM-GCM 的 GAP→MLP→scale/shift 天然匹配。复杂空间退化时 per-block 调制显著优于 input 级。

### Gamma/Stretch 系列（曲线型）

| 退化 | 有效组件 | 无效组件 |
|------|------|------|
| SF8 pure | 全部 ~0 | FreqMod -9.67 |
| B1 + noise | 全部 ~0 | ChannelCurve -0.05 |
| B3 + stretch | ChannelCurve +0.38 | — |
| SF5 + blur+noise | 全部 ~0 | FreqMod -0.59 |

**原则**：gamma 太简单（MLP 已能学），不需要额外组件。stretch 略有用。纯 gamma 上 FreqMod 灾难。

### HSV/颜色空间系列

| 退化 | 有效组件 | 无效组件 |
|------|------|------|
| L1 + blur | ColorMLP +0.28, ColorPre +0.26 | FiLM NaN, CSN -0.07 |
| L6 + impulse+lens | FiLM +0.22 | **全部负**（CSN -4.60, ColorMLP -0.52） |
| SF7 + noise | 全部负 | FreqMod -3.67, ColorPre -0.34 |
| L4 + blur+noise | 全部 ~0 | — |

**原则**：HSV 退化 + 简单空间 = ColorMLP/ColorPre 略有用。HSV 退化 + 复杂空间 = **任何额外模块都有害**。这是核心瓶颈。

### Brightness_shift 系列（加法型）

| 退化 | 有效组件 |
|------|------|
| BS + noise | 全部 ~0 |
| BSt + blur+noise | 全部 ~0 |

**原则**：brightness_shift 是简单的加法，标准 MLP bias 已能处理。

## FreqMod 失败总结

| 退化 | Δ | 严重度 |
|------|:--:|:--:|
| SF8 gamma_pure | -9.67 | 🔴 灾难 |
| SF7 saturate+noise | -3.67 | 🔴 灾难 |
| L2 contrast+noise | -0.79 | 🟡 差 |
| SF5 gamma+blur+noise | -0.59 | 🟡 差 |
| L1 brightness_HSV | -0.13 | ≈ |
| SF1 contrast_pure | +15.15 | 🟢 好 |
| L5 contrast+triple | +2.41 | 🟢 好 |

**6/7 退化上 ≤ Swin 基线。** 仅在最简单的纯 contrast 和最复杂的三退化上有帮助。+276K 参数 / 2.1x 慢 / 60% 参数膨胀 = 完全放弃。

## 困难退化专项

| 退化 | 特征 | 实验数 | 最佳 Δ | 突破? |
|------|------|:--:|:--:|:--:|
| L1 | HSV + 简单空间 | 8 | +0.28 | 微弱 |
| L6 | HSV + 复杂空间 | 8 | +0.22 | ❌ 无法 |
| SF5 | 曲线 + 复杂空间 | 6 | +0.03 | 不需要 |
| SF7 | HSV + 中等空间 | 5 | -0.34 | ❌ 无法 |

**根因**：单次前向传播中全局修正与空间修复共享特征 → 梯度冲突。Phase 6 分步训练尝试突破。

## 实验统计

| Phase | 组件 | 组数 | 状态 |
|:--:|------|:--:|:--:|
| NAS R1-R3 | Swin/MDTA/OCAB, window, SwiGLU | ~15 | ✅ |
| 1 | FiLM-GCM | 10 | ✅ |
| 2 | ChannelCurve, ColorMLP | 12 | ✅ |
| 3 | ColorPre, DualBranch, FreqMod | 30+ | ✅ |
| 4 | CSN, PCP | 20 | ✅ |
| 5 | Top3缺口, OutputCM, 组合 | 16 | 部分运行 |
| 6 | Cascade, SeqFreeze, Order | 24 | 排队 |
| **合计** | | **130+** | |

## Phase 6: Cascade 分步训练（36 组）

### Spatial Phase A（Swin 仅训练空间退化）

| 退化 | Swin基线 | CP→Swin | Δ | Swin→CP | Δ | 结论 |
|------|:--:|:--:|:--:|:--:|:--:|------|
| L2 noise+contrast | 28.64 | 29.28 | +0.64 | 17.72 | -10.92 | CP→Swin ✅ |
| L2 ADP | 28.64 | 29.71 | +1.07 | 17.83 | -10.81 | ADP 最优 |
| L5 motion+jpeg+contrast | 19.73 | 21.92 | +2.19 | 17.10 | -2.63 | CP→Swin ✅ |
| L6 impulse+lens+saturate | 26.98 | 19.86 | -7.12 | 20.50 | -6.48 | 💀 均灾难 |
| L4 blur+noise+brightness | 22.90 | 21.14 | -1.76 | 21.04 | -1.86 | ❌ 均失败 |
| L1 blur+brightness | 23.21 | 19.68 | -3.53 | 19.87 | -3.34 | ❌ 均失败 |

### Full Phase A（Swin 训练完整退化，reversed 顺序）

| 退化 | Swin | CP→Swin | Swin→CP | Δ 范围 |
|------|:--:|:--:|:--:|:--:|
| SF5_rev | 22.22 | 22.25 | 22.28 | +0.03~0.06 |
| L2_rev | 21.33 | 21.48 | 21.42 | +0.09~0.15 |
| L4_rev | 21.31 | 21.35 | 21.37 | +0.04~0.06 |
| L6_rev | 26.69 | 26.72 | 26.76 | +0.03~0.07 |
| L5_rev | 18.49 | 18.56 | 18.53 | +0.04~0.07 |

### Cascade 结论

1. **Cascade 仅对 contrast（仿射型）退化有效**：L2 +0.64~1.07, L5 +2.19
2. **Swin→CP 在所有 spatial Phase A 上灾难**：-2.6~-10.9 dB
3. **HSV 退化 cascade 完全失败**：L6 -7.12, L1 -3.53
4. **Reversed 退化无 cascade 收益**：+0.03~0.15
5. **ADP 进一步提升**：L2 ADP +1.07 vs Fixed +0.64

## 困难退化最终结论

| 退化 | 实验数 | 最佳 Δ | 根因 |
|------|:--:|:--:|------|
| L1 brightness_HSV+blur | 10 | +0.28 (ColorMLP) | HSV 非线性, 简单空间允许共存 |
| L6 saturate+impulse+lens | 12 | +0.22 (FiLM) | HSV + 复杂空间 = 全局/局部冲突 |
| SF5 gamma+blur+noise | 8 | +0.08 (CSN) | gamma 太简单, MLP 已能处理 |
| SF7 saturate+noise | 8 | +0.12 (OutputCM) | 同上, saturate+noise 冲突 |

