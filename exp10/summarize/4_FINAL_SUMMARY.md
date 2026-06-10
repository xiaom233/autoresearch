# 最终总结

> 合并自 FINAL_SUMMARY.md + CONSOLIDATED_RESULTS.md

---

# 全局退化架构实验：最终总结

## 实验历程

四轮实验，共 60+ 组，系统验证了 8 个架构组件在各类全局退化上的表现。

| Phase | 组件 | 组数 | 核心发现 |
|:--:|------|:--:|------|
| 1 | FiLM-GCM | 10 | L5 contrast+triple **+3.52** dB, L1 brightness_HSV **NaN** |
| 2 | ChannelCurve, ColorMLP | 12 | 通道特性曲线帮助 stretch (+0.38), 不帮助 gamma (0); ColorMLP 修复 L1 NaN (+0.28) 但害 L6 (-0.52) |
| 3 | ColorPre, DualBranch, FreqMod | 30+ | ColorPre 最稳; FreqMod 失败 (SF8 -9.67) |
| 4 | CSN, PCP | 20 (进行中) | CSN 稳定 +1.39 (L5), 0 NaN |

## 架构组件总排名

| 排名 | 组件 | 参数增量 | 速度 | 退化胜场 | 稳定性 | 推荐 |
|:--:|------|:--:|:--:|:--:|:--:|:--:|
| 1 | **FiLM-GCM** | +26K (5.7%) | -6% | L5 **+3.52**, L2 +0.79 | L1 NaN | 有条件推荐 |
| 2 | **ColorPre** | +0.8K (0.2%) | -1% | L2 +0.81, L5 +2.56 | ✅ 全稳定 | ✅ 推荐 |
| 3 | **CSN** | +1K (0.2%) | -0% | L5 +1.39 | ✅ 全稳定 | ✅ 推荐 |
| 4 | DualBranch | +3K (0.7%) | -3% | L5 +2.60 | ✅ 全稳定 | 备选 |
| 5 | ColorMLP | +0.1K | -1% | L1 +0.28 (修复NaN) | L6 -0.52 | 特定场景 |
| 6 | ChannelCurve | +0.1K | -0% | stretch +0.38 | gamma 0 | 有限 |
| 7 | PCP (shared +3K) | +3K (0.7%) | 待测 | 待测 | 待测 | 待验证 |
| — | PCP (per-block +91K) | +91K (20%) | -19% | L5 +3.28 | ✅ | 不公平 |
| — | FreqMod | +276K (61%) | -112% | SF8 **-9.67** | ❌ | ❌ 放弃 |

## 各退化类型最佳方案

| 退化类型 | 最佳方案 | Δ PSNR | 次选 | 备注 |
|------|------|:--:|------|------|
| **contrast + 结构化局部** (L5) | FiLM-GCM | +3.52 | PCP +3.28 (旧) | FiLM 最大收益场景 |
| **contrast + 随机噪声** (L2) | ColorPre | +0.81 | FiLM +0.79 | 两者持平 |
| **contrast_pure** (SF1) | ColorPre | +18.91 | FreqMod +15.15 | Trivial case |
| **brightness_HSV + blur** (L1) | ColorMLP | +0.28 | CSN -0.07 (但稳定) | 无 NaN 是主要目标 |
| **brightness_HSV 三退化** (L4) | FiLM-GCM | -0.06 | — | 都无所谓 |
| **saturate_HSV** | ColorPre | -0.34 | FreqMod -3.67 | 无明显胜者 |
| **gamma 系列** (SF8/SF5/B1) | 无需额外组件 | ~0 | Swin 基线已处理 | MLP 足够 |
| **brightness_shift_RGB** (BS) | ColorPre | +0.03 | — | 太简单，不需要 |

## 五条核心原则

### 原则 1：退化难度决定显式组件的边际价值

不是"退化类型 X → 组件 Y"，而是"当修正复杂度超出网络隐式容量时，显式组件才有价值"。

| 退化难度 | 示例 | 网络能隐式学到? | 显式组件价值 |
|:--:|------|:--:|:--:|
| 低 | gamma, brightness_shift_RGB | ✅ | 0 dB |
| 中 | stretch, contrast_scale | ❌ | +0.4 ~ +0.8 dB |
| 高 | contrast + 结构化局部退化 | ❌ | +2.4 ~ +3.5 dB |

### 原则 2：Per-block 深层次调制 > 输入端浅层处理（复杂退化）

- **简单退化** (SF1, BS): 输入级 ColorPre 足够
- **复杂退化** (L5): per-block FiLM-GCM > input ColorPre (+3.52 vs +2.56)
- 组件放置位置与退化复杂度匹配

### 原则 3：稳定性取决于调制输出是否有界

- **FiLM-GCM**: MLP 输出无界 → L1 NaN (HSV 空间操作 + blur 产生极端统计量 → 发散)
- **CSN**: InstanceNorm affine 有界 → L1 稳定
- **ColorPre**: 低分辨率空间处理作为隐式正则化 → L1 稳定

### 原则 4：参数效率优先

- **+0.1~3K 的组件** (CSN, ColorPre, ChannelCurve, ColorMLP): 受益归因于设计，非参数
- **+26K 的 FiLM-GCM**: 部分受益可能来自参数增量，但 +3.52 dB 远超 5.7% 参数可解释范围
- **+276K 的 FreqMod**: 参数大增但效果差 → 设计失败
- **公平对比**：模块总参数量应与 Swin 基线 (EMBED_DIM=64, 454K) 对齐，而非让基线膨胀匹配模块

### 原则 5：全局退化需要 Direct 训练，Ft 产生负迁移

exp9 验证：Ft (盲预训练→微调) 在 L2 上崩溃 -4.51 dB。全局退化从零训练比盲预训练后微调更好。

### 原则 6：架构模块必须在参数对齐约束下设计 ⚠️ 实验纪律

**模块总参数量必须与 Swin 基线 (EMBED_DIM=64, 454,531) 匹配。** 不能通过增加参数获取不公平优势。

满足条件的模块（Δ ≤ 5K）：
- CSN (+1K), ColorPre (+0.8K), DualBranch (+3K), PCP-shared (+3K), ColorMLP (+0.1K), ChannelCurve (+0.1K)

不满足条件的模块：
- FiLM-GCM (+26K): 需减小 EMBED_DIM 使总参数 ~454K 后重新验证
- PCP per-block (+91K): ❌ 已废弃
- FreqMod (+276K): ❌ 已废弃（且 6/7 退化不如 454K 基线）

**未来实验规则**：
1. 任何新模块的总参数量必须 ≤ 基线 × 1.05（即 ≤ 477K）
2. 如果超出，必须通过减小 EMBED_DIM 使总参数对齐 454K
3. 所有对比实验在相同参数量下进行，不调大基线

## 文档（已合并为 4 个文件）

| 文档 | 内容 |
|------|------|
| [FINAL_SUMMARY.md](FINAL_SUMMARY.md) | 总览：实验历程、组件排名、6 条原则、待完成 |
| [CONSOLIDATED_RESULTS.md](CONSOLIDATED_RESULTS.md) | 全部 130+ 组实验结果（按退化类型组织） |
| [ARCHITECTURE_CATALOG.md](ARCHITECTURE_CATALOG.md) | 所有组件设计、参数、性能对比 |
| [NAS_RESULTS.md](NAS_RESULTS.md) | NAS R1-R3: 注意力/窗口/门控消融 |

### 原则 7：Cascade 分步训练仅在 contrast 退化上有效 🔴 Phase 6 结论

24 组 Cascade 实验结论：
- **CP→Swin 对 contrast 有效**：L2 +0.64, L5 +2.19（ADP 进一步提升至 +1.07）
- **Swin→CP 在所有 spatial Phase A 上灾难**：-2.6~-10.9 dB
- **HSV 退化（L6/L4/L1）cascade 完全失败**：-1.76~-7.12 dB
- **Reversed 退化 cascade 无效果**：+0.03~0.15 dB

结论：Cascade 不是通用方案。仅在 contrast（仿射型）退化 + CP→Swin 顺序下有效。与 ColorPre 的仿射调制原理一致。

## 最终结论

经过 7 轮 130+ 组实验，对全局退化的架构设计形成以下结论：

**可用的方案**：
- **contrast 退化**：FiLM-GCM (+3.52 L5) > Cascade CP→Swin (+2.19) > ColorPre (+0.81)
- **gamma/stretch 退化**：无需额外模块（MLP 已能处理）
- **brightness_shift**：无需额外模块

**无法突破的场景**：
- **HSV 类退化 + 复杂空间**（L6 saturate+impulse+lens）：所有方案 ≤ 0.22 dB
- **HSV 类退化 + 简单空间**（L1 brightness+blur）：ColorMLP +0.28（微弱）

**失败的方案**：
- FreqMod：6/7 退化 ≤ baseline，SF8 -9.67
- CSN：L6 -4.60，纯全局退化灾难
- Cascade Swin→CP：所有 spatial Phase A -2.6~-10.9 dB

**有效的原则**：
1. 退化难度决定组件边际价值
2. Per-block > Input-level（复杂退化）
3. 稳定性 = 调制输出有界
4. 参数效率优先
5. Direct 训练 > Ft（全局退化）
6. 模块参数 ≤ 基线 × 1.05
7. Cascade 仅对 contrast + CP→Swin 顺序有效

## 待完成

1. **Phase 6 分步训练** (Cascade, SeqFreeze, Cascade Order) — 运行中/排队
2. **Output-ColorMLP 和组合方案** — 运行中 (~22%)
3. **CSN 泛化验证** (L2/SF5/SF7/BS) — 运行中 (~23%)
4. **FiLM-GCM 参数对齐**：减小 EMBED_DIM 使总参数 ≤ 477K 后重新验证
5. **严重度效应**：L2 sev=1,3,5 完整对比
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

