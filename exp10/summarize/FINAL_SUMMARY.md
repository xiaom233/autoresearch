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
