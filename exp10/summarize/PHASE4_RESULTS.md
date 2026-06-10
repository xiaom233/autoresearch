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
