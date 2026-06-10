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
