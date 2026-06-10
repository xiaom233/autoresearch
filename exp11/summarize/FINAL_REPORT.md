# IRAgent 全局退化架构与反思机制 — 最终报告

## 项目目标

> 输入任意退化管线 → 输出量身定制的训练策略 + 模型架构

## 实验规模总览

| 实验 | 内容 | 规模 |
|------|------|:--:|
| exp9 | 训练策略（Ft/Curric/Direct） | 450+ 组 |
| exp10 | 架构设计（8 组件 × 多退化） | 130+ 组 |
| exp11 | 交叉测试 + 反思验证 | 1284 组 |
| **合计** | | **~1864 组** |

## exp10 架构结论

### 可用方案

| 退化类型 | 最佳架构 | 最大 Δ | 参数 |
|------|------|:--:|:--:|
| contrast + 结构化局部 | **FiLM-GCM** (per-block) | +3.52 dB | +26K |
| contrast + 随机噪声 | ColorPre (input) | +0.81 dB | +0.8K |
| contrast (cascade) | CP→Swin ADP | +1.07 dB | +0.8K |
| brightness_HSV + 简单空间 | ColorMLP | +0.28 dB | +0.1K |
| gamma 系列 | **无需额外组件** | ~0 dB | — |

### 不可用方案

| 方案 | 原因 |
|------|------|
| **FreqMod** | 6/7 退化 ≤ baseline，SF8 -9.67 dB |
| **CSN** | L6 -4.60，纯全局退化灾难 |
| **Cascade Swin→CP** | 所有 spatial Phase A -2.6~-10.9 dB |
| **FiLM-GCM@HSV** | L1 NaN（brightness_HSV 触发发散） |

### 7 条架构原则

1. **退化难度决定组件价值**（gamma=0, contrast+structured=+3.52）
2. **Per-block > Input**（复杂退化需深层次调制）
3. **稳定性 = 输出有界**（IN/low-res > unbounded MLP）
4. **参数效率优先**（+0.8K ColorPre 不输 +26K FiLM）
5. **Direct > Ft**（全局退化禁盲预训练）
6. **参数对齐**（模块 ≤ 基线 × 1.05）
7. **Cascade 仅对 contrast + CP→Swin 有效**

## exp11 反思机制结论

### 数据规模

| 阶段 | 测试数 | 图像 |
|------|:--:|:--:|
| v3 交叉测试 | 839 | 839 tar |
| Round1 探索 | 8 | 0 |
| phaseE 扩展 | 308 | 308 tar |
| Severity 矩阵 | 104 | 0 |
| **合计** | **1284** | **1147 tar** |

### 验证的诊断规则

| 规则 | 方法 | Precision | Recall |
|------|------|:--:|:--:|
| **R1** | PSNR < 自我基线 - 6dB | 90% | — |
| **R2** | edge > 2σ | **97%** | 50% |
| **R3** | chroma > 2σ | **98%** | 50% |
| **R4** | 图像 entropy > 2.0 + color_asym > 0.08 | — | — |
| **R5** | 严重度偏差 ±1 PSNR gap ≥ 10 dB | — | — |

### 修正流程图

```
M_spec PSNR < M_blind - 6dB?
  ├─ edge > 0.10 + chroma > 0.03 → 高置信误识别
  │   ├─ entropy > 2.0 + color_asym > 0.08 → 类型误识别 → 重诊断
  │   ├─ entropy > 2.0 + saturation < 0.15 → gamma 误判
  │   └─ edge_frac < 0.13 → 严重度偏差 → ±1 调整
  └─ 否 → 识别基本正确
修正后重训练 → 预期 PSNR 提升 ≥ 3 dB
```

### 新发现

- **纯全局模型极易检测**：SF1(contrast)/SF8(gamma) 在复杂退化上崩溃 10-22 dB
- **严重度可检测**：±1 sev 的交叉测试 gap ≥ 10 dB（但自测试中不可检测）
- **图像特征区分误差子类型**：entropy +203% 最敏感，color_asymmetry 区分 type vs order
- **架构鲁棒性排序**：DualBranch > CSN > Swin > FiLM > FreqMod > ColorPre
- **ColorPre 悖论**：在自己退化上最强，跨退化最差（高度特化）

## 代码资产

| 文件 | 用途 |
|------|------|
| `REFLECTION_MECHANISM.md` | 修订后的反思机制（含 5 条实验规则） |
| `exp10/summarize/` | 架构实验完整总结 |
| `exp11/scripts/cross_test.py` | 通用交叉测试脚本 |
| `exp11/summarize/` | 反思实验完整总结 |
| `train.py` | 支持 15+ 架构模块（含 LOAD_CKPT/FREEZE_BODY） |
