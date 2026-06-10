# 架构组件目录

## 测试过的所有组件

| 组件 | Phase | 位置 | 参数增量 | 速度 | 状态 |
|------|:--:|:--:|:--:|:--:|:--:|
| **ColorPre** | 3 | input | +0.8K | -1% | ✅ 推荐 |
| **CSN** | 4 | per-block | +1K | -0% | ⚠️ 有条件 |
| **FiLM-GCM** | 1 | per-block | +26K | -6% | ⚠️ 有条件 |
| **ColorMLP** | 2 | input | +0.1K | -1% | ⚠️ 特定场景 |
| **DualBranch** | 3 | 架构级 | +3K | -3% | 备选 |
| ChannelCurve | 2 | input | +0.1K | -0% | 有限 |
| PCP (shared) | 4 | per-RSTB | +3K | +5% | 不如果 |
| PCP (per-block) | 4 | per-block | +91K | +19% | ❌ 不公平 |
| **FreqMod** | 3 | per-block | +276K | +112% | ❌ 放弃 |
| Output-ColorMLP | 5 | output | +59 | -0% | 测试中 |

## 组件设计细节

### ColorPre（最稳定）

```
输入 (B,3,H,W) → AdaptiveAvgPool(64×64) → Conv(3→16,3)
→ ReLU → GAP → Linear(16→16) → ReLU → Linear(16→6)
→ scale/shift → x*(1+scale)+shift
```

- 来源：CSEC (CVPR 2024) 低分辨率全局处理
- 原理：低分辨率空间处理作为隐式正则化
- 插入位置：输入端（像素空间）

### CSN（最轻量）

```
特征 (B,C,H,W) → InstanceNorm2d(C, affine=True)
```

- 来源：风格迁移 IN 设计
- 原理：IN 天然逆 brightness (mean) / contrast (variance)
- 插入位置：每个 SwinBlock FFN 之后

### FiLM-GCM（最强但 NaN 风险）

```
特征 → GAP → Linear(C→C/4) → ReLU → Linear(C/4→2C)
→ scale/shift → x*(1+scale)+shift
```

- 来源：SE-Net, FiLM, HAT
- 原理：全局统计量 (GAP) 预测仿射调制参数
- 插入位置：每个 SwinBlock FFN 之后

### FreqMod（失败）

```
特征 → FFT → [Real+Imag concat] → BN → DWConv(FCPE)
→ PWConv → GELU → PWConv → IFFT → 输出
```

- 来源：SFHformer (ECCV 2024) FourierUnit
- 失败原因：FFT 在小特征图引入伪影；bfloat16 AMP 不兼容；参数膨胀 60%

## 推荐组合

| 场景 | 推荐组件 | 参数 | 理由 |
|------|------|:--:|------|
| 默认安全选择 | ColorPre | +0.8K | 最稳，无 NaN |
| contrast+结构化局部 | FiLM-GCM | +26K | L5 +3.52 |
| contrast (cascade) | CP→Swin ADP | +0.8K | L2 +1.07, L5 +2.19 |
| 需要稳定 + strong | ColorPre + CSN | +1.8K | 覆盖更多退化类型 |

## Cascade 分步训练（Phase 6 新增）

| 方案 | 描述 | 参数 | 有效范围 |
|------|------|:--:|------|
| **Cascade CP→Swin** | Phase A: Swin(spatial-only) → Phase B: ColorPre@input(frozen) | +0.8K | **仅 contrast** |
| Cascade Swin→CP | Phase A: Swin(spatial-only) → Phase B: ColorMLP@output(frozen) | +59 | ❌ 灾难 |
| Cascade ADP | 同 CP→Swin, Phase A 用 ADP 早停 | +0.8K | 进一步提升 |

**关键限制**：Cascade 仅在 contrast（仿射型）退化上有效。HSV 类退化 cascade 完全失败（L6 -7.12）。Swin→CP 顺序在所有 spatial Phase A 上灾难。
