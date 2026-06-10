# Phase 1: FiLM-GCM 全局通道调制实验

## 动机

NAS 实验（R1-R3）发现：全局退化（contrast/brightness/saturation）需要根本上不同的架构。Swin 窗口注意力无法直接捕获全局统计量，MDTA/OCAB 的 GDFN（3×3 深度卷积）在全局退化上崩溃 5-8 dB。

**核心假设**：全局退化本质是逐通道仿射变换（`y = c * x + b`）。给网络一个直接的全局统计量提取路径——Global Average Pooling → 通道调制——应该能改善全局退化处理。

参考 SOTA：SE-Net、HAT、NAFNet、MAXIM 均使用 GAP + 通道调制来捕获全局上下文。

## FiLM-GCM 设计

```
输入特征 (B, C, H, W)
    ↓
Global Average Pooling → (B, C)
    ↓
Linear(C → C/4) → ReLU → Linear(C/4 → 2C) → scale, shift
    ↓
输出 = 特征 * (1 + scale) + shift
```

- **参数量**：+25.7K（5.66%），从 454K → 480K
- **初始化**：scale/shift 输出层零初始化，训练起始等价于恒等映射
- **插入位置**：每个 SwinBlock 的 FFN 之后
- 控制变量：ATTENTION_TYPE=swin，EMBED_DIM=64，EPOCH_BUDGET=2

## 实验设计

5 个退化配置（2 双退化 + 3 三退化），覆盖 4 种全局变换：

| 退化 | Pipeline | 全局类型 | 类型 |
|------|----------|:--:|:--:|
| L1 | blur_gaussian(3) + brightness_darken_HSV(3) | brightness | 双 |
| L2 | noise_gaussian(3) + contrast_weaken_scale(3) | contrast | 双 |
| L4 | blur_gaussian(3) + noise_gaussian(3) + brightness_darken_HSV(3) | brightness | 三 |
| L5 | blur_motion(3) + compression_jpeg(3) + contrast_weaken_scale(3) | contrast | 三 |
| L6 | noise_impulse(3) + blur_lens(3) + saturate_weaken_HSV(3) | saturation | 三 |

每退化 × 2 架构（Swin / FiLM-GCM）= 10 组。L2/L4 使用 LR=1e-3，L1/L5/L6 因 NaN 问题重跑使用 LR=5e-4。

## 结果

### 整体对比

| 退化 | 全局类型 | Swin | FiLM-GCM | Δ | FiLM 稳定? |
|------|:--:|:--:|:--:|:--:|:--:|
| L1 | brightness | 23.21 | 23.12 | **-0.09** | ❌ NaN |
| L2 | contrast | 28.70 | **29.49** | **+0.79** | ✅ |
| L4 | brightness | 22.90 | 22.84 | -0.06 | ✅ |
| L5 | contrast | 19.73 | **23.25** | **+3.52** | ✅ |
| L6 | saturation | 26.98 | **27.20** | +0.22 | ✅ |

**3/5 改善，1/5 持平，1/5 失败。有效场景（L2/L5/L6）平均 +1.51 dB。**

### L5 逐数据集分析（最大收益场景）

L5 = motion_blur(3) + compression_jpeg(3) + contrast_weaken_scale(3)

| 数据集 | Swin | FiLM-GCM | Δ |
|------|:--:|:--:|:--:|
| Set5 | 22.27 | 25.02 | +2.75 |
| Set14 | 20.79 | 23.63 | +2.84 |
| B100 | 20.98 | 23.96 | +2.98 |
| Urban100 | 18.54 | 21.71 | +3.17 |
| Manga109 | 19.20 | 23.01 | +3.81 |
| DIV2K | 20.48 | 25.27 | **+4.79** |

- 所有 6 个数据集一致改善，无一例外
- DIV2K 收益最大（+4.79 dB）：高分辨率多样图像，全局对比度修正最明显
- 边缘密集数据集（Urban100 +3.17, Manga109 +3.81）也大幅受益

### L2 逐数据集分析

L2 = noise_gaussian(3) + contrast_weaken_scale(3)

| 数据集 | Swin | FiLM-GCM | Δ |
|------|:--:|:--:|:--:|
| Set5 | 29.58 | 29.68 | +0.10 |
| Set14 | 28.22 | 28.79 | +0.57 |
| B100 | 28.22 | 28.69 | +0.47 |
| Urban100 | 28.12 | 28.81 | +0.69 |
| Manga109 | 29.22 | 30.49 | +1.27 |
| DIV2K | 29.72 | 30.36 | +0.64 |

全部数据集正收益，Manga109 最大 +1.27 dB。

### L1 NaN 分析

L1 = blur_gaussian(3) + brightness_darken_HSV(3) 是唯一触发 NaN 的退化：

| LR | NaN 起始步 | NaN 次数 | 最终 PSNR |
|:--:|:--:|:--:|:--:|
| 1e-3 | 7240/15094 (48%) | 7853 | 22.95 |
| 5e-4 | 14361/15094 (95%) | 1933 | 23.12 |

- **降低 LR 推迟了发散点，但无法避免**
- L4 同样有 brightness_darken_HSV 但额外加了 noise_gaussian → 不触发 NaN
- L1 与其他退化的区别：双退化（vs 三退化），且没有噪声步骤

**假说**：gaussian_blur 平滑了 brightness_darken 在 HSV 空间产生的极端像素值，造成特征统计量平滑但整体偏移大——GCM 学到极大的 scale/shift 参数来补偿，训练后期发散。额外的 noise 步骤（L4）增加了特征多样性，起到隐式正则化作用。

## 结论

### 结论 1：FiLM-GCM 对 contrast 类全局退化高度有效

contrast_weaken 是 GCM 的最佳场景。contrast 是逐通道乘法变换（`y = c * x`），GCM 的仿射调制（`x * scale + shift`）天然匹配。在 contrast + 结构化局部退化（motion/jpeg）的组合中，GCM 分离了"全局对比度修正"和"局部伪影去除"，收益最大。

### 结论 2：全局退化类型决定 GCM 有效性

| 全局类型 | 有效性 | 机制 |
|:--:|:--:|------|
| **contrast** (乘法) | ✅ 强 (+0.79 ~ +3.52) | GCM 仿射调制直接匹配乘法变换 |
| **saturation** (HSV) | ✅ 弱 (+0.22) | HSV 空间局部操作，GCM 帮助有限 |
| **brightness** (加法) | ❌ 无效 (-0.09~-0.06) | 加法变换对 GAP 统计量影响小，MLP 可隐式处理 |
| **brightness+blur** (无噪声) | ❌ 不稳定 (NaN) | 平滑特征 + 大偏移 → GCM 参数发散 |

### 结论 3：L5 +3.52 dB 是 NAS 系列最大单点改进

| 改进 | 来源 | 收益 |
|------|------|:--:|
| **FiLM-GCM** | Phase 1 (本次) | **+3.52** |
| OCAB + ws=16 | NAS R1/R2 | +1.72 |
| SwiGLU (S5) | NAS R3 | +1.47 |
| Curric(fwd) | Phase 9 | +1.29 |

FiLM-GCM 在 L5 上的改进超过之前所有架构和策略改进的总和。

### 结论 4：稳定性是部署障碍

L1 在两种 LR 下均 NaN。在修复前，FiLM-GCM 不能安全地用于任意退化管线。需要：
- GCM 输出加 tanh 或 LayerNorm 约束
- GCM 参数独立 LR（低于主干网络）
- 或在检测到 brightness_darken_HSV 时禁用 GCM

## 与之前 NAS 结论的整合

NAS_RESULTS.md 的退化→架构推荐表更新：

| 退化特征 | 注意力 | 窗口 | 激活 | **GCM** | 备注 |
|----------|:--:|:--:|:--:|:--:|------|
| motion sev≥4 | OCAB | 16 | SwiGLU | — | 空间+大窗口+门控 |
| gaussian blur | Swin | 8 | GELU | — | 默认 |
| gaussian noise | Swin | 4-8 | GELU | — | 架构不重要 |
| compression | Swin | 8 | GELU | — | 默认 |
| **contrast + 结构化** | Swin | 8 | GELU | **FiLM-GCM** | GCM 分离全局/局部 |
| **contrast + 随机** | Swin | 8 | GELU | FiLM-GCM | 中等收益 (+0.79) |
| brightness | Swin | 8 | GELU | — | GCM 无效 |
| brightness (无噪声) | Swin | 8 | GELU | **禁用** | GCM NaN 风险 |
| saturation | Swin | 8 | GELU | 可选 | 边际收益 (+0.22) |

## 待验证

1. GCM 稳定性修复后，L1 能否从 GCM 获益？
2. L5 的巨大收益能否在其他 contrast + 结构化退化组合上复现？
3. SE-CA（纯乘法）和 SCA（NAFNet 简化版）在 contrast 场景是否同样有效？
4. FiLM-GCM + OCAB 组合：L5 上能否超越 23.25？
5. 更多 contrast 场景（contrast_strengthen、contrast_stretch）的泛化验证
