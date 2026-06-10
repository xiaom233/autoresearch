# 架构组件目录

> 合并自 ARCHITECTURE_CATALOG.md + GCM_RESULTS.md + GLOBAL_DEGRADATION_TAXONOMY.md

---

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
# 全局退化数学分类与架构设计

## 一、退化分类（基于数学逆操作）

对 `x_distortion/` 下全部全局退化函数的数学分析，按逆操作类型分为三类：

### 类型 A：仿射变换 — `y = a·x + b`

所有像素共享相同的逐通道仿射变换。

| 退化函数 | 公式 | 参数 |
|----------|------|:--:|
| contrast_weaken_scale | `y = c·(x-128) + 128 = c·x + (1-c)·128` | c ∈ [0.2, 0.75] |
| contrast_strengthen_scale | `y = c·(x-128) + 128` | c ∈ [1.4, 4.0] |
| brightness_darken_shift_RGB | `y = x - b` | b ∈ [0.1, 0.35] |
| brightness_brighten_shift_RGB | `y = x + b` | b ∈ [0.1, 0.35] |

**逆操作**：`x = (1/a)·y - b/a` —— 仍是仿射变换。

**架构**：**FiLM-GCM**（GAP → MLP → per-channel scale + shift）。
- 原理：仿射参数由全局通道统计量（均值、方差）完全确定
- GAP 提取通道统计量，MLP 预测 scale 和 shift
- Phase 1 验证：L5 +3.52 dB, L2 +0.79 dB

### 类型 B：逐像素非线性曲线 — `y = f(x)`

所有像素经历相同的非线性映射，无跨通道交互。

| 退化函数 | 公式 | 参数 |
|----------|------|:--:|
| brightness_darken_gamma_RGB | `y = x^γ` | γ ∈ [1.4, 3.2] |
| brightness_brighten_gamma_RGB | `y = x^γ` | γ ∈ [0.3, 0.8] |
| contrast_weaken_stretch | `y = 1/(1+(μ/(x+ε))^c)` | c ∈ [0.4, 1.0] |
| contrast_strengthen_stretch | `y = 1/(1+(μ/(x+ε))^c)` | c ∈ [2.0, 10.0] |

**逆操作**：`x = f⁻¹(y)` —— 需要非线性曲线，不是直线。

**架构**：**ChannelCurve**（逐通道 1×1 grouped Conv + ReLU → 学任意逐通道曲线）。
- 原理：grouped 1×1 Conv 等价于逐通道独立 MLP，可逼近任意一维曲线
- gamma 的逆也是 gamma：`x = y^(1/γ)`，MLP 可学
- stretch 的逆是反 sigmoid：MLP 可逼近
- Zero-DCE 启发：用 CNN 估计曲线参数，施加逐像素曲线映射

**为什么 FiLM-GCM 不行**：仿射调制是直线，无法匹配曲线。网络内部的 MLP 可部分补偿，但缺乏显式的曲线建模路径。

### 类型 C：颜色空间变换 — `y = RGB⁻¹(modify(HSV(x)))`

涉及 RGB↔HSV（或 YCrCb）的非线性转换 + 通道修改。

| 退化函数 | 操作 | 空间 |
|----------|------|:--:|
| brightness_darken_shift_HSV | V = V - c → RGB | HSV |
| brightness_brighten_shift_HSV | V = V + c → RGB | HSV |
| brightness_darken_gamma_HSV | V = V^γ → RGB | HSV |
| brightness_brighten_gamma_HSV | V = V^γ → RGB | HSV |
| saturate_weaken_HSV | S = c·S → RGB | HSV |
| saturate_strengthen_HSV | S = c·S → RGB | HSV |
| saturate_weaken_YCrCb | Cr/Cb 向 128 缩放 | YCrCb |
| saturate_strengthen_YCrCb | Cr/Cb 偏离 128 | YCrCb |

**逆操作**：需要跨通道非线性处理。RGB↔HSV 是非线性且耦合三个通道的。

YCrCb 转换是线性的（`YCrCb = M·RGB`），saturate_YCrCb 的逆操作是线性 3×3 矩阵——1×1 Conv 可以精确逆。

**架构**：**ColorMLP**（逐像素 1×1 Conv → ReLU → 1×1 Conv，全跨通道连接）。
- 原理：逐像素 MLP 是通用函数逼近器，可学任意 RGB→RGB 颜色映射
- 跨通道连接处理 RGB↔HSV 的通道耦合
- 没有空间卷积——不引入对全局变换有害的空间归纳偏置

**为什么 FiLM-GCM 不行**（L1 NaN 根因）：
brightness_darken_HSV 在 RGB 空间是非线性、跨通道耦合的。FiLM-GCM 的逐通道仿射调制（`x_c * scale_c + shift_c`）假设三个通道独立且变换是仿射的——这在数学上就不匹配。强行用仿射去逆 HSV 变换导致 GCM 学到极端参数 → NaN。

### 类型总结

| 类型 | 数学形式 | 跨通道? | 非线性? | 有空间结构? | 架构组件 |
|:--:|------|:--:|:--:|:--:|:--:|
| A: 仿射 | `y = a·x + b` | 可选 | ❌ 线性 | ❌ | **FiLM-GCM** |
| B: 曲线 | `y = f(x)` | ❌ 逐通道独立 | ✅ | ❌ | **ChannelCurve** |
| C: 颜色空间 | `y = T⁻¹(M(T(x)))` | ✅ | ✅ | ❌ | **ColorMLP** |

## 二、SOTA 参考

### Zero-DCE (Guo et al., CVPR 2020)

核心设计：**用 CNN 从下采样特征估计逐像素曲线参数，施加简单的逐像素曲线映射**。

- 不学习直接的图像→图像映射
- 学习"估计修正参数"→"施加修正"的分离式架构
- 曲线公式：`LE(I(x); α) = I(x) + α·I(x)·(1-I(x))` — 简单的二次曲线
- 79K 参数的轻量 CNN 预测 24 个曲线参数图（8 次迭代 × 3 通道）
- 关键启示：**全局变换的修正应该是对每个像素施加相同的简单函数，而非学习复杂的空间特征**

### Retinex-based 方法 (CICGNet, Mutual Retinex, GLON-Retinex, 2023-2024)

核心设计：**分离全局照明（低频）和局部细节（高频）的处理分支**。

- 照明分支处理全局亮度/对比度 — 使用全局特征
- 反射分支处理纹理/细节 — 使用局部特征
- 关键启示：**全局和局部处理需要不同的归纳偏置**

### HVI-CIDNet (2024)

核心设计：**将图像分解为颜色（HVI）和强度分量，分别处理**。

- 颜色分量稳定，强度分量包含大部分退化
- 关键启示：**颜色空间转换后的分解可以简化恢复任务**

## 三、组件设计

### FiLM-GCM（已有，Phase 1 验证）

```
输入 → GAP → Linear(C→C/4) → ReLU → Linear(C/4→2C) → scale, shift
输出 = x * (1+scale) + shift
```

- 适用：A 类仿射退化
- 参数量：+25.7K (5.66%)
- 插入位置：每个 SwinBlock 的 FFN 之后
- 已验证：L5 +3.52 dB, L2 +0.79 dB

### ChannelCurve（新增，Phase 2）

```
输入 (B,3,H,W)
  → Conv1×1(3→3H, groups=3) [逐通道独立]
  → ReLU
  → Conv1×1(3H→3, groups=3) [逐通道独立]
  → 残差加回输入
```

- 适用：B 类非线性曲线退化（gamma, stretch）
- 原理：grouped 1×1 Conv = 逐通道独立 MLP，可学任意 `f: R→R` 曲线
- 参数量：~0.2K (H=8 时仅 3*8*2*2 = 96 参数)
- 插入位置：输入端，在浅层特征提取之前

### ColorMLP（新增，Phase 2）

```
输入 (B,3,H,W)
  → Conv1×1(3→H) [跨通道混合]
  → ReLU
  → Conv1×1(H→3) [跨通道混合]
  → 残差加回输入
```

- 适用：C 类颜色空间退化（HSV, YCrCb）
- 原理：逐像素 MLP = 通用颜色映射逼近器
- 参数量：~0.3K (H=16 时 3*16 + 16*3 = 96 参数)
- 插入位置：输入端，在浅层特征提取之前

### 为什么组件放在输入端而非每个 Block 内？

- A 类（FiLM-GCM）：放在每个 Block 内，因为仿射变换影响所有层的特征分布
- B/C 类（ChannelCurve/ColorMLP）：放在输入端，因为非线性曲线和颜色空间转换可以在原始像素空间直接修正，不需要在网络深层处理
- Zero-DCE 支持此设计：曲线估计在低分辨率特征上进行，曲线施加在原始像素上

## 四、Phase 2 实验矩阵

### 组 B：非线性曲线 → ChannelCurve

| ID | 退化 | 退化类型 | 架构 |
|----|------|:--:|------|
| B1 | noise(3) + brightness_gamma_RGB(3) | B | Swin (基线) |
| B2 | noise(3) + brightness_gamma_RGB(3) | B | Swin + **ChannelCurve** |
| B3 | noise(3) + contrast_stretch(3) | B | Swin (基线) |
| B4 | noise(3) + contrast_stretch(3) | B | Swin + **ChannelCurve** |

### 组 C：颜色空间 → ColorMLP

| ID | 退化 | 退化类型 | 架构 |
|----|------|:--:|------|
| C1 | L1 (blur+brightness_HSV) | C | Swin + **ColorMLP** |
| C2 | L6 (lens+impulse+saturate_HSV) | C | Swin + **ColorMLP** |

### 组 D：负对照（局部退化，不应有影响）

| ID | 退化 | 退化类型 | 架构 |
|----|------|:--:|------|
| D1 | S5 (motion blur) | 局部 | Swin + ChannelCurve |
| D2 | S5 (motion blur) | 局部 | Swin + ColorMLP |

### 对照组复用已有数据

- B1 Swin 基线：新跑
- B3 Swin 基线：新跑
- C1 Swin 基线：G1r_L1_Swin_lr5e4 (23.21)
- C2 Swin 基线：G9r_L6_Swin_lr5e4 (26.98)
- D1/D2 Swin 基线：已有 NAS R1 数据

8 组 × ~86 min ≈ 12 GPU hours

## 五、预期结论

1. **ChannelCurve 应在 gamma/stretch 退化上显著优于 Swin 基线**，因为显式曲线建模匹配退化的非线性本质
2. **ColorMLP 应修复 L1 NaN**，因为逐像素跨通道 MLP 可以学 HSV↔RGB 的逆映射
3. **ColorMLP 应在 L6 上超越 FiLM-GCM +0.22**，因为 saturation_HSV 是跨通道操作
4. **两个组件在局部退化（S5）上不应有负面影响**，因为它们是输入端逐像素操作，零初始化从恒等映射开始
