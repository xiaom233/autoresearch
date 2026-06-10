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
