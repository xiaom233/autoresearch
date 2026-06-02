# exp10: 退化条件化架构生成

## 终极目标

> **构建一个系统：输入任意退化管线，输出为该退化量身定制的训练策略和模型架构。**
>
> 不是找"最好"的通用方案——是让系统的每个组件（策略、架构、超参）都根据退化特征自适应调整。

```
输入: 退化管线 P = [(blur_motion,5), (noise_gauss,1), (comp_jpeg,1)]
  │
  ├→ 退化编码器: 提取退化特征 (类型, 严重度, 交互强度, 结构化程度)
  │
  ├→ 策略生成器: Ft/Curric? Fwd/Rev? LR? Loss?  ← exp9 已验证
  │
  ├→ 架构生成器: Attention? Norm? Window? Depth? ← exp10 待验证
  │
  └→ 输出: 为该退化定制的训练方案 (代码 + 配置)
```

exp9 解决了策略生成器（~450 组实验）。exp10 解决架构生成器。

## 核心思想

**不是搜索一个最优架构——是根据退化类型自动生成适配架构。** 不同退化需要不同的网络结构偏置，架构应该是退化特征的函数。

```
退化管线 P → 退化编码器 → 架构配置 A → 生成模型代码 → 训练
```

## 与传统 NAS 的本质区别

| 传统 NAS | 退化条件化架构 |
|----------|---------------|
| 搜索**一个**通用架构 | 为**每种**退化生成专用架构 |
| 预定义搜索空间 | 退化特征 → 架构映射 |
| 架构搜索与退化无关 | 架构由退化属性决定 |
| RL/进化驱动 | 退化特征驱动 |

## 退化-架构耦合规律（基于 ~450 组实验 + XRestormer 比较研究）

### motion blur
- **特征**: 方向性, 结构化, 1D反卷积可解
- **架构偏置**: 方向感知空间注意力, 大窗口(12-16), 深层(3+ RSTB)
- **XRestormer组件**: OCAB (overlap空间注意力), 相对位置编码

### lens blur  
- **特征**: 径向对称, PSF复杂, 单独修复困难
- **架构偏置**: 径向对称注意力 or 大感受野, 双分支(channel+spatial)
- **XRestormer组件**: 双分支TransformerBlock (channel + spatial)

### gaussian noise
- **特征**: 独立同分布, 像素级, 通道独立
- **架构偏置**: 通道注意力(MDTA), 浅层(1-2 RSTB), 小窗口(4)
- **XRestormer组件**: MDTA (通道注意力), InstanceNorm

### impulse noise
- **特征**: 稀疏离群值, 非高斯
- **架构偏置**: 小窗口(4), 浅层, 中值滤波 bias
- **XRestormer组件**: 轻量MDTA

### compression
- **特征**: 8×8块效应, 频域结构化
- **架构偏置**: 频域增强, 窗口=8 (匹配块大小), SE通道混合
- **XRestormer组件**: SE-like channel recalibration

### 混合退化 (blur+noise+compression)
- **特征**: 多种梯度冲突
- **架构偏置**: 双分支(channel+spatial), 多尺度(U-Net skip)
- **XRestormer组件**: 完整双分支 + U-Net encoder-decoder

## 实施：退化条件化的模块化架构

### 退化编码器

输入退化管线，输出架构配置：

```python
def degradation_to_architecture(pipeline):
    features = extract_degradation_features(pipeline)
    return {
        "attention_type": select_attention(features),
        "norm_type": select_norm(features),
        "window_size": select_window(features),
        "depth": select_depth(features),
        "channel_mix": select_mix(features),
    }
```

### 可替换模块池

| 模块 | 选项 | 退化条件 |
|------|------|----------|
| Attention | Swin(窗口) / MDTA(通道) / OCAB(空间) / Dual | 退化结构 |
| Norm | LayerNorm / InstanceNorm / BatchNorm | 退化统计 |
| FFN | MLP / GDFN(门控) | 容量需求 |
| Window | 4 / 8 / 12 / 16 | 感受野需求 |
| ChannelMix | none / SE / ECA / LayerScale | 通道交互 |

## Round 1：退化-注意力耦合验证（最小可行实验）

验证一个核心假设：**空间注意力对blur更有效，通道注意力对noise更有效。**

### 实验设计

3 组退化 × 3 种注意力 = 9 组实验

| 退化 | Swin(基线) | MDTA(通道) | OCAB(空间) |
|------|:--:|:--:|:--:|
| D3 (comp+blur, 结构化) | ✓ 已有 | ✓ 新 | ✓ 新 |
| N4 (noise+blur, 随机主导) | ✓ 已有 | ✓ 新 | ✓ 新 |
| S5 (motion, 结构化) | ✓ 已有 | ✓ 新 | ✓ 新 |

每实验 × Ft 策略 × EPOCH=1 = 9 × 43 min ≈ 6.5 GPU hours

### 预期

- D3/S5 (结构化): OCAB > Swin > MDTA
- N4 (噪声): MDTA > Swin > OCAB
- 如果成立 → 退化-注意力耦合存在 → 扩展全模块池
- 如果不成立 → 架构差异 < 0.3 dB → 放弃架构方向
