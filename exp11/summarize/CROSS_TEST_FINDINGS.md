# exp11 交叉测试总报告

## 数据规模

| 轮次 | 训练模型 | 测试退化 | 测试数 | 图像 |
|:--:|:--:|:--:|:--:|:--:|
| Round1 | 4 | 8 | 8 | 0 |
| v2 | 15 | 14 | 19 | 0 |
| v3 | 63 | 14 | 839 | 839 tar |
| phaseE | 11 | 28 | 308 | 进行中 |
| **合计** | **~70** | **~30** | **~1174** | **~839** |

## 诊断特征发现

### 数值特征（839 组 v3 分析）

| 特征 | Self-test 均值 | 异常检测 Precision | 异常检测 Recall | 区分力 |
|------|:--:|:--:|:--:|:--:|
| **edge** | 0.044±0.020 | **97%** | 50% | ✅ 强 |
| **chroma** | 0.013±0.010 | **98%** | 50% | ✅ 强 |
| dir | 0.041±0.015 | 96% | 38% | ⚠️ 弱 |
| hflf | 0.285±0.139 | 96% | 39% | ⚠️ 弱 |

**规则**：edge > 0.10 且 chroma > 0.03 → 高置信误识别（98% 精度）

### 图像特征（65 cases, 6 categories）

| 类别 | edge_frac | entropy | color_asym | sat | 特征 |
|------|:--:|:--:|:--:|:--:|------|
| Self-OK | 0.160 | 0.92 | 0.052 | 0.208 | 基准 |
| Order wrong | 0.127 (-21%) | 2.29 (+148%) | 0.067 (+30%) | 0.240 (+16%) | 高熵 |
| Severity | 0.116 (-28%) | 2.64 (+186%) | 0.067 (+28%) | 0.184 (-11%) | 最高熵 |
| Type mismatch | 0.124 (-23%) | 2.80 (+203%) | **0.100 (+93%)** | **0.401 (+93%)** | 高颜色+高饱和 |
| SF1(pure contrast) | 0.140 (-12%) | 2.80 (+203%) | **0.129 (+150%)** | **0.387 (+86%)** | 极高颜色 |
| SF8(pure gamma) | 0.131 (-18%) | 2.57 (+178%) | 0.061 (+18%) | 0.148 (-29%) | 高熵低饱和 |

**关键发现**：
- **entropy 是最强单一区分特征**（+148%~203%）
- **color_asymmetry 可区分误差子类型**：type mismatch (+93%) vs order (+30%) vs severity (+28%)
- **saturation 方向不同**：SF8 是负向 (-29%)，其他正向
- **edge_frac 一致下降**：所有误差类型的残差都比正确退化更均匀

## 硬编码规则（可部署）

### 规则 1：PSNR 异常检测
```
if PSNR_spec < PSNR_self_baseline - 6dB:
    → 大概率误识别 (90% 非 self-test 满足此条件)
```

### 规则 2：标量特征确认
```
if edge > 0.10 and chroma > 0.03:
    → 高置信误识别 (98% precision)
elif edge > architecture_self_edge + 2σ:
    → 架构专属检测 (94-98% precision)
```

### 规则 3：图像特征推断误差类型
```
if entropy > 2.0 and color_asymmetry > 0.08:
    → 类型误识别 (很可能遗漏或错误全局退化类型)
elif entropy > 2.0 and saturation < 0.15:
    → 可能误识别为 gamma (SF8 特征)
elif edge_frac < 0.13 and entropy > 2.0:
    → 严重度偏差 (sev 高估或低估)
```

### 规则 4：架构敏感度
```
FiLM: threshold edge > 0.068 → 100% recall (最敏感检测器)
Swin: threshold edge > 0.081 → 97% precision
CSN:  threshold edge > 0.086 → 97% precision
```

## 架构鲁棒性排序

| 排名 | 架构 | avg_gap | 解读 |
|:--:|------|:--:|------|
| 1 | DualBranch | 10.1 dB | 最鲁棒 |
| 2 | CSN | 11.7 dB | — |
| 3 | Swin | 13.2 dB | 基线 |
| 4 | FiLM | 13.6 dB | 过拟合 |
| 5 | FreqMod | 15.2 dB | 差 |
| 6 | ColorPre | 16.0 dB | 最不鲁棒 |

**ColorPre 在自己退化上最强(+0.81~+18.91)，但跨退化最差(16.0 dB gap)**。高度特化 = 泛化差。

## 相互混淆的退化对

基于 839 组数据分析，以下退化对最容易相互混淆（双向 gap 都大）：

| 对 | gap | 特征 |
|------|:--:|------|
| BS ↔ SF7 | 16-20 dB | 全局颜色变换相互混淆 |
| L4 ↔ SF7 | 16-20 dB | — |
| SF5 ↔ SF7 | 14-21 dB | — |
| L2 ↔ L6 | 15-16 dB | 相同全局类型 |
| L2 ↔ L4 | 13-13 dB | 相同全局，不同步数 |

## 已知局限

1. 仅测试 Swin/FiLM/ColorPre/CSN/DualBranch/FreqMod/PCP (7 种架构)
2. 未覆盖 compression/lens/glass/poisson/speckle (phaseE 中)
3. 仅 3 数据集 (Set5/Urban100/DIV2K)
4. 仅 sev=1,3,5 (phaseE 新增 sev=2,4)
5. 仅 63 模型 (phaseE 新增 11)
