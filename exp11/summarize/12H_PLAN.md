# exp11 Phase E: 12 小时验证 — 聚焦困难案例

## 数据驱动的困难案例识别

基于 839 组交叉测试分析：

| 难度 | 案例 | 特征 | 最大 gap |
|:--:|------|------|:--:|
| 🔴 极难 | **SF8(纯gamma)↔任意退化** | 纯全局模型对复杂退化完全崩溃 | 26 dB |
| 🔴 极难 | **SF1(纯contrast)↔复杂退化** | 同上，纯仿射无法处理混合退化 | 24 dB |
| 🟠 困难 | **BS↔SF7** | 同为全局颜色变换，相互混淆 | 16-20 dB |
| 🟠 困难 | **L2↔L4, L2↔L6** | 相同全局类型，不同局部伙伴 | 13-16 dB |
| 🟡 中等 | **Sev=2 vs Sev=3, Sev=3 vs Sev=4** | 严重度接近 | 未知 |
| 🟡 中等 | **L2(contrast)↔L1(brightness_HSV)** | 不同全局类型 | 11-14 dB |

## 关键假设

1. **H1**: 纯全局退化(SF1/SF8)训练模型 → 应在任意测试退化上产生巨大 gap → 最容易检测
2. **H2**: BS↔SF7 相互混淆 → 诊断特征中 color_asymmetry 应能区分
3. **H3**: Sev±1 的 PSNR gap 在 2-5 dB（L6 更大，L2 更小）
4. **H4**: 修正训练（检测→纠错→重训）PSNR > 初始错误训练 PSNR + 3dB

## 实验矩阵（65 组，~12h）

### Wave 1: 严重度精细扫描（20 组, ~3.5h）

新增 sev=2,4 测试，验证 ±1 sev 的检测能力。

| ID | GT | 架构 | 训练 sev | 验证 |
|----|------|------|:--:|------|
| S1-S4 | L2 contrast+noise | Swin | 1,2,4,5 | L2 严重度全扫描 |
| S5-S8 | L6 saturate+impulse+lens | Swin | 1,2,4,5 | L6 严重度全扫描 |
| S9-S12 | SF7 saturate+noise | Swin | 1,2,4,5 | SF7 严重度扫描 |
| S13-S16 | L2 | ColorPre | 1,2,4,5 | 架构对严重度的敏感度 |
| S17-S20 | L6 | ColorPre | 1,2,4,5 | HSV 退化架构敏感度 |

**关键**：首次测试 sev=2,4 → 验证 ±1 sev 是否可检测。

### Wave 2: 纯全局退化误识别（12 组, ~2.5h）

809 组数据中 SF8/SF1 是最大混淆源。专项验证。

| ID | GT | 训练(SF8误识别) | 训练(SF1误识别) | 修正后 |
|----|------|:--:|:--:|:--:|
| P1-P2 | L2 | SF8→L2 | SF1→L2 | ColorPre修正 |
| P3-P4 | L6 | SF8→L6 | SF1→L6 | ColorPre修正 |
| P5-P6 | SF7 | SF8→SF7 | SF1→SF7 | ColorPre修正 |
| P7-P8 | L5 | SF8→L5 | — | ColorPre修正 |
| P9-P12 | 控制组: Swin on GT(4组) | — | — | baseline |

### Wave 3: 全局类型相互混淆（10 组, ~2h）

BS↔SF7 是最难区分的对（mutual 36 dB）。

| ID | GT | 训练(混淆) | 诊断重点 |
|----|------|------|------|
| M1-M2 | BS | SF7 | color_asymmetry 区分 |
| M3-M4 | SF7 | BS | saturation 区分 |
| M5-M6 | L2 | L4 | 步数差异检测 |
| M7-M8 | L2 | L6 | 局部退化差异检测 |
| M9-M10 | 控制组 | — | baseline |

### Wave 4: HSV 退化严重度（8 组, ~1.5h）

HSV 退化对严重度最敏感（L6 sev±1 = 14-17 dB）。

| ID | GT | 训练 | 验证 |
|----|------|------|------|
| H1-H4 | L1 sev=3 | sev=1,2,4,5 | brightness_HSV 严重度 |
| H5-H8 | bHSV sev=3 | sev=1,2,4,5 | 纯 brightness_HSV |

### Wave 5: Cascade 修正 vs 重训练（8 组, ~1.5h）

| ID | GT | Phase A | Phase B | 对比 |
|----|------|------|------|------|
| C1-C2 | L2 (遗漏contrast) | Swin spatial 1.5ep | +ColorPre 0.5ep | vs 重训练 |
| C3-C4 | L5 (遗漏contrast) | Swin spatial 1.5ep | +ColorPre 0.5ep | vs 重训练 |
| C5-C6 | L6 (遗漏saturate) | Swin spatial 1.5ep | +ColorPre 0.5ep | 已知失败 |
| C7-C8 | SF7 (遗漏saturate) | Swin spatial 1.5ep | +ColorPre 0.5ep | 新测试 |

### Wave 6: 架构敏感度（7 组, ~1h）

| ID | GT | 架构 | 验证 |
|----|------|------|------|
| A1-A3 | L2 sev=3→sev=1(误) | ColorPre, FiLM, CSN | 谁最敏感? |
| A4-A7 | L6 sev=3→sev=5(误) | ColorPre, FiLM, CSN, Swin | 谁最敏感? |

### Wave 7: 反思总结（0 组, ~1h）

汇总 + 更新 REFLECTION_MECHANISM.md。

## 时间线

```
0h ─── Wave 1 启动 (20 组并行, 8 GPU)
3.5h ─ Wave 2 启动 (12 组)
6h ─── Wave 3+4 启动 (18 组)  
8h ─── Wave 5 启动 (8 组)
9.5h ─ Wave 6 启动 (7 组)
11h ── 全部完成, 开始 Wave 7 分析
12h ── 提交最终报告
```

## 预期产出

1. Sev ±1 检测准确率（首次）
2. 纯全局→复杂退化的修正成功率
3. BS↔SF7 区分规则
4. Cascade vs 重训练 效率对比
5. 架构敏感度排序
6. 更新 REFLECTION_MECHANISM.md 可部署规则
