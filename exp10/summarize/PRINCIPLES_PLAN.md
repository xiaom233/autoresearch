# 12 小时实验计划：退化条件化架构的抽象原则

## 目标

不是建立"退化X→架构Y"的查表，而是发现**决定架构选择的原则性因素**：
> 输入任意退化管线 → 分析退化属性 → 原则决定架构组件

## 六条候选原则

### 原则 A：数学形式决定修正类型

| 退化数学形式 | 需要的修正 | 架构机制 |
|-------------|-----------|---------|
| 仿射 `y=a·x+b` | 逐通道 scale+shift | 全局统计量→仿射参数 |
| 非线性曲线 `y=f(x)` | 逐像素逆曲线 | 逐通道曲线逼近 |
| 颜色空间变换 | 跨通道非线性映射 | 跨通道逐像素函数 |

**关键假设**：不是退化名称（contrast/brightness/saturate）决定架构，而是**数学形式**（affine/curve/colorspace）决定架构。

**验证**：Phase 2 的交叉实验（B1-B4 测曲线, C1-C4 测颜色空间+交叉验证）

### 原则 B：严重度放大架构差异

**假设**：严重度 1 时所有架构差不多（修正量小），严重度 5 时架构差异最大（修正量大）

**验证**：每种退化×3 严重度×2 架构
- contrast_scale sev=1/3/5 → FiLM-GCM vs Swin (6组)
- gamma sev=1/3/5 → ChannelCurve vs Swin (6组)
- saturate_HSV sev=1/3/5 → ColorMLP vs Swin (6组)
- 共 18 组

**预期**：sev=1 时 Δ<0.3dB, sev=5 时 Δ>2dB

### 原则 C：颜色空间决定跨通道需求

**假设**：
- RGB 空间操作 → 逐通道处理足够
- HSV 空间操作 → 必须跨通道（RGB↔HSV 非线性通道耦合）
- YCrCb 空间操作 → 线性跨通道即可（YCrCb↔RGB 是线性变换）

**验证**：
- 同操作（saturate_weaken），不同颜色空间（HSV vs YCrCb），同架构（ColorMLP）
- Phase 2 已有 Y1-Y2，补充 sev=1/5 验证线性/非线性差异

### 原则 D：局部退化严重度干扰全局统计量估计

**假设**：FiLM-GCM 依赖 GAP 提取通道统计量。严重 blur 平滑特征→统计量偏移；严重 noise 增加统计量方差→估计不准确。

**验证**：固定 contrast_weaken(3)，变化局部退化
- blur_gaussian sev=1/3/5 + contrast → FiLM-GCM (3组)
- noise_gaussian sev=1/3/5 + contrast → FiLM-GCM (3组)
- 测 FiLM Δ 是否随局部退化严重度递减

### 原则 E：退化步数不改变架构原则（但可能改变效果幅度）

**假设**：双退化（1局部+1全局）和三退化（2局部+1全局）适用相同的架构原则，但三退化的架构收益可能更大（退化更难，好架构的优势更明显）。

**验证**：取最佳组件，对比双退化 vs 三退化
- contrast_scale 双退化(L2) vs 三退化(新) → FiLM-GCM
- gamma 双退化(B1) vs 三退化(新) → ChannelCurve
- saturate 双退化 vs 三退化(L6) → ColorMLP
- 6 组

### 原则 F：组件可叠加，无负面交互

**假设**：多个原则同时适用时（如既有 HSV 操作又有 gamma 操作），对应组件可以叠加使用。

**验证**：组合退化 + 组合组件
- gamma + saturate_HSV → ChannelCurve + ColorMLP
- contrast_scale + brightness_HSV → FiLM-GCM + ColorMLP
- 4 组

## 12 小时时间线

```
现在 (12:00)
├─ Phase 2 继续运行 (~2h 剩余)
│
├─ 14:00 Phase 2 完成 → 分析初步结果
│
├─ 14:00-17:30  Wave 1: 原则B 严重度实验 (18组)
│   - 3 退化类型 × 3 严重度 × 2 架构
│   - 18 × 86min ÷ 8GPU ≈ 3.2h
│
├─ 17:30-20:00  Wave 2: 原则D+E (12组)
│   - 局部退化干扰 (6组) + 步数验证 (6组)
│   - 12 × 86min ÷ 8GPU ≈ 2.2h
│
├─ 20:00-22:00  Wave 3: 原则F+补充 (6组)
│   - 组件叠加 (4组) + 边缘情况 (2组)
│   - 6 × 86min ÷ 8GPU ≈ 1.1h
│
└─ 22:00-24:00  分析总结
    - 汇总所有实验结果
    - 提炼最终原则
    - 写入 summarize/
```

## 需要新建的退化文件

### 严重度实验（原则B）

```json
// contrast sev=1,3,5 + noise(3)
{"pipeline": [{"function": "noise_gaussian_RGB", "severity": 3}, {"function": "contrast_weaken_scale", "severity": 1}]}
{"pipeline": [{"function": "noise_gaussian_RGB", "severity": 3}, {"function": "contrast_weaken_scale", "severity": 5}]}

// gamma sev=1,5 + noise(3)  (sev=3 already in B1)
{"pipeline": [{"function": "noise_gaussian_RGB", "severity": 3}, {"function": "brightness_darken_gamma_RGB", "severity": 1}]}
{"pipeline": [{"function": "noise_gaussian_RGB", "severity": 3}, {"function": "brightness_darken_gamma_RGB", "severity": 5}]}

// saturate sev=1,5 + noise(3)  (sev=3 need new file)
{"pipeline": [{"function": "noise_gaussian_RGB", "severity": 3}, {"function": "saturate_weaken_HSV", "severity": 1}]}
{"pipeline": [{"function": "noise_gaussian_RGB", "severity": 3}, {"function": "saturate_weaken_HSV", "severity": 3}]}
{"pipeline": [{"function": "noise_gaussian_RGB", "severity": 3}, {"function": "saturate_weaken_HSV", "severity": 5}]}
```

### 局部退化干扰（原则D）

```json
// contrast(3) + blur sev=1,5
{"pipeline": [{"function": "blur_gaussian", "severity": 1}, {"function": "contrast_weaken_scale", "severity": 3}]}
{"pipeline": [{"function": "blur_gaussian", "severity": 5}, {"function": "contrast_weaken_scale", "severity": 3}]}

// contrast(3) + noise sev=1,5  (sev=3 already L2)
{"pipeline": [{"function": "noise_gaussian_RGB", "severity": 1}, {"function": "contrast_weaken_scale", "severity": 3}]}
{"pipeline": [{"function": "noise_gaussian_RGB", "severity": 5}, {"function": "contrast_weaken_scale", "severity": 3}]}
```

### 三退化验证（原则E）

```json
// contrast + blur + noise (已有 L4: blur+noise+brightness, 新建 contrast 版)
{"pipeline": [{"function": "blur_gaussian", "severity": 3}, {"function": "noise_gaussian_RGB", "severity": 3}, {"function": "contrast_weaken_scale", "severity": 3}]}
// gamma + blur + noise
{"pipeline": [{"function": "blur_gaussian", "severity": 3}, {"function": "noise_gaussian_RGB", "severity": 3}, {"function": "brightness_darken_gamma_RGB", "severity": 3}]}
// saturate + blur + noise
{"pipeline": [{"function": "blur_gaussian", "severity": 3}, {"function": "noise_gaussian_RGB", "severity": 3}, {"function": "saturate_weaken_HSV", "severity": 3}]}
```

## 预期产出（6条原则的验证结论）

```
原则 A: ✅/❌ 数学形式决定架构 — Phase 2 验证
原则 B: ✅/❌ 严重度放大差异 — Phase 3 验证
原则 C: ✅/❌ 颜色空间决定跨通道 — Phase 2+3 验证
原则 D: ✅/❌ 局部退化干扰全局估计 — Phase 4 验证
原则 E: ✅/❌ 步数不改变原则 — Phase 4 验证
原则 F: ✅/❌ 组件可叠加 — Phase 5 验证
```

## 最终目标

不是一张退化→架构的查表，而是一套**决策规则**：

```
输入: 退化管线 P = [(op₁, sev₁), (op₂, sev₂), ...]

分析退化属性:
  ∀ op ∈ P:
    math_form(op) ∈ {affine, curve, colorspace}
    color_space(op) ∈ {RGB, HSV, YCrCb}
    severity(op) ∈ [1,5]

决策规则:
  R1: if ∃ op with math_form=affine → FiLM-GCM
  R2: if ∃ op with math_form=curve → ChannelCurve  
  R3: if ∃ op with color_space=HSV → ColorMLP
  R4: if max(severity) ≤ 2 → 默认架构足够（Δ小）
  R5: if ∃ severe_local_degradation → 全局统计量估计需鲁棒化
  R6: 多规则触发时，对应组件可叠加

输出: 架构配置
```

这套规则的好处：
- **可泛化**：对未见过的退化组合也适用
- **可解释**：每步决策有明确的退化属性依据
- **可学习**：未来可以从外部数据学习 math_form/color_space 分类器
- **可更新**：新增退化类型只需要分类到现有属性维度
