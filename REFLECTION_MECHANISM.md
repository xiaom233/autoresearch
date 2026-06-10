# Reflection Mechanism — 训练策略反思修正

> 来源: exp11 (~1174组 误识别分析) + exp12 (56退化 多轮修正) + exp13 (24退化 盲识别全流程)
> 核心洞察: 盲识别质量 > 训练策略调整 > LR 微调

---

## 一、反思的三个层次（按优先级排列）

```
第 1 层：盲识别修正（影响最大，-17.5 dB → +15+ dB 潜在收益）
第 2 层：架构选择（影响中等，-5.9 dB → +3.5 dB）
第 3 层：训练策略（影响最小，< 0.5 dB）
```

**关键教训** (exp12)：纯训练策略调整（LR 变化）几乎无法超越原始 Ft (EPOCH=2, LR=5e-4)。
改进应聚焦于修正退化参数（severity）或架构升级。

---

## 二、第一层：盲识别修正 — 用 PSNR 反推识别误差

盲识别错误是 PSNR 损失的**最大单一来源**。

### 各类型盲识别错误的代价 (来源: exp11 BLIND_VALIDATION.md)

| 错误类型 | 平均 PSNR 损失 | 测试数 |
|---------|:--:|:--:|
| **类型错误** (type wrong) | ~17.5 dB | 7 |
| 严重度错误 (sev wrong) | ~14.5 dB | 7 |
| 漏检全局退化 (missing global) | ~12.9 dB | 4 |

注：来源仅测试了上述三种错误类型。未单独测试"顺序错误"类别。

### 硬编码诊断规则 (来源: exp11 CROSS_TEST_FINDINGS.md)

#### 规则 1：PSNR 异常检测
```
if PSNR_spec < M_blind_baseline - 6 dB:
    → 高概率盲识别错误 (90% 错误案例满足此条件)
```

#### 规则 2：标量特征确认 (来源: v3 839 组分析)
```
if edge > 0.10 AND chroma > 0.03:
    → 高置信误识别 (98% precision, 50% recall)
```
数值特征基准 (self-test 均值 ± 标准差):
- edge: 0.044 ± 0.020
- chroma: 0.013 ± 0.010
- dir: 0.041 ± 0.015
- hflf: 0.285 ± 0.139

#### 规则 3：推断误差类型 (来源: 65 cases, 6 categories)
```
if entropy > 2.0 AND color_asymmetry > 0.08:
    → 退化类型误识别 (遗漏或错误全局退化)
elif entropy > 2.0 AND saturation < 0.15:
    → 可能误识别为 gamma 型退化
elif edge_frac < 0.13 AND entropy > 2.0:
    → 严重度偏差 (估计过高或过低)
```

参考基准 (Self-OK): edge_frac=0.160, entropy=0.92, color_asym=0.052, saturation=0.208

#### 规则 4：架构敏感度阈值
```
FiLM: threshold edge > 0.068 → 100% recall (最敏感检测器)
Swin: threshold edge > 0.081 → 97% precision
CSN:  threshold edge > 0.086 → 97% precision
```

### 容易混淆的退化对 (来源: exp11, 基于 839 组分析)

| 退化对 | PSNR Gap | 已知特征 |
|--------|:--:|------|
| BS ↔ SF7 | 16-20 dB | 全局颜色变换相互混淆 |
| L4 ↔ SF7 | 16-20 dB | (来源未记录因果关系) |
| SF5 ↔ SF7 | 14-21 dB | (来源未记录因果关系) |
| L2 ↔ L6 | 15-16 dB | 相同全局类型 |
| L2 ↔ L4 | 13 dB | 相同全局，不同步数 |

注：带有 "—" 的因果关系是推测性的，未经验证。

### 修正方向

1. **Spec << M_blind** → 盲识别可能错误，检查 predicted_params，参考误识别诊断规则
2. **Spec > M_blind 但大幅低于预期** → 退化参数可能偏轻/偏重，调整 severity
3. **Spec ≈ Ft ≈ M_blind** → 盲识别高度准确，无需修正退化参数

---

## 三、第二层：架构选择修正

### 架构-退化匹配规则

| 退化特征 | 推荐架构 | 避免 |
|---------|---------|------|
| 严重 motion blur (sev ≥ 5) | OCAB + ws=16 + SwiGLU | MDTA |
| 随机噪声为主 | 任意 (Swin 即可) | — |
| contrast 全局退化 | Swin (只用 Swin) | MDTA, OCAB |
| contrast + 结构化局部 | Swin + FiLM-GCM | ColorPre |
| 纯 gamma/brightness_shift | Swin 基线 (无需额外) | — |
| 盲识别不确定 (低置信度) | DualBranch 或 CSN | ColorPre (泛化差) |

### 推荐附加组件

| 组件 | 参数 | 适用场景 | 效果 | 风险 |
|------|:--:|------|:--:|------|
| ColorPre | +0.8K | 默认首选，全局退化 | 最稳定 | 跨退化泛化差 |
| CSN | +1K | 通用，最轻量 | L5 +1.39 | 效果温和 |
| FiLM-GCM | +26K | contrast+结构化 | L5 +3.52 | L1 NaN (超出参数限制需减小 EMBED_DIM) |
| DualBranch | +3K | 需要鲁棒性时 | 最鲁棒 (10.1 gap) | 速度 -3% |

---

## 四、第三层：训练策略修正

### 修正决策树

```
Spec < Ft?
├── 全局退化 (contrast 类型)?
│   → 废弃 Ft, 改用 Direct (来源: exp9, L2 Ft -4.51 dB)
├── Δ > 3 dB → 严重问题
│   ├── 退化复杂 (≥2种) → 检查是否局部退化 Ft 可用
│   └── 退化简单 (1种) → 检查是否训练崩溃, 重跑
├── 1 < Δ ≤ 3 dB → 中度差距
│   └── 检查 severity 是否合理: R1 常高估 severity
├── 0.3 < Δ ≤ 1 dB → 轻微差距
│   └── Ft + LR=5e-4, 可能已接近当前架构最优
└── Δ ≤ 0.3 dB → 基本持平
    └── 不追额外训练 (已收敛)
```

### 关键教训 (来源: exp12)

1. **降 LR (5e-4→1e-4) 很少帮助 Ft** — 仅在特殊案例有效 (如 T20, blur sev=4 + 低LR 防止盲特征被覆盖)
2. **R1 最常见的错误是高估 severity** — 多组失败案例源于 severity 被错误上调
3. **Ft (LR=5e-4, EPOCH=2) 在局部退化上已接近当前架构的训练策略上限**
4. **loss 函数选择影响 < 0.25 dB** (来源: exp9, l1/mse/huber/l1+edge/l1+fft 之间差异极小)，**不应作为 R2 修正方向**

---

## 五、多轮修正机制

### R1 (首轮修正)
- 如果 Spec << M_blind: 检查盲识别 → 修正 predicted_params (最高优先级)
- 如果 Spec 接近 M_blind 但 << Ft: 检查 severity → 调整参数 ±1
- 如果全局退化用了 Ft: 切换到 Direct
- 如果架构选择不当: 参考架构速查表更换

### R2 (二次修正)
- R1 后 Δ > 0.5 dB → 继续调整
- 方向：调整架构附件 (添加 CSN/ColorPre)、尝试不同的 Curric 方向
- **不能增加 EPOCH** (必须 EPOCH=2)
- **不建议尝试不同 loss 函数** (差异 < 0.25 dB, 已验证无效)

### R3 (三次修正)
- R2 后仍有明显差距 → 尝试 Curric、Cascade 等更大改动
- 最多 3 轮，之后标记"已收敛"

### 收敛判定
- Δ < 0.3 dB → 已收敛，不值得继续
- R1/R2 倒退 → 修正方向错误，回退到前一版本

---

## 六、训练公平性规则 ⚠️

1. **EPOCH_BUDGET 必须统一 = 2** — 与基线 Spec/Ft 一致。禁止 EPOCH=3
2. **模块参数 ≤ 基线 × 1.05 (≤ 477K)** — 架构对比必须在同参数下 (来源: exp10 原则 6)
3. **VAL_PARAMS_PATH 必须锁定为 GT** — R1 只允许修改 PARAMS (exp12 的致命错误)
4. **PSNR 提升必须 > 0.5 dB 才值得继续下一轮**
5. **所有对比在同一 GT 上评估** — 否则毫无意义

---

## 七、全局退化处理规则 (来源: exp9 + exp10)

| 规则 | 内容 |
|------|------|
| 识别 | 管线含 contrast_weaken/strengthen、brightness_*、saturation_*、gamma_* |
| contrast 型 | Direct (不用 Ft)，Swin 架构 |
| brightness 型 | Direct 保守，Ft 在 L1 上安全 (+0.41) |
| 架构 | Swin (不用 MDTA/OCAB) |
| 附加 | 可选 ColorPre (+0.8K) 或 CSN (+1K) |
| FiLM-GCM 预警 | brightness_HSV + blur 组合：gaussian_blur 平滑 HSV 极端值 → 统计量偏移大 → GCM 学到极大 scale/shift → 训练后期 NaN。有额外 noise (L4) 时特征多样性增加 = 隐式正则化 → 不 NaN |
| 验证 | exp9 L2(contrast+noise): Ft 崩溃 -4.51 dB; exp10: L1(brightness+blur): Ft 安全 +0.41 |
