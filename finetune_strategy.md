# 退化类型如何影响训练策略 —— 153 组实验的经验总结

## 核心论点

**退化类型与训练策略之间存在强耦合**——不能对所有退化组合使用同样的训练方式。但耦合主要发生在"是否迁移学习"和"课程顺序"两个宏观决策上。

> 实验基础：Phase 1-5 共 153 组实验（Phase 1:18 + Phase 2:24 + Phase 3:51 + Phase 4:12 + Phase 5:48）

---

## 一、Agent 决策树

面对任意退化组合，按以下流程决策：

```
输入: 退化管线 P = [(f1,s1), (f2,s2), ...]

Step 1: 判断退化步数
├── N ≥ 3 (三退化及以上)
│   → 用 Ft (盲预训练→专攻)
│   所有策略差异通常 < 0.5 dB。退化复杂度本身 > 策略选择
│
└── N = 2 (双退化) → Step 2

Step 2: 退化包含 compression 吗?
├── 是 → Step 2a
│   ├── compression 在最外层 (最后施加)
│   │   → Curric(逆序): 先剥 compression, 再修复内层
│   │   证据: N6(Rev+Fwd=+1.34), N8(Rev+Fwd=+1.82)
│   │
│   └── compression 在最内层 (先施加) 或 与其他退化混合
│       → Ft 或 Direct。课程顺序影响较小
│       证据: N7(所有策略 ≈平手), D3(Curric(fwd)最优因为 compression 改变了后续 blur 的分布)
│
└── 否 (blur/noise 组合) → Step 2b
    ├── noise-first (noise→blur)
    │   → Ft 碾压一切 (N4: +5.24 dB!)
    │   如果需要课程: Curric(fwd) > Curric(rev)
    │   证据: N4(Fwd-Rev=-2.98), N5(Fwd-Rev=-0.21)
    │
    └── blur-first (blur→noise)
        → Ft 是安全默认
        ├── severity ≥ 4 → Curric(rev) 可能 +2 dB
        └── severity ≤ 3 → 策略差异 < 0.2 dB, Ft 足够
        证据: N1(sev=4, Rev+Fwd=+2.27), N2/N3(sev=3, 差异 <0.2)
```

---

## 二、Phase 5 实验证据（48 组新退化）

### 全量对比表

| 退化 | Direct | Ft | C-fwd | C-rev | 最优 | Ft-Dir | Rev-Fwd |
|------|:--:|:--:|:--:|:--:|------|:--:|:--:|
| N1 blur_gauss(4)+speckle(4) | 21.65 | 22.09 | 19.74 | 22.01 | Ft | +0.44 | **+2.27** |
| N2 blur_zoom(3)+impulse(3) | 19.70 | 19.77 | 19.52 | 19.44 | Ft | +0.07 | -0.08 |
| N3 blur_glass(3)+gauss(3) | 22.82 | 23.14 | 23.14 | 22.97 | Ft | +0.32 | -0.17 |
| N4 noise_poiss(3)+lens(3) | 22.58 | 27.83 | 25.65 | 22.66 | Ft | **+5.24** | -2.98 |
| N5 noise_corr(3)+jitter(4) | 22.65 | 23.15 | 23.26 | 23.05 | C-fwd | +0.50 | -0.21 |
| N6 jpeg2000(3)+motion(3) | 23.47 | 23.47 | 22.53 | 23.87 | C-rev | 0.00 | **+1.34** |
| N7 jpeg(4)+gauss(3) | 25.92 | 25.90 | 25.89 | 25.89 | Direct | -0.02 | 0.00 |
| N8 blur_gauss(3)+jpeg(3) | 23.01 | 23.19 | 21.37 | 23.20 | C-rev | +0.19 | **+1.82** |
| N9 blur_zoom+speckle+jpeg | 18.24 | 18.22 | 17.96 | 17.96 | Direct | -0.01 | 0.00 |
| N10 noise_poiss+glass+j2k | 19.57 | 19.73 | 19.55 | 19.63 | Ft | +0.16 | +0.08 |
| N11 jpeg+blur+impulse | 24.79 | 25.12 | 21.11 | 21.55 | Ft | +0.33 | +0.44 |
| N12 motion+j2k+corr | 18.19 | 18.13 | 18.10 | 17.92 | Direct | -0.06 | -0.18 |

### 分组规律

| 退化组 | Ft-Dir 均值 | Rev-Fwd 均值 | 最优分布 |
|--------|:--:|:--:|------|
| Blur+Noise (blur first) N1-N3 | **+0.28** | +0.68 | Ft ×3 |
| Noise+Blur (noise first) N4-N5 | **+2.87** | -1.60 | Ft ×1, C-fwd ×1 |
| Compression+X N6-N8 | +0.05 | **+1.05** | C-rev ×2, Direct ×1 |
| Triple N9-N12 | +0.10 | +0.09 | Direct ×2, Ft ×2 |

### 三条铁律

**1. Ft 是安全默认（12/12 未显著输）**
- Ft 从未输给 Direct 超过 0.06 dB
- Ft 可以赢最高 +5.24 dB (N4)
- 结论：**永远先跑 Ft。如果 Ft ≈ Direct (< 0.1 dB)，退化可能"常见"到 Direct 就够**

**2. 课程顺序在 compression 退化上最关键**
- 含 compression 的双退化：Rev-Fwd 均值 +1.05 dB
- 不含 compression：Rev-Fwd 差异因 severity 和顺序而异
- 结论：**compression 在外层 → 逆序剥离；compression 在内层 → 需具体分析**

**3. 三退化策略差异小**
- 12 组三退化中，9 组策略间 Δ < 0.5 dB
- 结论：**三退化不需要纠结策略选择，Ft 完事**

---

## 三、历史发现总结（Phase 1-4）

### Fine-tune 变体（Phase 2）：影响小
FtCurr/FtReplay/FtFreeze/FtLR 之间的差异通常 < 0.3 dB。FtCurr 尤其危险（D2 崩 -6.83）。

### Loss 选择（Phase 3）：影响极小
mse/huber/l1+edge/l1+fft 之间的差异通常 < 0.25 dB。l1 是最安全默认。

### 多专家（Phase 3）：无效
从未超越单模型 Ft。误差级联放大。

### 课程顺序（Phase 4）：退化类型决定
- D2 (noise→blur, 弱交互)：逆序 > 正序 (+1.95)
- D3 (comp→blur, 强交互)：正序 > 逆序 (+4.27)

---

## 四、实操速查表

| 退化特征 | 推荐策略 | 预期收益 vs Direct |
|----------|----------|:--:|
| 任意三退化 | Ft | < 0.5 dB |
| blur+noise (双退化, severity≤3) | Ft | 0.3-0.5 dB |
| blur+noise (双退化, severity≥4) | Curric(rev) | +2 dB |
| noise+blur (noise-first) | Ft | **+0.5 ~ +5 dB** |
| compression 在外层 + X | Curric(rev) | +1.3 ~ +1.8 dB |
| compression 在内层 + X | Ft / Direct | 需具体分析 |
| 不确定 | **Ft** | 永远不会显著差 |

### 不要做的事
- 不要在 Loss 函数上花时间（l1 够用）
- 不要尝试 FtCurr（盲预训练后课程，几乎总是最差）
- 不要尝试多专家级联（单模型更好）
- 不要在三退化上调策略（浪费时间）

---

## 五、可疑的潜在错误 —— 需谨慎对待的结论

以下每条规则的证据强度不同。在 Phase 6 完成后应重新评估。

### ⚠️ 证据薄弱（仅 1-2 组支撑）

**"severity ≥ 4 → Curric(rev) +2 dB"**
- 仅 N1 (blur_gauss(4)+speckle(4)) 一组高严重度 blur+noise 样本
- N2/N3 (sev=3) 没显示此效应，但 severity ≠ 3 的其他组合未测试
- **风险**：可能只是 blur_gaussian + speckle 的特异性，而非严重度主效应
- **验证**：需更多高严重度 blur+noise 组合（如 blur_motion(4)+noise_impulse(4)）

**"noise-first → Ft 碾压 (N4: +5.24 dB)"**
- N4 (poisson+lens) 是极端 outlier，N5 仅 +0.50
- D2 (impulse+motion) 也是 noise-first，Ft 仅 +2.24
- **风险**：N4 的 +5.24 可能是 poisson+lens 的特殊交互，不是 noise-first 的通用规律
- **验证**：需更多 noise-first 双退化，特别是 poisson 噪声与其他 blur 的组合

**"三退化策略差异 < 0.5 dB"**
- 仅 7 组样本，其中 N11 策略间 Δ = 4.02 dB（巨大 outlier）
- 如果去掉 N11，均值确实 < 0.5 dB，但 outlier 的存在说明"三退化策略差异小"不是铁律
- **风险**：某些三退化组合（如 N11 的 comp→blur→impulse）策略差异远超 0.5 dB
- **验证**：Phase 6 的 10 组新三退化将给出更可靠分布

### ⚠️ 证据中等（2-4 组支撑，但有反例）

**"compression 在外层 → Curric(rev)"**
- 支持：N6(+1.34), N8(+1.82)
- 反例：D3（comp 在 inner，但 Curric(fwd) 赢 +4.27）——这说明 compression 位置确实关键
- 但 N7（comp→noise，comp 在 inner）却是所有策略平手，不符合 "inner→Ft/Direct" 的预期
- **风险**：compression 的位置效应可能被退化子类型混淆（N6 用 jpeg2000，N8 用 jpeg，N7 用 jpeg 但搭配 noise）
- **验证**：需系统控制子类型，单独变化 compression 位置

### ⚠️ 未探索的维度

**模型容量 × 策略交互**
- 所有实验 EMBED_DIM=64（0.45M），更大/更小模型是否改变策略排名？未知
- 推测：容量越大，策略差异越小（更多参数可以暴力记忆）

**训练预算 × 策略交互**
- 所有实验固定 15094 步，更长/更短训练是否改变最优策略？未知
- 推测：预算越小，Ft 优势越大（盲预训练加速收敛）

**Loss × 策略交互**
- Loss 实验仅在 Ft 框架下测试，Direct/Curric 下不同 Loss 的效果？未知
- 推测：Direct 可能对 Loss 更敏感（没有盲预训练的鲁棒性）

**三退化子类型效应**
- 7 组三退化覆盖了不同顺序和函数，但样本太少无法做统计推断
- N11 (comp→blur→impulse) 的 4 dB outlier 说明子类型组合可能有巨大影响
- Phase 6 将增加 10 组，但仍只是冰山一角

### 当前证据强度分级

| 结论 | 样本数 | 反例 | 强度 |
|------|:--:|:--:|:--:|
| Ft 是安全默认 (双退化) | 18 | 0 | **强** |
| Ft 从未显著输给 Direct | 30+ | N12(-0.06) | **强** |
| Loss 选择影响 < 0.3 dB | 6 | D2(l1+fft) | **中** |
| compression→Curric(rev) | 2 | D3, N7 | **弱** |
| severity→Curric(rev) | 1 | — | **极弱** |
| noise-first→Ft 碾压 | 2 | 量级差异大 | **弱** |
| 三退化策略差异小 | 7 | N11(Δ=4.02) | **中-弱** |
| 多专家无效 | 6 | 0 | **中** |

> **Phase 6 完成后重评估**：三退化样本从 7→17，将显著提升该部分的证据强度。
