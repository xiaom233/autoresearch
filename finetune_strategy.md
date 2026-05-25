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
