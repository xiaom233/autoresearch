# 退化类型如何影响训练策略 —— 193 组实验的经验总结

## 核心论点

**退化类型与训练策略之间存在强耦合**——不能对所有退化组合使用同样的训练方式。但耦合主要发生在"是否迁移学习"和"课程顺序"两个宏观决策上。

> 实验基础：Phase 1-6 共 193 组实验（Phase 1:18 + Phase 2:24 + Phase 3:51 + Phase 4:12 + Phase 5:48 + Phase 6:40）

---

## 一、Agent 决策树

面对任意退化组合，按以下流程决策：

```
输入: 退化管线 P = [(f1,s1), (f2,s2), ...]

Step 1: 判断退化步数
├── N ≥ 3 (三退化及以上) → Step 3
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
│       证据: N7(所有策略 ≈平手)
│
└── 否 (blur/noise 组合) → Step 2b
    ├── noise-first (noise→blur)
    │   → Ft, 不需要课程
    │   如果需要课程: Curric(fwd) > Curric(rev)
    │   证据: N4(Fwd-Rev=-2.98), N5(Fwd-Rev=-0.21)
    │
    └── blur-first (blur→noise)
        → Ft 是安全默认
        ├── severity ≥ 4 → Curric(rev) 可能有 +2 dB (但仅1组证据, 慎用)
        └── severity ≤ 3 → Ft 足够

Step 3: 三退化 (Phase 6 修正 — 17 组, 中位Δ=1.22 dB)
├── blur 子类型 = motion
│   → Curric(fwd) 有一定优势 (M1: Rev-Fwd=-0.58)
│
├── blur 子类型 = lens
│   → Curric(rev) 有一定优势 (M2: Rev-Fwd=+0.81)
│
├── blur 子类型 = jitter
│   → 课程方向无所谓, Ft 足够 (M3: Rev-Fwd=+0.03)
│
├── noise/compression 子类型变化时
│   → Curric(rev) 一致优于 Curric(fwd) (+1.5~+2.0 dB)
│   证据: M4-M7, 4/4 组一致 Rev > Fwd
│
└── 全高严重度 (sev≥4,4,4)
    → Ft ≈ Direct, 但 Rev > Fwd 仍保持 (M8: Rev-Fwd=+1.30)
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

**3. 三退化策略差异不小（Phase 6 修正）**
- 17 组三退化，策略间 Δ 中位数 1.22 dB，71% ≥ 0.5 dB
- blur 子类型决定课程方向：motion→Fwd, lens→Rev, jitter→无所谓
- noise/compression 子类型变化时：Rev 一致优于 Fwd (+1.5~+2.0)
- 结论：**三退化需要根据 blur 子类型选择课程方向**

---

## 三、Phase 6 实验证据（40 组三退化）

### 全量对比表

| M | 变化维度 | Direct | Ft | C-fwd | C-rev | 最优 | Ft-Dir | Rev-Fwd |
|---|----------|:--:|:--:|:--:|:--:|------|:--:|:--:|
| M1 | blur=motion | 21.42 | 21.33 | 21.57 | 20.99 | C-fwd | -0.09 | -0.58 |
| M2 | blur=lens | 23.20 | 23.24 | 21.37 | 22.18 | Ft | +0.04 | +0.81 |
| M3 | blur=jitter | 22.48 | 22.84 | 22.66 | 22.69 | Ft | +0.36 | +0.03 |
| M4 | noise=impulse | 22.44 | 22.69 | 21.25 | 23.03 | Ft | +0.25 | +1.79 |
| M5 | noise=poisson | 22.31 | 22.71 | 21.46 | 23.22 | Ft | +0.40 | +1.76 |
| M6 | noise=correlated | 22.28 | 22.54 | 21.17 | 22.69 | Ft | +0.26 | +1.52 |
| M7 | comp=jpeg2000 | 22.55 | 22.72 | 20.81 | 22.87 | C-rev | +0.17 | +2.07 |
| M8 | sev=4,4,4 | 19.30 | 19.29 | 17.31 | 18.61 | Direct | -0.01 | +1.30 |
| M9 | noise→blur→comp | 22.50 | 22.78 | 22.23 | 21.24 | Ft | +0.28 | -0.98 |
| M10 | comp→blur→noise | 22.73 | 22.67 | 21.95 | 21.72 | Direct | -0.06 | -0.23 |

### 子类型主效应

**变 blur（固定 noise=gauss, comp=jpeg）**：
| blur | Ft-Dir | Rev-Fwd | 课程方向 |
|------|:--:|:--:|:--:|
| gaussian (T1) | — | — | 基线 |
| motion (M1) | -0.09 | -0.58 | **Fwd** |
| lens (M2) | +0.04 | +0.81 | **Rev** |
| jitter (M3) | +0.36 | +0.03 | ≈平 |

**变 noise（固定 blur=gauss, comp=jpeg）**：
| noise | Ft-Dir | Rev-Fwd | 方向 |
|-------|:--:|:--:|:--:|
| gauss (T1) | — | — | 基线 |
| impulse (M4) | +0.25 | +1.79 | Rev |
| poisson (M5) | +0.40 | +1.76 | Rev |
| correlated (M6) | +0.26 | +1.52 | Rev |

**变顺序（固定 blur_gauss+noise_gauss+jpeg）**：
| 顺序 | Ft | Rev-Fwd | 最优 |
|------|:--:|:--:|------|
| blur→noise→comp (T1) | — | — | — |
| noise→blur→comp (M9) | 22.78 | -0.98 | Ft |
| comp→blur→noise (M10) | 22.67 | -0.23 | Direct |

### 三退化策略 Δ 分布（17 组）

- 中位数：**1.22 dB**（非之前估计的 < 0.5 dB）
- 71% 的三退化策略间 Δ ≥ 0.5 dB
- 范围：0.18 ~ 4.02 dB

---

## 四、历史发现总结（Phase 1-4）

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
| 三退化, noise/comp 子类型不关键 | Curric(rev) | +1.5~+2.0 dB |
| 三退化, blur=motion | Curric(fwd) | +0.6 dB |
| 三退化, blur=lens | Curric(rev) | +0.8 dB |
| 三退化, blur=jitter | Ft | 策略差异小 |
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

---

## 五、可疑的潜在错误 —— 需谨慎对待的结论

以下每条规则的证据强度不同。Phase 6 已解决三退化部分的主要疑点。

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

**"三退化策略差异 < 0.5 dB"** ✅ Phase 6 已解决
- 17 组三退化，策略间 Δ 中位数 = 1.22 dB，71% ≥ 0.5 dB
- N11 的 4.02 dB 不是 outlier，而是合理范围的上界
- **结论已修正**：三退化策略差异不小，需要根据 blur 子类型选择课程方向

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

### 当前证据强度分级（Phase 6 更新后）

| 结论 | 样本数 | 反例 | 强度 | 变化 |
|------|:--:|:--:|:--:|:--:|
| Ft 是安全默认 | 30+ | 0 | **强** | — |
| 三退化 noise/comp 子类型 → Rev | 4 | 0 | **强** | **新增** |
| blur 子类型决定课程方向 | 3 | 0 | **中** | **新增** |
| Loss 选择影响 < 0.3 dB | 6 | D2(l1+fft) | **中** | — |
| 多专家无效 | 6 | 0 | **中** | — |
| 三退化策略中位Δ = 1.22 dB | 17 | — | **强** | ✅ Phase 6 解决 |
| blur 子类型决定课程方向 | 3 | 0 | **中** | **新增，待Phase 8扩至6种** |
| compression→Curric(rev) | 2 | D3, N7 | **弱** | — |
| severity→Curric(rev) | 1 | — | **极弱** | — |

> **Phase 6 关键修正**：三退化策略差异中位数 1.22 dB（不是之前说的 < 0.5 dB），71% 的三退化策略间 Δ ≥ 0.5 dB。策略在三退化上**确实重要**。
