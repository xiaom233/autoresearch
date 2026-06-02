# 训练策略调优总结：什么有效、什么无效

> ~450 组实验，EMBED_DIM=64 (0.45M)，EPOCH_BUDGET=2 (15094步)

---

## 一、有效策略（按效应量排序）

### 1. Ft (盲预训练→专攻) — 最安全默认 ★★★★★

| 条件 | 效应 |
|------|:--:|
| sev≤3 | Ft > Direct +0.3~+5 dB |
| sev≥4 | Ft ≈ Direct |
| noise-first 双退化 | Ft 优势最大 (+5.24) |
| 任意退化 | Ft 从未显著输 (> -0.1 dB) |

**规则**：永远先跑 Ft。它是最安全的基线，任何新策略必须先证明能超越 Ft。

### 2. Curric 方向 — 由 blur 子类型决定 ★★★★

| blur | 方向 | 效应 | 交叉验证 |
|------|:--:|:--:|:--:|
| motion | **Fwd** | -0.58~-1.80 | ✅ 换 noise 仍成立 |
| lens | **Rev** | +0.81~+1.75 | ✅ 换 noise 仍成立 |
| gaussian/jitter/zoom/glass | 无所谓 | <0.2 | — |

### 3. 低 LR 对噪声 Phase1 — 新发现 ★★★

**噪声 Phase1 用低 LR (5e-4) 一致优于默认 LR (1e-3)**

| 退化 | 低 LR 收益 |
|------|:--:|
| N4 noise(3)+blur(3), fwd | **+1.64** |
| N4 rev | **+2.32** |
| T2 三退化 fwd | +0.20 |
| S6 三退化 fwd/rev | +0.05~0.06 |

**规则**：Curric Phase1 如果是噪声→用 LR=5e-4。结构化 Phase1 不要调 LR。

### 4. DFPIR 对比：专攻小模型 > 通用大模型 ★★★

15 组退化，小模型(0.45M) 12/15 超越 DFPIR(31M)。G1 oversharpen 三退化 +6.50 dB。

---

## 二、部分有效的策略

### ADP (自适应 Phase 切换) ★★

- D3 最优参数 (T=0.005, P=5): **+0.79 dB** vs Fixed
- 但不稳定：rev 方向全崩，大多数参数组合无效
- **暂不推荐使用**，需要更多调优

### 动态 split (TA/SL/PR/MD) ★

- TA: T2 +2.91, N12 +3.97 (特定退化有效)
- 但无法复现：Phase 13 全部 ≤ Fixed
- **不推荐使用**

---

## 三、无效策略

### ✗ 类型感知 Loss (l1+edge/l1+fft/MSE/huber)
全部 ≤ l1，最大差距 -5.06 dB (huber on D2)

### ✗ FtCurr (盲预训练后课程学习)
11/12 差于 Ft。盲预训练通用性→课程特化破坏

### ✗ 多专家级联
从未超越单模型 Ft。误差级联放大

### ✗ 高 LR 对结构化 Phase1
全差 -0.7~-1.4 dB。结构化不需要高 LR

### ✗ 频谱峰值比 PR 分配
0/13 最优。过度偏好结构化退化

---

## 四、调优策略决策树

```
输入退化管线 P

Step 1: 基础策略
  → 默认用 Ft (EPOCH_BUDGET=2, LR=1e-3, LOSS_FN=l1)
  → 如果想尝试 Curric → Step 2

Step 2: Curric 方向 (只看 blur 子类型)
  ├── blur = motion  → Curric(fwd)
  ├── blur = lens    → Curric(rev)
  └── 其余 blur     → 方向无所谓

Step 3: Curric Phase1 超参
  ├── Phase1 = noise → LR = 5e-4 (低LR防过拟合)
  └── Phase1 = 其他  → LR = 1e-3 (默认)

Step 4: 不要做的事
  ├── 不要用 FtCurr (预训练+课程)
  ├── 不要调 Loss (l1 最优)
  ├── 不要用动态 split (Fixed 最稳健)
  └── 不要用多专家
```

## 五、关键数字

| 策略 | vs Direct 提升 | 可靠性 |
|------|:--:|:--:|
| Ft (安全默认) | +0.3~+5 dB | 30+/30+ |
| Curric(方向正确) | +0.5~+2 dB | 4/4 (motion/lens) |
| 低 LR (noise P1) | +0.2~+2.3 dB | 4/4 |
| ADP | +0.8 (best) | 不稳定 |
| Ft vs DFPIR(31M) | +1.7 avg | 12/15 |
