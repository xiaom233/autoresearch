# 退化类型与训练策略耦合 —— 最终实验总结

> ~500 组实验，EMBED_DIM=64 (0.45M)，EPOCH_BUDGET=2 (15094步)

---

## 一、核心结论：三个铁律 + 一个警示

### 铁律 1：Ft 是局部退化的安全默认 ★★★★★

30+ 组局部退化（blur/noise/compression），Ft 从未显著输给任何策略。最大优势 +5.24 dB (N4 noise+blur)。

### 铁律 2：Curric 方向由 blur 子类型决定 ★★★★

| blur | 方向 | 交叉验证 |
|------|:--:|:--:|
| motion | Fwd | ✅ 换 noise 仍成立 |
| lens | Rev | ✅ 换 noise 仍成立 |
| 其他 | 无所谓 | — |

### 铁律 3：低 LR 对噪声 Phase1 一致有效 ★★★

| 退化 | 低 LR 收益 |
|------|:--:|
| N4 (noise+blur) | +1.64 / +2.32 |
| T2 (三退化) | +0.20 |

### ⚠️ 警示：全局退化不遵循局部退化的规律

| 退化 | Ft-Direct | 规律 |
|------|:--:|------|
| L1 (blur+brightness) | +0.41 | Ft 安全 ✅ |
| L2 (noise+contrast) | **-4.51** | Ft 崩溃 ❌ |
| L3 (JPEG+brightness) | **-1.76** | Ft 差 ❌ |
| L4 (blur+noise+brightness) | **-0.75** | Ft 差 ❌ |

**全局退化（contrast/brightness）Ft 不安全——盲预训练未覆盖这些退化，产生负迁移。**

---

## 二、有效策略（按效应量排序）

| 策略 | 效应量级 | 适用条件 | 证据强度 |
|------|:--:|------|:--:|
| Ft | +0.3~+5 dB | 局部退化, sev≤4 | 30+/30+ |
| Curric方向 | ±2 dB | motion/lens blur | 4/4 交叉验证 |
| 低LR (noise P1) | +1.6~+2.3 dB | 噪声作为Curric Phase1 | 4/4 |
| ADP | +0.8 (best) | D3, 不稳定 | 2/18 有效 |
| Direct | — | 全局退化(contrast), sev≥5 | 特定case |

---

## 三、无效策略

| 策略 | 结论 | 证据 |
|------|------|------|
| FtCurr (预训练+课程) | 几乎总是最差 | 11/12 崩 |
| 类型感知 Loss | 全差于 l1 | D3 l1+fft -3.17 |
| 动态 split (TA/SL/PR/MD) | 无法超越 Fixed | 四种方案全失败 |
| 高 LR (structured P1) | 有害 | -0.7~-1.4 dB |
| 多专家级联 | 从未超越 Ft | 15 专家 |

---

## 四、全局退化 vs 局部退化

这是最重要的新发现：

| 属性 | 局部退化 (blur/noise/comp) | 全局退化 (brightness/contrast) |
|------|:--:|:--:|
| Ft 是否安全？ | ✅ 是 | ❌ 否 (L2 -4.51) |
| Curric 是否有效？ | 有时 | 待验证 |
| 盲预训练覆盖？ | ✅ | ❌ (从未训练过) |
| 与局部退化组合？ | — | 交互模式完全不同 |

**全局退化需要独立的策略体系，不能套用局部退化的经验。**

---

## 五、DFPIR 对比

小模型(0.45M) 12/15 超越 DFPIR(31M)。平均优势 +1.7 dB。G1 oversharpen +6.50 dB。

---

## 六、调优决策

```
输入退化 → 判断类型
  ├── 局部退化 (blur/noise/compression)
  │   ├── sev ≤ 3 → Ft (安全默认)
  │   ├── sev ≥ 4 → Direct 或 Ft (差异小)
  │   ├── 用 Curric? → blur=motion→Fwd, lens→Rev
  │   └── Phase1=noise? → LR=5e-4
  │
  └── 全局退化 (brightness/contrast/saturation)
      ├── Ft 不安全! → 先跑 Direct
      └── 规律待建立 (需要更多实验)
```

## 七、待验证

- L5-L6 全局退化实验 (在跑)
- NAS Round 1: 退化-注意力耦合验证 (待启动)
- 全局退化的系统策略研究 (需要大规模实验)
