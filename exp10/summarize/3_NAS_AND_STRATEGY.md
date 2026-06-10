# NAS 探索与训练策略

> 合并自 NAS_RESULTS.md + FINETUNE_STRATEGY.md

---

# 退化-架构耦合实验总结

## 核心发现：退化-架构交互效应确认

**不同退化需要不同的注意力机制——架构选择不是普适的。**

| 退化 | Swin(基线) | MDTA(通道) | OCAB(空间) | 最优 | Δ |
|------|:--:|:--:|:--:|------|:--:|
| S5 motion结构化 | 20.51 | 21.34 | **21.70** | OCAB | +1.19 |
| N4 noise随机 | **27.83** | 27.76 | 27.75 | Swin | +0.08 |
| L2 contrast全局 | **24.21** | 18.92 | 18.32 | Swin | +5.90 |

### 交互效应：排名因退化而异 ✅

```
S5: OCAB > MDTA > Swin   (空间注意力对结构化退化最优)
N4: Swin ≈ MDTA ≈ OCAB   (噪声不关心注意力类型)
L2: Swin >> MDTA >> OCAB (MDTA/OCAB在全局退化上崩溃)
```

## 微架构消融结果

### 窗口大小

| 退化 | ws=4 | ws=8(基线) | ws=16 |
|------|:--:|:--:|:--:|
| S5 | — | 20.51 | **20.80** |
| N4 | — | 27.83 | 27.83 |
| D3 | — | **25.84** | 25.62 |

S5 受益于大窗口 (+0.29)，N4 无影响，D3 小窗口略好。

### SwiGLU 门控

| 退化 | SwiGLU | GELU(基线) | Δ |
|------|:--:|:--:|:--:|
| S5 | **21.98** | 20.51 | **+1.47** |
| M1 | 21.39 | 21.33 | +0.06 |
| T2 | 18.97 | 19.00 | -0.03 |

严重 motion blur (S5, sev=5) 极度受益于门控机制。

### S5 组合实验

| 配置 | PSNR | vs 基线 |
|------|:--:|:--:|
| Swin+GELU+ws=8 (基线) | 20.51 | — |
| +OCAB | 21.70 | +1.19 |
| +SwiGLU | 21.98 | +1.47 |
| +OCAB+SwiGLU | 21.75 | +1.24 |
| **+OCAB+ws=16** | **22.23** | **+1.72** |

OCAB+大窗口是最优组合。OCAB+SwiGLU 收益不叠加（部分重叠）。

## 意外发现：L2 全局退化上 MDTA/OCAB 崩溃

预测 MDTA 帮助全局退化——实际 **MDTA/OCAB 全崩 -5 dB**。

原因假设：MDTA (通道注意力) 和 OCAB (空间注意力) 都依赖 GDFN (门控深度卷积 FFN)。GDFN 的 3×3 深度卷积假设相邻像素有空间关联——这对于 contrast_weaken (全局像素级变换) 是错误的归纳偏置。Swin 的 MLP 没有这个假设，反而更灵活。

**教训：全局退化需要避免空间卷积偏置的架构组件。**

## 退化-架构耦合规律

| 退化特征 | 推荐架构 | 避免 |
|----------|----------|------|
| 严重结构化模糊 (motion sev≥5) | OCAB + ws=16 + SwiGLU | MDTA |
| 随机噪声 | Swin(默认), 架构不重要 | — |
| 全局退化 (contrast/brightness) | Swin(默认) | MDTA/OCAB/GDFN |
| 普通结构化退化 (motion sev≤4, compression) | OCAB | — |

## 下一步

1. **验证 MDTA/OCAB 在 L2 上的崩溃是否与 GDFN 有关** — 纯 MLP+MDTA 能否修复？
2. **在其他全局退化 (L3, L4) 上验证 Swin > MDTA/OCAB**
3. **OCAB+ws=16 在更多结构化退化上验证**
# 训练策略指南：基于 exp9 + exp10 的 500+ 组实验

## 目标

> 输入任意退化管线 → 输出最优训练策略 + 模型架构

本文档汇总 exp9（训练策略）和 exp10（架构设计）中关于训练策略的全部结论，形成可操作的决策规则。

## 一、四种训练策略

| 策略 | 方法 | 原理 |
|------|------|------|
| **Direct** | 在目标退化上从零训练 | 无预训练偏置 |
| **Ft** | 随机退化盲预训练 → 目标退化微调 | 通用特征 → 专用 |
| **Curric** | 渐进式多阶段（单退化→双退化→完整退化） | 难度递进 |
| **FtCurr** | 盲预训练 → 渐进微调 | Ft + Curric 组合 |

## 二、退化类型决定策略选择

### 2.1 局部退化（blur / noise / compression）

**首选 Ft。**

| 退化 | Direct | Ft | Curric | 最优 |
|------|------|------|------|------|
| motion blur | 基线 | ✅ 可靠 | fwd方向有效 | Ft 或 Curric(fwd) |
| gaussian blur | — | ✅ | — | Ft |
| lens blur | — | ✅ | rev方向有效 | Ft 或 Curric(rev) |
| gaussian noise | — | ✅ | — | Ft |
| impulse noise | — | ✅ | — | Ft |
| compression | — | ✅ | — | Ft |

**规则**：对于局部退化，Ft 从不显著落后。盲预训练的通用修复能力为专用微调提供了良好的初始化。

### 2.2 全局退化（contrast / brightness / saturation / gamma）

**首选 Direct。**

| 退化 | Direct | Ft | Curric | 最优 |
|------|------|------|------|------|
| contrast_scale | ✅ 稳定 | ❌ -4.51 (L2) | — | Direct |
| brightness_HSV | ✅ 稳定 | — | — | Direct |
| saturate_HSV | ✅ 稳定 | — | — | Direct |
| gamma | ✅ 稳定 | — | — | Direct |
| brightness_shift_RGB | ✅ 稳定 | — | — | Direct |

**规则**：Ft 在全局退化上崩溃。盲预训练注入的"局部修复"归纳偏置对全局退化产生负迁移。全局退化必须用 Direct 训练。

### 2.3 混合退化（局部 + 全局）

**规则**：退化管线中如果包含全局退化 → Direct。纯局部退化 → Ft。

## 三、策略选择的决策树

```
输入退化管线 P

1. P 包含全局退化（contrast/brightness/saturation/gamma）？
   ├─ 是 → Direct 训练
   └─ 否 → 进入第 2 步

2. P 是单步局部退化？
   ├─ 是 → Ft（盲预训练→微调）
   └─ 否 → 进入第 3 步

3. P 包含 motion blur 且 sev ≥ 3？
   ├─ 是 → Curric(fwd): P1=motion → P2=full
   └─ 否 → 进入第 4 步

4. P 包含 lens blur？
   ├─ 是 → Curric(rev): P1=lens → P2=full
   └─ 否 → Ft
```

## 四、严重度的影响

| 严重度 | Ft 效果 | 说明 |
|:--:|------|------|
| sev ≤ 2 | Ft 与 Direct 差异小 | 退化太弱，策略不重要 |
| sev 3-4 | Ft 优势最大 | 盲预训练提供有效初始化 |
| sev 5 | Ft 优势递减至 0 | 极端退化超出盲预训练分布 |

**规则**：sev ≤ 2 时不需要特殊策略。sev 3-4 是 Ft/Curric 的甜蜜点。sev=5 时 Direct 与 Ft 持平。

## 五、架构选择与策略的交互

| 架构组件 | 类型 | 策略 | 原因 |
|------|:--:|:--:|------|
| ColorPre / CSN / FiLM-GCM | 全局 | **Direct** | 全局架构模块必须在 Direct 下训练 |
| OCAB / 大窗口 | 局部 | Ft 或 Curric | 局部架构模块与 Ft 兼容 |
| Cascade CP→Swin | 分步 | Direct (1.5+0.5 ep) | 分步训练要求 Direct |

**规则**：架构选择和策略选择是**耦合的**——全局退化架构模块（ColorPre, FiLM-GCM）需要 Direct 训练，不能与 Ft 共用。

## 六、训练预算分配

### 6.1 标准预算

| 模式 | EPOCH_BUDGET | 适用 |
|------|:--:|------|
| Direct (标准) | 2 | 所有标准训练 |
| Ft (预训练) | 2 | 盲预训练阶段 |
| Ft (微调) | 2 | 微调阶段 |
| Curric (每阶段) | 等分 | 多阶段渐进 |
| Cascade Phase A | 1.5 | 空间专家 |
| Cascade Phase B | 0.5 | 颜色专家（822参数） |

### 6.2 动态预算（ADP）

ADP（自适应相位切换）在 fixed budget 内优化步数分配：

| 退化 | Fixed (1.5+0.5) | ADP (≤1.5+0.5) | Δ |
|------|:--:|:--:|:--:|
| L6 spatial Swin | 25.95 | 26.26 | +0.31 |
| L2 spatial Swin | 29.47 | 29.83 | +0.36 |
| L2 Cascade CP→Swin | 29.28 (+0.64) | 29.71 (+1.07) | +0.43 |

**规则**：在相同总预算下，ADP 优于固定分配。推荐使用 `ADAPTIVE_SWITCH=500, THRESHOLD=0.005`。

## 七、参数对齐原则

| 约束 | 值 |
|------|:--:|
| 基线参数 | EMBED_DIM=64, 454,531 params |
| 模块上限 | ≤ 477K（基线 × 1.05） |
| 总训练预算 | 2 epochs（所有对比） |

**规则**：所有架构对比必须在相同参数和训练预算下进行。超出预算的模块（FreqMod +276K, PCP-old +91K）直接废弃。

## 八、失败的策略（不应再尝试）

| 策略 | 失败原因 |
|------|------|
| **FtCurr** | 盲预训练通用性被单任务专用性破坏（12/12 失败） |
| **Ft 用于全局退化** | L2 崩溃 -4.51 dB |
| **Split-tuning (TA/SL/PR/MD)** | 无普适性，仅特定场景 4 WIN |
| **Loss-tuning (l1+edge/l1+fft/MSE/huber)** | 对全局退化无显著影响 |
| **Multi-expert cascade (旧版)** | 未解耦梯度 |

## 九、最终策略推荐

```
输入: 退化管线 P + 预算 B

Step 1: 判断退化类型
  has_global = any(op in {contrast, brightness, saturation, gamma} for op in P)
  is_structured = has(motion_blur) or has(lens_blur)

Step 2: 选择训练策略
  if has_global:    strategy = Direct
  elif is_structured: strategy = Curric(fwd if motion else rev)
  else:              strategy = Ft

Step 3: 分配预算
  if strategy == Direct and has_global:
    if use_cascade: B_phaseA = 1.5, B_phaseB = 0.5 (with ADP)
    else:           B = 2

Step 4: 选择架构（参考 ARCHITECTURE_CATALOG.md）
  if has(contrast) and is_structured:  FiLM-GCM 或 Cascade CP→Swin
  elif has(contrast):                  ColorPre
  elif has(HSV):                       ColorMLP (input) 或 ColorPre
  else:                                Swin (默认)

输出: (strategy, architecture, budget)
```

## 十、相关文档

| 文档 | 内容 |
|------|------|
| [FINAL_SUMMARY.md](FINAL_SUMMARY.md) | 全域退化架构实验 7 轮 130+ 组总结 |
| [CONSOLIDATED_RESULTS.md](CONSOLIDATED_RESULTS.md) | 全部实验结果按退化类型组织 |
| [ARCHITECTURE_CATALOG.md](ARCHITECTURE_CATALOG.md) | 所有架构组件目录 |
| [exp9/summarize/](../exp9/summarize/) | 训练策略 450+ 组实验 |
