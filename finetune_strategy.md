# 退化类型如何影响训练策略

> 基于 exp9 (~400组) + exp10 (60+组) + exp12 (56退化 × 多策略)
> 核心转变：从"什么退化用什么策略" → "什么机制决定什么策略"

---

## 一、三个核心机制

退化与训练策略的耦合由三个独立机制驱动：

```
机制 1: 梯度干扰 → Direct vs Curric
机制 2: Phase1 迁移价值 → Curric 方向 (Fwd/Rev)
机制 3: Pretrain 通用性 → Ft vs Direct vs FtCurr
```

### 机制 1：梯度干扰决定 Direct 是否可用

多退化同时训练时，每个退化给模型发送不同方向的梯度：

```
去模糊: 梯度要求参数 → "让边缘更锐利"
去噪声: 梯度要求参数 → "让平坦区域更平滑"
去压缩: 梯度要求参数 → "消除 8×8 块边界"

三个梯度方向不同 → 参数被三个信号撕扯 → 训练困难
```

Curric 的作用不是"从易到难"，而是**减少训练早期的梯度方向数量**：
- Phase 1 只有 1 个退化 → 梯度方向纯粹
- Phase 2 有 2 个退化 → 已有基础，干扰减半
- Phase 3 有 3 个退化 → 已是"叠加"而非"撕扯"

**最强梯度干扰案例**：compression + blur。D3 (comp+blur) Direct=20.99, Curric=25.26 (+4.27 dB)。其他退化对的梯度干扰 < 0.3 dB。

### 机制 2：Phase1 迁移价值决定 Curric 方向

给定 N 步管线，Curric Phase1 应该学**迁移价值最高**的那个退化：

| 退化类型 | Phase1 学到什么 | 迁移价值 | 方向影响 |
|----------|---------------|:--:|------|
| motion blur | 1D 线性反卷积 | **极高** | Fwd 始终优于 Rev |
| compression | 结构化伪影识别 | **高** | Rev 中当 Phase1 也有效 |
| gaussian noise | 信号/噪声分离 | 高 | 方向无关 |
| lens blur | 径向 PSF 反卷积 | **低** | Rev 始终优于 Fwd |
| impulse noise | 离群值检测 | 低 | 太特殊，难迁移 |

**决策规则**：
- motion blur 在管线中 → Fwd（motion 的 1D 反卷积是所有修复的基础）
- lens blur 在管线中 → Rev（lens 的径向 PSF 太专一，换 compression 当 Phase1）
- 两者都不在 → Fwd/Rev 均可

### 机制 3：Pretrain 通用性决定 Ft 是否有效

盲预训练（随机退化管线）给的是通用特征。微调 (Ft) 把通用特征特化到目标退化。

- **局部退化 (blur/noise/compression)**：盲预训练已覆盖 → Ft 安全且有效
- **全局退化 (brightness/contrast/saturation)**：盲预训练从未覆盖 → Ft 可能崩溃（L2 -4.51 dB）

---

## 二、全局退化 vs 局部退化 ⚠️ 最关键区分

| 属性 | 局部退化 (blur/noise/comp) | 全局退化 (brightness/contrast/saturation/gamma) |
|------|:--:|:--:|
| 盲预训练覆盖？ | ✅ 随机管线包含 | ❌ 从未训练 |
| Ft 安全？ | ✅ 是 | ❌ 否 (L2 Ft-Direct = -4.51 dB) |
| Curric 有效？ | 有时 | 未知 |
| 推荐策略 | Ft (EPOCH=2, LR=5e-4) | **Direct** (EPOCH=2, LR=5e-4) |
| 最佳架构 | Swin (基线) | Swin (MDTA/OCAB 崩溃) |

**判定规则**：管线包含以下任一函数 → 全局退化，用 Direct 不用 Ft：
`contrast_strengthen/weaken_scale/stretch`、`brightness_brighten/darken_shift/gamma_HSV/RGB`、
`saturate_strengthen/weaken_HSV/YCrCb`、`oversharpen`

---

## 三、严重度：调节"迁移价值能否在预算内兑现"

迁移价值高 ≠ Phase1 一定能学好。严重度决定能否在预算内学好：

```
motion sev=3: 5000步 ✅ 能学好1D反卷积 → 迁移价值兑现 → Fwd 优
motion sev=5: 5000步 ❌ 学不好 → 迁移价值无法兑现 → 换 Rev
lens sev=3:   5000步 ❌ 学不好径向反卷积 → 从未兑现 → 始终 Rev
```

**修复难度 ≠ 破坏程度**：
- noise_impulse(4): damage PSNR=16.7（极重），expert PSNR=43.1（极易修）
- blur_lens(3): damage PSNR=21.5（较轻），expert PSNR=24.5（难修）

---

## 四、策略效应量级分层

```
梯度干扰     → 决定 Direct vs Curric       (0 ~ +5 dB)
迁移价值     → 决定 Curric 方向            (0 ~ +2 dB)
全局退化     → 决定 Ft vs Direct           (−4.5 ~ +5 dB)
修复难度差   → 决定是否需要动态 split      (0 ~ +0.5 dB，有时无效)
LR/loss 微调 → 影响最小                    (< 0.3 dB)
```

---

## 五、架构-退化耦合表 (来源: exp10)

### 注意力类型选择

| 退化类型 | 最佳注意力 | Δ vs Swin | 备注 |
|---------|:--------:|:--:|------|
| motion 结构化模糊 | **OCAB** (空间) | +1.19 | ws=16 配合使用 |
| noise 随机噪声 | 任意 | ~0 | 噪声不关心注意力 |
| contrast 全局退化 | **Swin** | +5.90 | MDTA/OCAB 崩溃 |
| saturate 全局 | Swin | 基线 | 无明显胜者 |
| brightness+blur | Swin | 基线 | MDTA/OCAB 无优势 |

### 附加组件推荐

| 场景 | 推荐组件 | 参数 | 效果 | 风险 |
|------|------|:--:|:--:|------|
| contrast+结构化局部 | **FiLM-GCM** | +26K | L5 +3.52 | L1 NaN (HSV极端值) |
| contrast 通用 | **ColorPre** | +0.8K | L2 +0.81, 最稳 | 跨退化泛化差 |
| contrast+噪声 | **CSN** | +1K | L5 +1.39 | 最轻量 |
| brightness 系列 | ColorMLP | +0.1K | 修复 NaN | 特定场景 |
| 默认安全 | ColorPre | +0.8K | 全稳定 | 推荐首选 |

### 废弃组件

| 组件 | 原因 |
|------|------|
| FreqMod (+276K) | SF8 -9.67；参数膨胀 61%；bfloat16 不兼容 |
| PCP per-block (+91K) | 参数不公平 (+20%) |
| PCP shared (+3K) | 不如 CSN |

### 架构鲁棒性排序 (盲识别可能出错时)

DualBranch (10.1 dB gap) > CSN (11.7) > Swin (13.2) > FiLM (13.6) > FreqMod (15.2) > ColorPre (16.0)

ColorPre 在自身退化上最强 (+0.81~+18.91 dB)，但跨退化泛化最差——仅在盲识别置信度极高时使用。

---

## 六、策略选择框架

```
输入退化管线 →

Step 1: 判断退化类型
  管线含 contrast/brightness/saturation/gamma/oversharpen?
    → 是: 全局退化，用 Direct + Swin (不用 Ft, 不用 MDTA/OCAB)
    → 否: 局部退化，继续 Step 2

Step 2: 评估梯度干扰
  管线含 compression + blur?
    → 是: 强干扰，需要 Curric (Direct → Curric +4.09 dB)
    → 否: 干扰弱，Direct 或 Ft 即可

Step 3: 选择 Ft 还是从零训练
  sev ≤ 3: Ft (优势 +0.3 ~ +5 dB)
  sev ≥ 4: Ft ≈ Direct (选简单的，Ft 即可)
  用 Curric: 不要预训练 (原则 4)

Step 4: 如果用 Curric，Phase1 选哪个退化
  motion blur 在管线 → Fwd
  lens blur 在管线 → Rev
  两者都不在 → Fwd/Rev 均可 (差异 < 0.7 dB)

Step 5: 步数分配
  默认 Fixed 5000/5000/5094
  仅在明确瓶颈 + 上升区退化时考虑动态 split (MD)
```

---

## 七、实操速查表

| 场景 | 策略 | LR | 原因 |
|------|------|:--:|------|
| 不确定 | Ft | 5e-4 | 永远不会显著差 (局部退化) |
| 全局退化 | **Direct** | 5e-4 | Ft 可能崩溃 -4.51 dB |
| compression+blur | Curric(Rev) | 5e-4 | 强梯度干扰 +4.09 dB |
| motion+任意 | Curric(Fwd) | 5e-4 | motion 迁移价值极高 |
| lens+任意 | Curric(Rev) | 5e-4 | lens 迁移价值低 |
| 严重度 ≥ 4 | Ft | 5e-4 | 预训练优势消失，但 Ft 仍安全 |
| 严重度 ≤ 3 | Ft | 5e-4 | Ft 优势 +0.3~+5 dB |
| noise_impulse | 任意 | 5e-4 | 极易修，策略不重要 |
| 极轻微退化 | Spec | 5e-4 | Ft 特征"过强"可能有害 |
| 架构选择 | 严重motion→OCAB+ws16 | — | contrast→Swin 不用 MDTA/OCAB |
