# 退化类型如何影响训练策略

> 基于 exp9 (~400组) + exp10 (60+组) + exp12 (56退化)
> 核心转变：从"什么退化用什么策略" → "什么机制决定什么策略"

---

## 一、三个核心机制

退化与训练策略的耦合由三个独立机制驱动：

```
机制 1: 梯度干扰 → Curric vs Direct
机制 2: Phase1 迁移价值 → Curric 方向 (⚠️ 未验证假设)
机制 3: Pretrain 通用性 → FtCurr 为什么崩溃 (互斥关系)
```

### 机制 1：梯度干扰决定 Direct 是否可用

多退化同时训练时，每个退化给模型发送不同方向的梯度更新。Curric 的作用不是"从易到难"，而是**减少训练早期的梯度方向数量**。

D3 (compression_jpeg(3)+blur_lens(4)) 是最强梯度干扰案例：Direct=20.99, Curric=25.26 (+4.27 dB)。

**关键发现**：梯度干扰存在，但**不保证 Curric > Ft**。D3 上 Ft=25.84, FtLR=25.89 均优于 Curric(25.26)。
唯一 Curric 可靠 > Ft 的场景：**motion P1 + sev ≥ 3** (4/4 案例验证)。
对于 compression+blur：推荐 **Ft** (最安全)，Curric 是备选但非必要。

### 机制 2：Phase1 迁移价值决定 Curric 方向 ⚠️ 未验证

> 状态：来源标记为 "❌ 未成立" (V1-V3 Fwd/Rev 差 < 0.7dB)

以下观察来自 Phase 1-8 实验，但**尚未在独立实验中验证**：

| 退化类型 | Phase1 学到什么 | 迁移价值 | 观察到的方向偏好 |
|----------|---------------|:--:|------|
| motion blur | 1D 线性反卷积 | 极高 | Fwd 优 (交叉验证通过) |
| compression | 结构化伪影识别 | 高 | 无明显方向偏好 |
| lens blur | 径向 PSF 反卷积 | 低 | Rev 优 (交叉验证通过) |

**已验证的规则**：
- motion blur 在管线中 → Fwd (交叉验证通过：换 noise 仍成立)
- lens blur 在管线中 → Rev (交叉验证通过：换 noise 仍成立)
- 两者都不在 → Fwd/Rev 差异 < 0.7 dB

### 机制 3：Pretrain 通用性 — FtCurr 为什么崩溃

盲预训练 + 课程学习 (FtCurr) **互斥**：预训练给的通用特征，Curric 的单退化阶段会把通用特征特化 → 不可逆 (11/12 崩溃)。决策规则：如果用 Ft，就不要加 Curric；如果用 Curric，就不要 Pretrain。

---

## 二、全局退化 vs 局部退化

**判定管线是否包含全局退化**：管线含以下函数 → 注意，盲预训练**可能**未充分覆盖：

`contrast_strengthen/weaken_scale/stretch`、`brightness_brighten/darken_shift/gamma_HSV/RGB`、
`saturate_strengthen/weaken_HSV/YCrCb`

| 属性 | 局部退化 (blur/noise/comp) | 全局退化 (contrast/brightness/saturation) |
|------|:--:|:--:|
| 盲预训练覆盖？ | ✅ 随机管线包含 | ⚠️ contrast_weaken 等特定类型未覆盖 |
| Ft 安全性 | ✅ 已验证安全 | ⚠️ 类型依赖：L2(contrast+noise)上 Ft 崩溃 -4.51 dB，但 L1(brightness+blur)上 Ft 优于 Direct +0.41 |
| 推荐策略 | Ft (EPOCH=2, LR=5e-4) | 含 contrast 时优先 Direct；含 brightness 时 Ft 可能安全 |

**注意**：Ft 对全局退化的安全性**依赖全局退化类型**。contrast_weaken 在盲预训练中未出现过，L2 上导致了 -4.51 dB 崩溃。但 brightness 类型 (L1) 上 Ft 表现正常 (+0.41)。

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

**Motion blur 的 sev 阈值**：sev ≤ 4 时 Fwd 仍有效；sev ≥ 5 时 Fwd 翻转（来源：sev=3 Fwd -0.58, sev=4 Fwd -0.48, sev=5 Rev +0.39）。

---

## 四、策略效应量级分层

```
梯度干扰     → 决定 Direct vs Curric       (0 ~ +4 dB)
迁移价值     → 决定 Curric 方向            (0 ~ +2 dB)
全局退化类型 → 决定 Ft vs Direct           (−4.5 ~ +5 dB)
修复难度差   → 决定是否需要动态 split      (0 ~ +4 dB, 异常值存在)
LR/loss 微调 → 影响最小                    (< 0.3 dB, loss 函数之间差异 < 0.25 dB)
```

---

## 五、架构-退化耦合表 (来源: exp10)

### 注意力类型选择

| 退化类型 | 最佳注意力 | Δ vs Swin | 备注 |
|---------|:--------:|:--:|------|
| motion 结构化模糊 | **OCAB** (空间) | +1.19 | ws=16 (+0.29); SwiGLU (+1.47); OCAB+ws16 (+1.72) |
| noise 随机噪声 | 任意 | ~0 | 噪声不关心注意力 |
| contrast 全局退化 | **Swin** | +9.78~10.38 | MDTA/OCAB 崩溃 (L2: Swin 28.70 vs MDTA 18.92, OCAB 18.32) |
| saturate 全局 | Swin | 基线 | 无明显胜者 |
| brightness+blur | Swin | 基线 | MDTA/OCAB 无优势 |

### 附加组件推荐

| 场景 | 推荐组件 | 参数 | 效果 | 风险 |
|------|------|:--:|:--:|------|
| contrast+结构化局部 | **FiLM-GCM** | +26K | L5 +3.52 | L1 后期发散但 best ckpt 可用 (-0.09~-0.30) |
| contrast+噪声 | **ColorPre** | +0.8K | L2 +0.81, L5 +2.56 | 跨退化泛化差 (见鲁棒性) |
| contrast+噪声 (轻量) | **CSN** | +1K | L5 +1.39 | 最轻量，最稳定 |
| 通用鲁棒 | **DualBranch** | +3K | L5 +3.27, L2 +0.78 | 鲁棒性排序第一 |
| 强空间结构 (L6/SF8) | **PCP shared** | +3K | L6 +4.21, SF8 +9.00 | 部分退化不如 Swin (SF9 -1.26) |
| brightness_HSV | ColorMLP | +0.1K | L1 +0.28 (修复 NaN) | L6 -0.52 |
| stretch | ChannelCurve | +0.1K | stretch +0.38 | gamma 无效 |
| 默认安全首选 | ColorPre | +0.8K | 全稳定 | 推荐首选 |

### 各退化类型最佳方案 (来源: exp10 Phase 1-4)

退化类型定义见 exp9/degradation/，下表汇总跨 8 组件的对比结果：

| 退化 | 管线 | 最佳组件 | Δ PSNR | 次选 |
|------|------|------|:--:|------|
| L1 | blur(3)+brightness_HSV(3) | ColorMLP | +0.28 | CSN -0.07 (稳定) |
| L2 | contrast(3)+noise_poisson(3) | ColorPre | +0.81 | FiLM +0.79 |
| L4 | brightness_HSV(3)+noise(3)+blur(3) | FiLM-GCM | -0.06 | 都无所谓 |
| L5 | contrast(3)+motion(5)+jpeg(1) | **FiLM-GCM** | **+3.52** | DualBranch +3.27 |
| L6 | contrast(3)+blur_gauss(3)+jpeg2000(1) | **PCP shared** | **+4.21** (vs CSN) | FiLM +0.22 |
| SF1 | contrast_pure(3) | ColorPre | +18.91 | DualBranch +13.36 |
| SF5 | saturate_HSV(3) | ColorPre | +0.03 | DualBranch -0.05 |
| SF7 | contrast(3)+saturate(5)+brightness_HSV(5) | Swin 基线 | — | PCP +0.04 (vs CSN) |
| SF8 | gamma(3)+saturate(3) | **PCP shared** | **+9.00** (vs CSN) | Swin 41.32 |
| SF9 | contrast(3)+saturate(3)+gamma(3) | Swin 基线 | 47.04 | PCP 45.78 |
| BS | brightness_shift_RGB(3) | ColorPre | +0.03 | 太简单，不需要 |
| bHSV | brightness_HSV(1) | PCP | +0.08 (vs Swin) | CSN 差 -3.62 |

### 组件详细验证结果 (来源: exp10 Phase 4)

**PCP shared (+3K) vs CSN (+1K) vs Swin 基线：**

| 退化 | Swin | CSN | PCP | PCP vs CSN |
|------|:--:|:--:|:--:|:--:|
| bHSV | 32.13 | 28.51 | **32.21** | +3.70 |
| L1 | — | 23.14 | **23.27** | +0.13 |
| L2 | — | 29.01 | **29.41** | +0.40 |
| L5 | — | 21.12 | **23.01** | +1.89 |
| L6 | — | 22.38 | **26.59** | +4.21 |
| SF8 | — | 32.64 | **41.64** | +9.00 |
| SF9 | **47.04** | 35.31 | 45.78 | +10.47 |

PCP 在 L6、SF8 上有异常大幅领先 CSN，但 SF9 反而不如 Swin 基线。场景依赖性极强。

**DualBranch (+3K) vs Swin 基线：**

| 退化 | Swin | DualBranch | Δ |
|------|:--:|:--:|:--:|
| L2 | 28.64 | 29.42 | +0.78 |
| L5 | 19.73 | 22.33 | +2.60 |
| SF1 | 27.15 | 40.51 | +13.36 |
| SF8 | 41.32 | 40.93 | -0.39 |

DualBranch 在 L5/SF1 上大幅超越 Swin，且无 NaN 风险，鲁棒性最好。

**FiLM-GCM (+26K) lr=5e-4 重跑：**

| 退化 | Swin (lr5e4) | FiLM (lr5e4) | Δ |
|------|:--:|:--:|:--:|
| L1 | 23.21 | 23.12 | -0.09 |
| L5 | 19.73 | 23.25 | **+3.52** |
| L6 | 26.98 | 27.20 | +0.22 |

lr=5e-4 修复了 L1 NaN 问题（lr=1e-3 时崩溃），但 L1 仍无收益。

### 参数公平性规则 (来源: exp10 原则 6)

- 模块总参数 ≤ 基线 × 1.05 (≤ 477K)
- 满足的模块：CSN (+1K), ColorPre (+0.8K), DualBranch (+3K), PCP-shared (+3K), ColorMLP (+0.1K), ChannelCurve (+0.1K)
- 超出需重新对齐的：FiLM-GCM (+26K, 5.7%) — 但 +3.52 dB 远超 5.7% 参数可解释范围

### 废弃组件

| 组件 | 原因 |
|------|------|
| FreqMod (+276K) | SF8 -9.67；参数膨胀 61%；bfloat16 不兼容 |
| PCP per-block (+91K) | 参数不公平 (+20%，超出 ≤1.05× 限制)，PCP shared 已替代 |

### 架构鲁棒性排序 (盲识别可能出错时，越低越好)

DualBranch (10.1 dB gap) > CSN (11.7) > Swin (13.2) > FiLM (13.6) > FreqMod (15.2) > ColorPre (16.0)

**关键警告**：ColorPre 在自身退化上最强 (+0.81~+18.91 dB)，但跨退化泛化最差——盲识别错误时退化可能完全失效。仅在盲识别置信度极高时使用。盲识别不确定时优先 **DualBranch** 或 **CSN**。

---

## 六、策略选择框架

```
输入退化管线 →

Step 1: 判断退化类型
  管线含 contrast_weaken/strengthen (仿射型全局)?
    → 是: 优先 Direct (不用 Ft, L2 上 Ft 崩溃 -4.51 dB)
  管线含 brightness/saturation 全局?
    → Ft 可能安全 (L1 上 Ft +0.41)，但仍推荐 Direct 保守
    → 架构: Swin (不用 MDTA/OCAB)

Step 2: 处理局部退化
  管线含 compression + blur?
    → 梯度干扰存在但 Ft 可处理 (Ft > Curric, +0.6 dB)
    → 推荐 Ft，Curric 为备选
  管线含 motion + 其他?
    → motion P1 + sev ≥ 3: Curric(Fwd) 可能 > Ft (4/4 案例)
    → 否则 Ft 即可

Step 3: 选择 Ft 还是从零训练
  局部退化 sev ≤ 3: Ft (优势 +0.3 ~ +5 dB)
  局部退化 sev ≥ 4: Ft ≈ Direct (选简单的，Ft 即可)
  全局退化: Direct (保守)，含 brightness 时可试 Ft

Step 4: 如果用 Curric，Phase1 选哪个退化
  motion blur → Fwd (已验证)
  lens blur → Rev (已验证)
  两者都不在 → Fwd/Rev 均可 (差异 < 0.7 dB, ⚠️ 未验证)

Step 5: 步数分配
  默认 Fixed 5000/5000/5094
  动态 split (MD/TA) 在特定退化有效 (T2 +2.91, N12 +3.97) 但非普适
```

---

## 七、实操速查表

| 场景 | 策略 | LR | 原因 |
|------|------|:--:|------|
| 不确定 | Ft | 5e-4 | 局部退化下已验证安全 |
| contrast 型全局退化 | **Direct** | 5e-4 | L2 上 Ft 崩溃 -4.51 |
| brightness 型全局退化 | Direct (保守) 或 Ft | 5e-4 | L1 上 Ft 安全 +0.41 |
| compression+blur | Ft | 5e-4 | Ft > Curric, 梯度干扰存在但 Ft 可处理 |
| motion+任意 (sev≥3) | Curric(Fwd) | 5e-4 | 唯一 Curric>Ft 的可靠场景 (4/4) |
| lens+任意 | Curric(Rev) 或 Ft | 5e-4 | lens 迁移价值低, Ft 也安全 |
| 严重度 ≥ 4 (局部) | Ft | 5e-4 | 预训练优势缩小但仍安全 |
| 严重度 ≤ 3 (局部) | Ft | 5e-4 | Ft 优势 +0.3~+5 dB |
| noise_impulse | 任意 | 5e-4 | 极易修，策略不重要 |
| 架构: 严重motion(sev≥5) | OCAB (+1.19) 或 OCAB+ws16 (+1.72) | — | SwiGLU 独立 +1.47, 无三合一组合实验 |
| 架构: contrast | Swin | — | 不用 MDTA/OCAB |
| 架构: contrast+结构化 | FiLM-GCM (lr=5e-4) | — | +3.52, L1 无收益但无 NaN |
| 架构: 盲识别不确定 | DualBranch 或 CSN | — | 鲁棒性最好，跨退化泛化强 |
| 架构: 默认安全首选 | ColorPre (+0.8K) | — | 全稳定，但跨退化泛化差 |
| 架构: 含 gamma/stretch | ChannelCurve or PCP | — | gamma 不需要额外组件，stretch +0.38 |
| 架构: brightness_HSV | ColorMLP (+0.1K) | — | 防 NaN，其他场景不用 |
