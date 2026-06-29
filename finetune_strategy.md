# 退化类型如何影响训练策略

> 基于 exp9 (~400组) + exp10 (60+组) + exp12 (56退化) + **exp37 (24消融)**
> 核心转变：从"什么退化用什么策略" → "什么机制决定什么策略"
> **🔴 exp37 修正**: exp9/10 中的 "Ft" 实际是 Random Curriculum (AR_CURRICULUM_CONFIG)，不是 Checkpoint Fine-Tune (AR_LOAD_CKPT)。下文已全面修正术语。

**⚠️ 防止策略退化**: 本文档的目标是**精准匹配**——在什么条件下用什么策略。
exp37 的教训不是"组件无效"，而是"组件在盲识别不自信时盲目使用会适得其反"。
exp10 已证明组件在匹配正确的退化上收益巨大 (FiLM-GCM +3.52, ColorPre +18.91)。
**正确做法**: 自信匹配 → 用组件；不确定 → 保守。一刀切用 Direct+Swin 等于放弃优化。

---

## 一、三个核心机制

退化与训练策略的耦合由三个独立机制驱动：

- **机制 1**: 梯度干扰 → Curric vs Direct
- **机制 2**: Phase1 迁移价值 → Curric 方向 (⚠️ 未验证假设)
- **机制 3**: Pretrain 通用性 → Curric(ckpt) 为什么崩溃 (互斥关系)

### 机制 1：梯度干扰决定 Direct 是否可用

多退化同时训练时，每个退化给模型发送不同方向的梯度更新。Curric 的作用不是"从易到难"，而是**减少训练早期的梯度方向数量**。

D3 (compression_jpeg(3)+blur_lens(4)) 是最强梯度干扰案例：Direct=20.99, Curric=25.26 (+4.27 dB)。

**关键发现**：梯度干扰存在，但**不保证 Curric > RandomCurric**。D3 上 RandomCurric=25.84, FtLR(Curric+低LR)=25.89 均优于 Curric(25.26)。
唯一 Curric 可靠 > RandomCurric 的场景：**motion P1 + sev ≥ 3** (4/4 案例验证)。
对于 compression+blur：推荐 **RandomCurric** (最安全)，Curric 是备选但非必要。

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

### 机制 3：Pretrain 通用性 — Curric(ckpt) 为什么崩溃

盲预训练 + 课程学习 (Curric+ckpt) **互斥**：预训练给的通用特征，Curric 的单退化阶段会把通用特征特化 → 不可逆 (11/12 崩溃)。决策规则：如果用盲预训练 ckpt，就不要加 Curric；如果用 Curric，就不要 Pretrain。

---

## 二、全局退化 vs 局部退化 + 🔴 exp37 真实验证

**判定管线是否包含全局退化**：管线含以下函数 → 注意，盲预训练**可能**未充分覆盖：

`contrast_strengthen/weaken_scale/stretch`、`brightness_brighten/darken_shift/gamma_HSV/RGB`、
`saturate_strengthen/weaken_HSV/YCrCb`

### Random Curriculum (exp9/10, 旧称 "Ft")

| 属性 | 局部退化 (blur/noise/comp) | 全局退化 (contrast/brightness/saturation) |
|------|:--:|:--:|
| 盲预训练覆盖？ | ✅ 随机管线包含 | ⚠️ contrast_weaken 等特定类型未覆盖 |
| RandomCurric 安全性 | ✅ 已验证 (+0.1~+5 dB) | ⚠️ L2(contrast+noise)上崩溃 -4.51 dB |
| 推荐策略 | RandomCurric | 含 contrast 时 Direct |

### 各全局退化类型的差异化处理

**为什么只有 contrast 被特殊处理？** 因为不同的全局退化对策略和架构的敏感度完全不同：

| 全局退化 | RandomCurric | 敏感组件 | 最佳实践 | 原因 |
|------|:--:|------|------|------|
| **contrast** (scale/stretch) | ❌ 崩溃 -4.51 | **FiLM-GCM +3.52, ColorPre +18.91** | Direct + 组件(自信时) | 仿射变换改变方差, 梯度方向冲突最严重 |
| **brightness** (shift) | ✅ +0.41 | ColorMLP +0.28 | Direct 即可 | 加性变换, 梯度干扰小, MLP可处理 |
| **brightness** (gamma) | ⚠️ 未测试 | 无 | Direct, severity 必须精确 | gamma sev 偏差代价 -17 dB, 但组件无效 |
| **saturation** | ⚠️ 未测试 | 无 (全部 ≤ 基线) | Direct + 纯 Swin | 所有组件均有 ≤ 0 dB |
| **quantization** (median/hist) | ⚠️ 未测试 | 无 | Direct + 纯 Swin | 信息损失不可逆, 组件无法补偿 |

> 核心规律：**contrast 是唯一同时满足"策略受限 (必须 Direct)"和"组件敏感 (FiLM/ColorPre 大幅有效)"的全局退化。**
> 其他全局退化要么策略灵活 (brightness)，要么组件无效 (saturation/quantization/gamma)。

> **术语说明**: exp9/10 中的 "Ft" (D1f, D2f...) 实际是 `AR_CURRICULUM_CONFIG=0:random,...` — 前 50% 步数随机单退化，后 50% 目标退化。这是 **Random Curriculum（随机课程学习）**，不是 Checkpoint Fine-Tune。

### 🔴 True Checkpoint Fine-Tune (exp37, AR_LOAD_CKPT)

exp37 首次测试真正的盲预训练 checkpoint 微调 (AR_LOAD_CKPT=blind_pretrain_step15092.pt)，24 挑战 R0 消融结果：

| 退化类型 | Ft (ckpt微调) | Direct (从头) | Δ | 结论 |
|------|:--:|:--:|:--:|------|
| 纯局部 (14) | 24.55 | 24.60 | -0.05 | **持平** |
| 混合 (10) | 20.39 | 20.42 | -0.03 | **持平** |
| 全部 (24) | 22.82 | 22.86 | -0.04 | **持平** |

**结论: 1.5h 预算下，真正的 checkpoint fine-tune 与 Direct 训练无统计差异。盲预训练特征迁移在有限预算下无优势。**

对比 Random Curriculum (exp9): +0.1~+5 dB 来自**梯度干扰减少**（先单退化再组合），而非预训练特征迁移。

---

## 三、严重度：调节"迁移价值能否在预算内兑现"

迁移价值高 ≠ Phase1 一定能学好。严重度决定能否在预算内学好：

- motion sev=3: 5000步 ✅ 能学好1D反卷积 → 迁移价值兑现 → Fwd 优
- motion sev=5: 5000步 ❌ 学不好 → 迁移价值无法兑现 → 换 Rev
- lens sev=3: 5000步 ❌ 学不好径向反卷积 → 从未兑现 → 始终 Rev

**修复难度 ≠ 破坏程度**：
- noise_impulse(4): damage PSNR=16.7（极重），expert PSNR=43.1（极易修）
- blur_lens(3): damage PSNR=21.5（较轻），expert PSNR=24.5（难修）

**Motion blur 的 sev 阈值**：sev ≤ 4 时 Fwd 仍有效；sev ≥ 5 时 Fwd 翻转（来源：sev=3 Fwd -0.58, sev=4 Fwd -0.48, sev=5 Rev +0.39）。

---

## 四、策略效应量级分层

| 因素 | 决定 | 量级 | 来源 |
|------|------|:--:|------|
| 梯度干扰 | Direct vs Curric | 0 ~ +4 dB | exp9 |
| 迁移价值 | Curric 方向 | 0 ~ +2 dB | exp9 |
| 全局退化类型 | RandomCurric vs Direct | -4.5 ~ +5 dB | exp9 |
| 🔴 True Ft | ckpt微调 vs Direct | ~0 dB (1.5h) | exp37 |
| 修复难度差 | 是否需要动态 split | 0 ~ +4 dB | exp9 |
| LR/loss 微调 | 影响最小 | < 0.3 dB | exp9 |

---

## 五、架构-退化耦合表 (来源: exp10, 130+组, 全部 Direct 训练从零开始, GT退化已知)

> 🔴 **重要**: 以下所有组件收益均在 **GT 退化已知** 的对照实验中测得。
> exp37 证明：盲识别不完美时，这些组件在错误退化上训练反而有害。
> **实用建议：除非盲识别 100% 准确，否则纯 Swin 是最安全的选择。**
> 此节保留作为架构设计的理论参考。

### 🔴 核心原则 0：退化难度决定组件价值，而非退化类型

不是"退化类型 X → 组件 Y"，而是"当修正复杂度超出网络隐式容量时，显式组件才有价值"。
⚠️ 前提: GT 退化已知。盲识别场景中纯 Swin 可能更优 (exp37)。

| 退化难度 | 示例 | 网络能隐式学到? | 显式组件价值 | 推荐 |
|:--:|------|:--:|:--:|------|
| 低 | gamma, brightness_shift_RGB | ✅ | ~0 dB | 纯 Swin，不加组件 |
| 中 | stretch, contrast_scale, contrast+噪声 | ❌ 部分 | +0.4~+1 dB | ColorPre/CSN (+0.8~1K) |
| 高 | contrast + 结构化局部 (motion/jpeg) | ❌ | **+2.4~+3.5 dB** | FiLM-GCM (+26K) |

**exp35 验证**: blind_0002 (JPEG sev=1) 纯 Swin 即可 PSNR=28.8，加组件无益。blind_0010 (blur+contrast) PSNR=18.0，属于高难度，FiLM-GCM 可能有 +2-3 dB 额外收益（未验证）。

### 注意力类型选择

| 退化类型 | 最佳注意力 | Δ vs Swin | 备注 |
|---------|:--------:|:--:|------|
| motion 结构化模糊 | **OCAB** (空间) | +1.19 | ws=16 (+0.29); SwiGLU (+1.47); OCAB+ws16 (+1.72) |
| noise 随机噪声 | 任意 | ~0 | 噪声不关心注意力 |
| contrast 全局退化 | **Swin** | +9.78~10.38 | MDTA/OCAB 崩溃 (L2: Swin 28.70 vs MDTA 18.92, OCAB 18.32) |
| saturate 全局 | Swin | 基线 | 无明显胜者 |
| brightness+blur | Swin | 基线 | MDTA/OCAB 无优势 |

### 附加组件推荐 (排名来自 exp10 60+组)

| 排名 | 场景 | 推荐组件 | 参数 | 最大收益 | 风险 | 推荐 |
|:--:|------|------|:--:|:--:|------|:--:|
| 1 | contrast+结构化局部 | **FiLM-GCM** | +26K | L5 **+3.52** | L1 NaN (HSV+brightness), 需 lr=5e-4 | 有条件 |
| 2 | contrast+噪声/全局 | **ColorPre** | +0.8K | L2 +0.81, SF1 +18.91 | 跨退化泛化差 (盲识别错误时退化严重) | ✅ 推荐 |
| 3 | 轻量通用 | **CSN** | +1K | L5 +1.39 | **L6 -4.60 (纯全局退化灾难)** | 有条件 |
| 4 | 鲁棒泛化 | **DualBranch** | +3K | L5 +2.60, L2 +0.78 | 无 NaN，最稳定 | 备选(盲识别不确定时) |
| 5 | brightness_HSV | ColorMLP | +0.1K | L1 +0.28 (修复NaN) | L6 -0.52 | 特定场景 |
| 6 | stretch | ChannelCurve | +0.1K | stretch +0.38 | gamma 无效 | 有限 |
| — | PCP (per-block +91K) | — | +91K | — | 参数不公平 (20%) | ❌ 废弃 |
| — | FreqMod | — | +276K | — | SF8 -9.67, 6/7退化≤基线 | ❌ 废弃 |

**⚠️ CSN 警告**: 在纯全局退化（SF9 saturate_pure）上 -11.73 dB，在 L6 (saturate+impulse+lens) 上 -4.60 dB。仅在退化包含噪声或模糊等空间结构时安全。

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

### 架构鲁棒性排序 (跨退化泛化能力)

盲识别可能出错时，组件在错误退化上的 PSNR 下降幅度（越低越好）：

| 组件 | PSNR gap | 解释 |
|------|:--:|------|
| DualBranch | 10.1 dB | 最鲁棒（双重分支提供退化无关通路） |
| CSN | 11.7 | 轻量泛化好，但纯全局退化上灾难 |
| Swin | 13.2 | 无额外组件的基线鲁棒性 |
| FiLM-GCM | 13.6 | 强但泛化一般 |
| ColorPre | 16.0 | 自身退化最强，跨退化最差 |

**关键警告**：
- **ColorPre**: 在自身退化上最强 (+0.81~+18.91 dB)，但跨退化泛化最差。仅在盲识别置信度极高时使用。
- **CSN**: 泛化好但纯全局退化上有灾难性失败（L6 -4.60, SF9 -11.73）。确保退化包含空间结构。
- **盲识别不确定时优先 DualBranch**（兼顾鲁棒性和稳定性）。

---

## 六、策略选择框架 🔴 exp37修正

输入退化管线，按以下流程决策：

**Step 0: 判断退化难度（决定是否需要额外架构组件）**

| 难度 | 条件 | 推荐架构 | 说明 |
|------|------|------|------|
| 低 | gamma, shift_RGB, JPEG sev≤2 | 纯 Swin | 组件收益 ≈ 0 dB |
| 中 | stretch, contrast+噪声 | ColorPre 或 CSN | +0.8~1K, 轻量稳定 |
| 高 | contrast+结构化(motion/jpeg) | FiLM-GCM | +26K, 需 lr=5e-4 防 NaN |

> 🔴 exp37: 以上组件收益在盲识别场景中未复现。仅在 GT 退化已知时有效。

**Step 1: 判断训练策略 (Direct vs RandomCurric vs True Ft)**

| 管线特征 | 策略 | 原因 |
|------|------|------|
| 含 contrast_weaken/strengthen | Direct | RandomCurric 崩溃 -4.51 dB |
| 多步退化 (≥2步) 无 contrast | RandomCurric | 梯度干扰减少 +0.1~+5 dB |
| 纯局部 (blur/noise/comp) | Direct | True Ft ≈ Direct (exp37) |
| True Ft (AR_LOAD_CKPT) | 不推荐 | 1.5h 预算下无显著优势 |

**Step 2: 选择附加组件（与盲识别自信度挂钩）**

| 盲识别 + 退化特征 | 组件 | 来源 |
|------|------|------|
| 高难度 + contrast + 结构化 | FiLM-GCM (lr=5e-4) | exp10 L5 +3.52 |
| 中难度 + 全局退化 | ColorPre | exp10 L2 +0.81 |
| 盲识别 UNCERTAIN | **纯 Swin + Direct** | exp37 验证 |
| brightness_HSV 有 NaN 风险 | +ColorMLP | +0.1K |
| 纯局部退化 | 纯 Swin + Direct | — |

**Step 3: 组件稳定性检查**

| 组合 | 风险 | 对策 |
|------|------|------|
| FiLM-GCM + brightness_HSV | NaN | 降 lr=5e-4 或换 ColorPre |
| CSN + 纯全局退化 | 灾难性失败 (-4.6~-11.7 dB) | 换 ColorPre 或 Swin |
| 盲识别场景 | 组件不匹配 | 不加组件，纯 Swin 最安全 |

**Step 4: 处理局部退化的训练顺序**

| 管线特征 | 策略 | 收益 |
|------|------|:--:|
| compression + blur | RandomCurric | +0.6 dB |
| motion + 其他 | Curric(Fwd), P1=motion sev≥3 | 4/4 案例有效 |

**Step 5: Curric Phase1 退化选择**

| 退化 | 方向 | 状态 |
|------|:--:|------|
| motion blur | Fwd | ✅ 已验证 |
| lens blur | Rev | ✅ 已验证 |

---

## 七、实操速查表 🔴 exp37修正

> 🔴 盲识别总是不完美的。组件选择应**与盲识别对该退化类型的自信程度挂钩**，
> 而非一刀切。

### 核心原则: 组件只在盲识别自信时使用

- **LIKELY/GOOD + 退化匹配组件专长** → ✅ 加组件 (exp10 收益可兑现)
- **UNCERTAIN/POOR** → 纯 Swin + Direct (保守, exp37验证)
- **自信但退化不匹配组件专长** → 纯 Swin (加错组件反而有害)

| 盲识别自信度 | 匹配的退化 | 推荐组件 | 预期收益 | 来源 |
|------|------|------|:--:|------|
| **LIKELY + contrast+结构化** | motion/jpeg+contrast | FiLM-GCM | +3.52 | exp10 L5 |
| **LIKELY + 纯 contrast** | contrast_scale/stretch | ColorPre | +18.91 | exp10 SF1 |
| **LIKELY + 纯局部多步** | blur+noise+comp | RandomCurric | +0.1~+5 | exp9 D1-D3 |
| **LIKELY + motion sev≥5** | motion blur | OCAB+ws16 | +1.72 | exp10 |
| **UNCERTAIN / POOR** | 任何 | **纯 Swin + Direct** | 基线 | exp37 |

### 反向规则: 什么时候不加组件

| 盲识别状态 | 不加组件的原因 |
|------|------|
| UNCERTAIN (盲识别不确定) | 组件可能在错误退化上训练, exp37 14连败 |
| 退化简单 (gamma/shift/JPEG sev≤2) | MLP 能隐式学到, 组件收益≈0 |
| 组件与退化不匹配 | 如 FiLM-GCM 对纯 noise 无效 |
| ColorPre + 盲识别不确定 | 跨退化泛化最差 (PSNR gap 16.0 dB) |

**⚠️ 组件风险速查**:
- FiLM-GCM: brightness_HSV 上 NaN → lr=5e-4 修复
- CSN: 纯全局退化上灾难 (-4.60~-11.73 dB)
- ColorPre: 自身退化最强但跨退化最差 → **仅在盲识别高置信 + 退化匹配时用**
- DualBranch: 鲁棒但 exp37 盲识别场景 0 胜 → **仅在盲识别自信时考虑**

---

## 八、🔴 exp37 架构消融验证 (24盲识别挑战, R0参数, 1.5h Direct训练)

### 消融 A: 架构优化价值

| 模型 | 平均 PSNR | 胜率 |
|------|:--:|:--:|
| Spec+Arch (DualBranch/FiLM-GCM) | 21.05 | 0/24 |
| **Direct+Swin (纯 Swin)** | **22.86** | **23/24** |

> 在 Direct 训练 + 盲识别预测参数下，所有架构组件（DualBranch, FiLM-GCM, ColorPre）均未超越纯 Swin。exp10 中验证的组件收益（+0.8~+3.5 dB）仅在 GT 退化已知的对照实验中成立。

### 消融 B: True Ft (checkpoint微调) vs Direct

| 退化类型 | Ft (ckpt微调) | Direct (从头) | Δ |
|------|:--:|:--:|:--:|
| 纯局部 (14) | 24.55 | 24.60 | -0.05 |
| 混合 (10) | 20.39 | 20.42 | -0.03 |
| 全部 (24) | 22.82 | 22.86 | -0.04 |

> 1.5h 预算下 AR_LOAD_CKPT (盲预训练微调) 与 Direct 从零训练无统计差异。exp9/10 中 "Ft 优于 Direct" 的结论实际来自 Random Curriculum (AR_CURRICULUM_CONFIG=0:random)，不是 checkpoint 微调。

### 修正要点

1. exp9/10 "Ft" → 改称 **RandomCurric** (前50%步随机单退化, 后50%目标退化)。所有变体均从零训练，无 AR_LOAD_CKPT
2. **RandomCurric ≠ True Ft**: 前者减少梯度干扰，后者迁移预训练特征。机制完全不同，不应混为一谈
3. True Ft (AR_LOAD_CKPT) 在 1.5h 预算下 ≈ Direct。**不排除更长预算下 Ft 可能超越 Direct**，exp37 仅验证了 1.5h 约束
4. 架构优化 (DualBranch/FiLM-GCM) 在盲识别场景下未复现 exp10 收益。**不排除在 GT 退化已知或更长预算下有效**
5. RandomCurric + True Ft 的**组合效果尚未测试** (exp9 无 ckpt, exp37 无 RandomCurric)

### 待验证假设

- True Ft 在更长预算 (≥3h) 下是否超越 Direct?
- RandomCurric + True Ft 组合是否优于单独使用?
- 架构优化在 True Ft (而非 Direct) 下是否有效? (exp10 组件收益均在 Direct 下测得)
- Curric 方向 (Fwd/Rev) 与 RandomCurric 的交互?
