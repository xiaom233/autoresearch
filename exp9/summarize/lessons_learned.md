# 实验教训总结

## 一、有效的发现

### 1. Ft 是最安全的默认策略（30+组验证）

盲预训练→专攻永远不会显著输给任何策略。sev=1 时优势最大(+2.55 dB)，sev=5 时 Direct 微弱反超。

### 2. Standalone Curric 在特定退化上显著优于 Direct

D3 (comp+blur): Curric +4.09 dB。机制是梯度干扰减少——comp 和 blur 产生"第三种伪影"，Direct 的梯度互相冲突。

### 3. blur 子类型决定 Curric 方向（motion→Fwd, lens→Rev）

交叉验证通过（换 noise 类型偏好不变）。机制是 Phase1 迁移价值：motion 的 1D 反卷积是可迁移的基础技能。

### 4. FtCurr（盲预训练+课程）几乎总是最差

12 组中 11 组差于 Ft。原因：盲预训练已给通用特征，再插单退化阶段破坏通用性。

## 二、被证伪的假设

### 1. "三退化策略差异 < 0.5 dB" → 证伪

17 组三退化中位 Δ = 1.22 dB，71% ≥ 0.5 dB。策略在三退化上同样重要。

### 2. "PR 峰值比能零成本预测最优分配" → 证伪

PR 全部差于 Fixed split。峰值比过度偏好结构化退化（motion peak=92），饿死其他 phase。

### 3. "TA 动态 split 能系统提升 Curric" → 证伪

Phase 13 系统验证：TA 在 B1-B9 上全部 ≤ Fixed。Phase 11 的 T2/N12 收益（+2.91/+3.97）是特例，无法复现。

### 4. "动态 split 是普适改进" → 证伪

Fixed 5000/5000/5094 在 13/13 组上都是最稳健的。动态 split 有时更好（T2,N12），但无法提前预测。

### 5. "多专家级联能超越单模型 Ft" → 证伪

15 专家 + 6 联合微调，从未超越 Ft。误差级联放大。

### 6. "Loss 函数选择对结果有显著影响" → 证伪

l1/mse/huber/l1+edge/l1+fft 之间 Δ < 0.25 dB。l1 是最安全默认。

### 7. "Mixed Warmup 是 Curric 的升级版" → 部分证伪

三退化上 MW 优于 Curric(fwd)，但双退化上 Curric 更好。MW 不是普适改进。

### 8. "Ft 优势来自盲预训练见过类似退化" → 证伪

Ft 优势在 sev=1 时最大(+2.55)，在 sev=5 时消失。如果"见过类似"是对的，应该 sev=5 时优势最大。真正机制是盲预训练→目标的难度差作 curriculum 效应。

## 三、仍需探索的未知

### 1. 边际难度分配（Phase 13b）

当前 TA 按累计严重度分配——Phase3 总是最多步数。但 Phase1 才是打地基的阶段。改为按新增退化的边际难度分配。待验证。

### 2. 严重度×预算的交互

EPOCH=0.5/1/2 的 budget-aware 实验（Phase 12）尚未完成。低预算下策略差异可能更大。

### 3. 为什么 T2/N12 的 TA 收益不能复现

这两个是唯一 TA > Fixed 的案例。需要理解是 noise_corr + blur_motion + jpeg2000 这个特殊组合的什么属性导致了 TA 有效。

## 四、实验方法教训

### 1. 每 GPU 最多 1 个实验

违反导致 OOM。已写入 CLAUDE.md。

### 2. 需要 Direct 和 Ft 基线

只比较 Curric 内部变体不够——需要知道 Curric 是否优于 Direct/Ft。

### 3. gaussian blur 不能代表所有 blur

Phase 13 用 gaussian 作为中性基线，但 TA 在 gaussian 上无效不等于在其他 blur 上无效。motion/lens 的行为不同。

### 4. 严重度非线性

sev=5 的 noise_impulse (PSNR 43) 比 sev=3 的 blur_lens (PSNR 24) 容易得多。严重度标签本身不反映修复难度。

### 5. 固定 split 是强基线

5000/5000/5094 经过大量实验检验。MD/SL/TA/PR 四种动态方案均未系统超越 Fixed。

### 6. motion P1 + sev≥3 是唯一 Curric > Ft 的可靠方向

4/4 满足条件：S5(+1.29), M1(+0.22), X2(+0.09), M13(+0.08)。其他条件均 Ft 胜。

### 7. DFPIR 大模型不如专攻小模型

15 组退化对比，小模型(0.45M) 12/15 超越 DFPIR(31M)。G1 oversharpen 三退化 -6.45 dB。

### 8. 梯度干扰存在但 Curric 无法利用

O1 oversharpen+blur Ft > Direct +3.55，但 Curric 未超越 Ft。梯度干扰确实存在但 Ft 已经解决了。

### 9. split 微调不应作为主要方向

TA/SL/PR/MD 四种动态 split 方案，无一种系统优于 Fixed split。效应量级 < 0.5 dB，不值得投入。
