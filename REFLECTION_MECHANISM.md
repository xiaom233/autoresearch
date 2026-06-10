# Reflection Mechanism — 训练策略反思修正

## 核心原理

反思的目标不是修正退化参数（exp13 场景），而是**调整训练策略**使 Spec 模型达到或超过 Ft 水平。

## 诊断框架

### 1. 分析 Spec 日志的诊断特征

从训练日志中提取：
- **最终 PSNR**：Spec vs Ft vs Blind baseline
- **Loss 收敛曲线**：是否提前饱和？是否需要更多步数？
- **退化类型**：blur/noise/compression/brightness/contrast/saturation/quantization 的组合

### 2. 识别 Ft >> Spec 的根本原因

| 原因 | 诊断信号 | 修正方向 |
|------|---------|---------|
| 盲预训练特征迁移 | Ft 大幅领先，退化复杂 | 使用 Ft (LOAD_CKPT) |
| 梯度冲突 | 退化方向相反（blur+sharpen） | 降低 LR 或 Curric |
| 训练步数不足 | loss 仍在下降 | 增加 EPOCH_BUDGET |
| LR 不当 | loss 震荡或下降太慢 | 调整 LR |
| 退化严重度过高 | PSNR 绝对值低 | 降低 sev 或分阶段 |

### 3. 基于 finetune_strategy.md 的原则

**原则 1（梯度干扰）**：blur + sharpen/contrast 组合 → 梯度冲突 → Ft 或 Curric
**原则 2（迁移价值）**：去噪/去压缩 → 高迁移价值 → Ft 有效
**原则 4（预训练通用性）**：盲预训练 → 通用特征 → Ft 利用已有知识
**原则 7（策略效应量级）**：梯度干扰 >> 迁移价值 > 修复难度 > LR/loss

## 修正决策树

```
Spec < Ft?
├── Δ > 3 dB → 严重问题
│   ├── 退化复杂（≥2种）→ Ft + 降低 LR (1e-4)
│   └── 退化简单（1种）→ 检查训练是否崩溃，重新训练
├── 1 dB < Δ ≤ 3 dB → 中度问题
│   ├── 盲预训练特征可迁移 → Ft + LR=5e-4
│   └── 梯度冲突 → Ft + LR=1e-4
├── 0.3 dB < Δ ≤ 1 dB → 轻微问题
│   └── Ft + 调整 LR 或 EPOCH_BUDGET
└── Δ ≤ 0.3 dB → 基本持平，可选 Ft 或保持 Spec
```

## R1 修正格式

```json
{
  "degradation": "D01_dual",
  "round": 1,
  "spec_psnr_before": 28.06,
  "ft_psnr": 28.17,
  "diagnosis": ["Ft ≈ Spec (Δ=0.11 dB)，盲预训练对此退化帮助有限"],
  "suggested_action": "use_ft",
  "ar_flags": "AR_LOAD_CKPT=exp12/experiments/M_blind/checkpoints/M_blind_step15092.pt AR_LEARNING_RATE=5e-4"
}
```

## 训练预算公平性 ⚠️

**所有修正训练 (R1/R2/R3) 必须使用与基线 (Spec/Ft) 相同的 EPOCH_BUDGET=2。**

- Spec 和 Ft 基线使用 EPOCH_BUDGET=2 (15094 steps)
- R1/R2/R3 修正训练也必须使用 EPOCH_BUDGET=2
- **禁止**通过增加训练步数 (EPOCH_BUDGET=3) 来获得 PSNR 提升
- PSNR 提升必须来自策略改进（LR、LOAD_CKPT、架构等），而非更多训练时间

**Why**: 若 R1 用 EPOCH=3，任何 PSNR 提升都可能只是因为多训练了 50% 步数，无法判断是策略改进还是训练时间的效果。这会导致错误的经验总结。

## 多轮修正机制

- **R1**：首次分析 → 提出修正 → 训练 (EPOCH=2) → 对比 PSNR
- **R2**：R1 后 Δ > 0.5 dB → 继续调整（如改 LR、加架构特性。**不能增加 EPOCH**）
- **R3**：R2 后仍有明显差距 → 最后尝试（如 Curric、loss function 变更。**不能增加 EPOCH**）
- 最多 3 轮，每轮 PSNR 提升需 > 0.5 dB 才值得继续

## 关键经验

1. **Ft 是所有退化的安全选择**——exp9 (~500 组) 证明 Ft 几乎不会显著差于 Spec
2. **盲识别质量决定一切**——exp11 证明类型错误代价 -17.5 dB，严重度错误 -14.5 dB，漏检 -12.9 dB。盲识别准确远比训练策略重要
3. **盲预训练的通用特征最有价值**——当退化涉及 blur + noise + compression 多类别时
4. **Spec 优势场景稀少**——仅在退化极简单（单一轻微退化）或盲识别预测参数接近 GT 时
5. **不要过度修正**——Δ < 0.3 dB 不值得额外训练
6. **EPOCH_BUDGET 必须统一**——所有修正训练必须用 EPOCH=2，与基线公平对比
7. **LR 微调难以超越 Ft**——exp12 的 R1/R2 实验证明，纯训练策略调整几乎无法超越原始 Ft (LR=5e-4)。改进应聚焦于修正退化参数或架构升级

## 误识别诊断阈值 (来源: exp11 ~1284 组交叉测试)

盲识别错误的最小化比训练策略调整更关键。以下硬规则可直接用于检测盲识别可能存在的错误：

| 指标组合 | 阈值 | 指示 |
|---------|------|------|
| `edge > 0.10` AND `chroma > 0.03` | → | 98% 精度误识别 |
| `entropy > 2.0` AND `color_asymmetry > 0.08` | → | 退化类型误判 |
| `entropy > 2.0` AND `saturation < 0.15` | → | gamma 误判 |
| `edge_frac < 0.13` AND `entropy > 2.0` | → | 严重度偏差 |

### 常见混淆退化对 (PSNR gap 14-21 dB)

| 退化对 | Gap | 注意 |
|--------|:--:|------|
| BS ↔ SF7 | 16-20 dB | brightness/saturation 混淆 |
| L4 ↔ SF7 | 16-20 dB | brightness 识别为 saturate |
| SF5 ↔ SF7 | 14-21 dB | 单退化内部的严重混淆 |
| L2 ↔ L6 | 15-16 dB | contrast/noise 组合混淆 |

### 架构鲁棒性排序 (误识别风险下)

在盲识别可能不准确的退化上，选择更鲁棒的架构可减少 PSNR 损失：
DualBranch (10.1 dB gap) > CSN (11.7) > Swin (13.2) > FiLM (13.6) > FreqMod (15.2) > ColorPre (16.0)

注意：ColorPre 在自身退化上最强 (+0.81~+18.91 dB)，但跨退化泛化最差——仅在盲识别置信度极高时使用。
