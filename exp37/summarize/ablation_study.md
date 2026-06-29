# exp37 消融实验报告

## 一、实验目的

全面对比三个因素在 R0/R1 参数下的表现，形成 3×2 矩阵：

| | R0 参数 (24挑战) | R1 参数 (8挑战) |
|------|:--:|:--:|
| **Spec** (架构优化) | ✅ | ✅ |
| **Ft+Swin** (预训练微调) | ✅ | 🔄 |
| **Direct+Swin** (从头训练) | ✅ | 🔄 |

从中提取三个消融：

| 消融 | 变量 | 对比 |
|------|------|------|
| **A: 架构优化** | DualBranch/FiLM-GCM | Spec vs Direct+Swin (同参数) |
| **B: 反思价值** | R0→R1 参数改善 | R1 vs R0 (同模型) |
| **C: 预训练迁移** | 盲预训练 ckpt | Ft+Swin vs Direct+Swin (同参数) |

## 二、实验设计

### 🔴 公平性保证

**所有模型使用完全相同的 R0 预测参数训练。** R1 对比仅在消融 B 中作为变量。

| 参数 | 值 |
|------|------|
| 训练退化 | **R0 预测参数** (`degradation/blind_XXXX_params.json`) |
| EPOCH_BUDGET | 2 |
| LR | 5e-4 |
| Loss | L1 |
| Batch size | 16 |
| AMP | bfloat16 |
| GPU 时间/挑战 | ~1.5h |
| GT 评估退化 | GT 参数 (`degradation_gt/`) |
| GT 评估数据 | DIV2K (100张) + LSDIR (250张) |

### 执行计划

| # | 实验 | 挑战数 | 状态 | 预计 GPU-h |
|:--:|------|:--:|:--:|:--:|
| 1 | Spec R0 | 24 | ✅ | 36 |
| 2 | Spec R1 | 8 | ✅ | 12 |
| 3 | Direct+Swin R0 | 24 | 🔄 训练中 | 36 |
| 4 | Direct+Swin R1 | 8 | ⏳ | 12 |
| 5 | Ft+Swin R0 | 24 | ⏳ | 36 |
| 6 | Ft+Swin R1 | 8 | ⏳ | 12 |
| **合计** | | | | **144 GPU-h** |

> R1 仅 8 挑战有 R1 改善参数，其余 16 挑战 R1=R0。

### 模型组

| 组 | 策略 | 架构 | 预训练 | 参数量 | 状态 |
|------|------|------|:--:|:--:|:--:|
| **Spec R0** | Direct(18)/Ft(6) | DualBranch(14)/FiLM-GCM(3)/Swin(7) | 部分 | 0.45-0.46M | ✅ |
| **Spec R1** | 同架构 | 同架构 | 部分 | 0.45-0.46M | ✅ |
| **Direct+Swin R0** | 统一 Direct | 纯 Swin | ❌ | 0.45M | 🔄 |
| **Direct+Swin R1** | 统一 Direct | 纯 Swin | ❌ | 0.45M | ⏳ |
| **Ft+Swin R0** | 统一 Ft | 纯 Swin | ✅ | 0.45M | ⏳ |
| **Ft+Swin R1** | 统一 Ft | 纯 Swin | ✅ | 0.45M | ⏳ |

## 三、消融 A: 架构优化价值

### 对比: Spec R0 vs Direct+Swin R0

| 维度 | Spec R0 | Direct+Swin R0 |
|------|:--:|:--:|
| 训练参数 | R0 | R0 |
| 训练策略 | Direct/Ft | Direct |
| 架构 | **DualBranch/FiLM-GCM/Swin** | **纯 Swin** |
| 预训练 | 部分 | 无 |

> 公平对比：同 R0 参数，只差架构。

### 结果

**Spec+Arch (21.05 dB) vs Direct+Swin (22.86 dB) — 架构优化 0:23 惨败。**

| Challenge | Spec+Arch | Direct+Swin | Δ | Winner |
|------|:--:|:--:|:--:|:--:|
| blind_0001 | 17.72 | 21.67 | -3.95 | Direct+Swin |
| blind_0002 | 30.96 | 31.88 | -0.92 | Direct+Swin |
| blind_0003 | 21.84 | 25.65 | -3.81 | Direct+Swin |
| blind_0004 | 21.48 | 24.63 | -3.15 | Direct+Swin |
| blind_0005 | 14.04 | 14.61 | -0.57 | Direct+Swin |
| blind_0006 | 28.72 | 30.54 | -1.82 | Direct+Swin |
| blind_0007 | 32.17 | 34.31 | -2.14 | Direct+Swin |
| blind_0008 | 38.71 | 40.81 | -2.10 | Direct+Swin |
| blind_0009 | 6.91 | 6.97 | -0.06 | Tie |
| blind_0010 | 30.14 | 30.46 | -0.32 | Direct+Swin |
| blind_0011 | 20.84 | 23.46 | -2.62 | Direct+Swin |
| blind_0012 | 20.51 | 21.85 | -1.34 | Direct+Swin |
| blind_0013 | 25.21 | 26.61 | -1.40 | Direct+Swin |
| blind_0014 | 21.03 | 23.35 | -2.32 | Direct+Swin |
| blind_0015 | 18.89 | 19.87 | -0.98 | Direct+Swin |
| blind_0016 | 22.37 | 23.91 | -1.54 | Direct+Swin |
| blind_0017 | 20.46 | 22.15 | -1.69 | Direct+Swin |
| blind_0018 | 16.04 | 17.55 | -1.51 | Direct+Swin |
| blind_0019 | 14.93 | 17.27 | -2.34 | Direct+Swin |
| blind_0020 | 13.35 | 15.88 | -2.53 | Direct+Swin |
| blind_0021 | 14.52 | 16.01 | -1.49 | Direct+Swin |
| blind_0022 | 20.72 | 22.78 | -2.06 | Direct+Swin |
| blind_0023 | 18.31 | 20.32 | -2.01 | Direct+Swin |
| blind_0024 | 15.35 | 16.03 | -0.68 | Direct+Swin |

**胜率: Spec+Arch 0 : 23 Direct+Swin : 1 Tie**

**结论: 在 1.5h Direct 训练预算下，finetune_strategy.md 的架构选择（DualBranch/FiLM-GCM）不仅无效，反而有害。纯 Swin + Direct 是更优策略。**

## 四、消融 B: 反思价值

### 对比: Spec R1 vs Spec R0

| 维度 | Spec R0 | Spec R1 |
|------|:--:|:--:|
| 训练参数 | R0 | **R1 (反思改善)** |
| 架构 | 相同 | 相同 |
| 策略 | 相同 | 相同 |

> 公平对比：同架构同策略，只差参数 (R0→R1)。

### 结果

| Challenge | R0 Spec | R1 Spec | Δ R1-R0 | R1 参数变化 |
|------|:--:|:--:|:--:|------|
| blind_0001 | 17.72 | 27.31 | +9.59 | 去掉 quantization_hist |
| blind_0005 | 14.04 | 13.87 | -0.17 | 维持 R0 (BEYOND_CAPABILITY) |
| blind_0009 | 6.91 | 11.55 | +4.64 | 3步→2步, shift→gamma |
| blind_0012 | 20.51 | 20.70 | +0.19 | noise sev 2→1 |
| blind_0014 | 21.03 | 21.53 | +0.50 | glass→gaussian, JPEG sev 调整 |
| blind_0018 | 16.04 | 16.77 | +0.73 | 添加 YCrCb noise |
| blind_0019 | 14.93 | 14.62 | -0.31 | gaussian→lens (误判) |
| blind_0021 | 14.52 | 17.55 | +3.03 | motion→lens, gamma→shift |

**R1 平均 Δ: +2.27 dB (8 挑战)**，有效 5/8，有害 2/8，持平 1/8。

## 五、消融 C: 预训练迁移价值

### 对比: Ft+Swin R0 vs Direct+Swin R0

| 维度 | Ft+Swin R0 | Direct+Swin R0 |
|------|:--:|:--:|
| 训练参数 | R0 | R0 |
| 架构 | 纯 Swin | 纯 Swin |
| 策略 | Ft (微调) | Direct (从头) |
| 预训练 | ✅ blind ckpt | ❌ |

> 公平对比：同 R0 参数，同 Swin 架构，只差预训练。

### 结果

**Ft+Swin (22.82 dB) vs Direct+Swin (22.86 dB) — 基本持平，差距仅 0.04 dB。**

胜率: Ft+Swin 8 : Direct+Swin 5 : Tie 11。预训练在 1.5h 预算下无明显优势，统计不显著。

## 六、Qwen 盲识别消融

### 对比: Qwen vs Skill R0

| 指标 | Qwen 3.7-Plus | Skill R0 |
|------|:--:|:--:|
| 函数匹配率 | 10.4% (5/48) | **58.3%** (28/48) |
| 空管线 | 3/24 (12.5%) | 0/24 |
| 完全正确 | 1/24 (4.2%) | — |

Qwen 主要失败模式：噪声类型全部判为 gaussian_RGB，频繁幻觉 contrast，多步退化无法识别。

> ⏳ Qwen 预测参数训练待启动。

## 七、GPU 资源

| 实验 | GPU-h | 墙钟 |
|------|:--:|:--:|
| Spec R0 | 36 | ~4.5h |
| Spec R1 | 12 | ~1.5h |
| Direct+Swin R0 | 36 | ~4.5h |
| Ft+Swin R0 | 36 | ~4.5h |
| Qwen 盲识别 | 0 (API) | ~1h |
| **合计** | **120** | — |
