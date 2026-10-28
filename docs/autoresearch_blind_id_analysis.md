# 三组实验盲识别分析报告

> 2026-06-20 | exp28/exp29/exp30 全流程完成

## 一、实验设计

| 实验 | 模式 | 挑战数 | 退化步数 | Seed 范围 |
|---|---:|---:|:--:|
| exp28 | cross-image (异图) | 16 | 双退化 (num-degs=2) | — |
| exp29 | same-image (同图) | 16 | 双退化 (num-degs=2) | 290001-290016 |
| exp30 | same-image (同图) | 16 | 双退化 (num-degs=2) | 300001-300016 |

## 二、盲识别质量对比

### 2.1 函数匹配统计

| 实验 | 完全匹配 | 部分匹配 | 不匹配 | 平均 R0-GT gap | Specialist>DFPIR |
|---|:---:|:---:|:---:|:---:|:---:|
| exp28 (cross) | 3/16 (19%) | 9/16 | 4/16 | +8.8 dB | 13/16 |
| exp29 (same) | **7/16 (44%)** | 9/16 | **0/16** | **+3.5 dB** | 12/16 |
| exp30 (same) | 2/16 (13%) | 12/16 | 2/16 | +6.1 dB | **15/16** |

### 2.二 exp28 (cross-image) 详细分析

**漏检分布**: noise=9, compression=0, blur=3, global=5

**R1 反思**: 0/16 保存了 R1 修正参数

### 2.三 exp29 (same-image) 详细分析

**漏检分布**: noise=8, compression=0, blur=1, global=0

**R1 反思**: 4/16 保存了 R1 修正参数

### 2.四 exp30 (same-image) 详细分析

**漏检分布**: noise=4, compression=3, blur=2, global=6

**R1 反思**: 3/16 保存了 R1 修正参数

## 三、核心发现

### 3.1 同图 vs 跨图模式

- **同图模式 (exp29/exp30) 平均 gap 约 5 dB，跨图模式 (exp28) 约 9 dB**
- 同图模式像素级校准使 content-dependent 指标可用，噪声 σ 估计更精确
- 但同图模式仍然存在 ~50% 的不完全匹配率，说明双退化识别本身极具挑战

### 3.2 退化掩盖效应 (Masking)

- **噪声掩盖全局退化**：强噪声 (σ>10) 破坏 variance_ratio、mean_shift 等全局信号
- **blur 掩盖 compression**：heavy blur (gm_ratio<0.5) 抹平 block_boundary
- **compression 掩盖 noise**：JPEG DCT 量化产生类似 noise 的高频伪影
- 这是双退化盲识别的根本瓶颈：SKILL.md 的逐层剥离依赖确定性退化先被识别

### 3.3 R1 反思机制有效性

- exp30 反思发现 3 个漏检退化 (0002 漏 blur +4.46dB, 0005 漏 saturation+noise +7.41dB, 0013 漏 JPEG +1.71dB)
- exp29 反思未发现改善：R0 质量更高 (平均 gap 仅 3.5dB)，剩余 gap 主要来自 severity 偏差而非函数错误
- **R1 反思的收益取决于 R0 盲识别质量**：R0 差 → R1 空间大；R0 好 → R1 边际收益小

### 3.4 Specialist vs DFPIR

- **三组实验 Specialist 一致且显著优于 DFPIR (31M)**：40/48 (83%) 的挑战 R0>DFPIR
- 同图模式下 Specialist 优势更大 (exp30: 15/16)，因为盲识别更准确 → 训练数据更匹配
- DFPIR 均值约 19 dB，受限于 all-in-one 架构的任务冲突

## 四、改进方向

1. **掩盖推理**：SKILL.md 的掩盖推理触发条件应放宽（当前要求 PSNR<35dB AND 已知退化数不足 AND 不含 compression）
2. **噪声-全局解耦**：强噪声场景需要先做去噪预处理再检测全局退化信号
3. **多假设训练**：双退化 UNCERTAIN 时，对 top-2 候选分别训练，训练后 PSNR 仲裁
4. **blind_0008 类型问题**：severe gamma/brightness 导致 unique_G 降低，触发 compression FP，需要新交叉验证规则
5. **blind_0013 类型问题**：quantization + 未知预处理 → 超出 x_distortion 覆盖范围 → 需要 OOD 检测机制