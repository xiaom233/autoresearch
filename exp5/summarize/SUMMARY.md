# exp5 — 强弱退化掩盖导致盲识别失败

## 目标

盲识别并恢复 `compression_jpeg(1) → blur_jitter(5)`。测试弱退化被强退化掩盖的场景。

## 退化管线

真实: `compression_jpeg(1) → blur_jitter(5)`

## 盲识别结果

**失败**——JPEG severity 1 被完全漏检。Jitter severity 5 将 JPEG 8×8 块边界打散，content-independent 指标无法检测 JPEG。

## 结果

| 模型 | PSNR |
|------|:--:|
| M_blind | 20.89 |
| M_specialist (仅 jitter) | **20.30** |
| M_truth | 22.27 |

## 结论

- **负迁移**：基于错误退化训练的 M_specialist 比盲模型更差 (-0.59 dB)
- 盲识别漏检是根本原因——弱退化被强退化完全掩盖
- 启示：盲识别协议需要退化掩蔽感知，当退化强度差异大时弱退化变得不可检测
- 后续实验（exp11）专门量化了此类漏检的代价
