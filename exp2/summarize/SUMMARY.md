# exp2 — 双 JPEG 2000 压缩盲识别

## 目标

盲识别并恢复 `noise_gaussian_RGB(2) → compression_jpeg_2000(3) → compression_jpeg_2000(2)`。测试对同类别重复退化（双 JPEG 2000）的识别能力。

## 退化管线

真实: `noise_gaussian_RGB(2) → compression_jpeg_2000(3) → compression_jpeg_2000(2)`

## 盲识别结果

**完全失败**。预测: `blur_gaussian(1) → compression_jpeg(4) → noise_impulse(2)`——类型、严重度全部错误。

## 结果

| 模型 | PSNR |
|------|:--:|
| M_blind | 19.93 |
| M_specialist (错误退化) | 23.04 |
| M_truth (GT) | **31.01** |

## 结论

- 盲识别完全失败，但 M_specialist (23.04) 仍优于 M_blind (19.93)，说明即使在错误退化上训练也能提供有限迁移
- 与 M_truth 的 8 dB 差距说明了盲识别准确性的关键重要性
- 双 JPEG 2000 同类别重复退化是极难识别的场景
