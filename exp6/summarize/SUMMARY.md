# exp6 — 近乎完美盲识别 + 架构搜索成功

## 目标

盲识别并恢复 `blur_gaussian(5) → compression_jpeg(3) → noise_gaussian_YCrCb(4)`。三步中高严重度退化。

## 退化管线

真实: `blur_gaussian(5) → compression_jpeg(3) → noise_gaussian_YCrCb(4)`
预测: `blur_gaussian(5) → compression_jpeg(2) → noise_gaussian_YCrCb(4)`（仅 JPEG severity 差 1 级）

## 盲识别结果

**近乎完美**。三种退化类型和顺序完全正确，仅 JPEG severity 偏差 1 级。

## 结果

| 模型 | PSNR |
|------|:--:|
| M_blind | 19.53 |
| M_specialist | **20.65** |
| M_truth | 20.64 |

## 结论

- **首次 Specialist 追平 M_truth 上限**（+0.01 dB）
- 识别中 severity 1 级的误差可通过架构优化完全弥补
- Cr/Cb 通道分析是区分 YCrCb 噪声与 RGB 噪声的关键
- 最佳配置：constLR + embed_dim=96 + gradient clipping
- 证明准确盲识别 + 架构搜索 = 可超越盲模型 +1.12 dB
