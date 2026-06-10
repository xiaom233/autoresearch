# exp7 — 跨图盲识别完全失败

## 目标

盲识别并恢复 `noise_spatially_correlated(4) → blur_lens(4)`。高严重度双退化。

## 退化管线

真实: `noise_spatially_correlated(4) → blur_lens(4)`
预测: `blur_gaussian(4) → compression_jpeg_2000(2)`（全部错误）

## 盲识别结果

**完全失败（等级 F，17%）**：
- 函数匹配 0/2
- 类别匹配 0/2
- 将 lens blur 误判为 gaussian blur
- 将 spatially correlated noise 误判为 JPEG 2000

## 结果

实验在盲识别失败后终止，未进行训练。

## 结论

- 跨图模式无法区分 lens blur vs gaussian blur 和 spatially correlated noise vs JPEG 压缩
- 空间相关噪声在跨图模式下表现为类似压缩的伪影
- 这是跨图模式固有缺陷的典型案例——不同图片的内容差异干扰退化判断
- 直接推动了 exp13 同图模式的采用
