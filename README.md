# autoresearch — Image Restoration via Degradation Simulation

给定一张退化图像，自动识别其退化类型与严重程度，在干净数据集上在线模拟相同退化，训练专用复原模型。

## Pipeline 概览

```
┌─────────────────┐      ┌──────────────────────────┐      ┌─────────────────────┐      ┌──────────────────┐
│  1. 目标退化图像  │ ───▶ │ 2. degradation-simulator │ ───▶ │ 3. prepare.py        │ ───▶ │ 4. train.py       │
│  (target.png)   │      │   识别退化 → params.json   │      │   WebDataset + 退化模拟 │      │   复原模型训练      │
└─────────────────┘      └──────────────────────────┘      └─────────────────────┘      └──────────────────┘
```

- **Step 1-2**: 使用 `image-degradation-simulator` skill 分析目标退化图像，输出 `params.json`（退化流程参数）
- **Step 3**: `prepare.py` 将 DIV2K 等数据集裁切打包为 WebDataset，根据 `params.json` 在线模拟退化，提供 `(inputs, targets, epoch)` 格式的 DataLoader
- **Step 4**: `train.py` 训练图像复原模型（inputs=退化图, targets=clean 原图）

## 快速开始

**Requirements:** NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. 安装依赖
uv sync

# 2. 准备数据集（裁切 + WebDataset 打包，首次运行约 5 分钟）
uv run prepare.py

# 3. 测试 DataLoader（用已有的 params.json 验证流水线）
uv run prepare.py --demo
```

## 项目结构

```
prepare.py                              — 数据准备（裁切/WDS打包）+ DataLoader（退化模拟）+ 评估
train.py                                — 复原模型定义 + 训练循环（agent 可修改）
program.md                              — agent 指令
pyproject.toml                          — 依赖
x_distortion/                           — 退化算法库（35 种退化 × 5 级严重度）
.claude/skills/image-degradation-simulator/  — 退化识别 skill
    ├── SKILL.md                        — skill 指令
    ├── scripts/
    │   ├── analyze_degradation.py      — 定量分析退化图像
    │   ├── apply_multi.py             — 应用多步退化
    │   └── compare_degradation.py      — 对比退化纹理
    └── references/
        └── severity_mappings.md        — severity→物理参数映射表
```

## x_distortion

位于 `x_distortion/`，包含 35 种图像退化函数，每种支持 5 级严重度 (1–5)。输入输出均为 `np.ndarray, uint8, H×W×3, RGB, [0,255]`。

### 快速使用

```python
import numpy as np
from PIL import Image
from x_distortion import add_distortion

# 读取干净图像
img = np.array(Image.open('clean.png').convert('RGB'), dtype=np.uint8)

# 单步退化
degraded = add_distortion(img, severity=3, distortion_name='blur_gaussian')

# 多步退化（顺序很重要！先模糊再加噪声 ≠ 先噪声再模糊）
for func, sev in [('blur_gaussian', 3), ('noise_gaussian_RGB', 2), ('compression_jpeg', 3)]:
    img = add_distortion(img, severity=sev, distortion_name=func)

Image.fromarray(img).save('degraded.png')
```

### 退化类型一览

| 类别 | 函数 | 说明 |
|------|------|------|
| blur | `blur_gaussian` | 高斯模糊 |
| blur | `blur_motion` | 运动模糊（带随机角度） |
| blur | `blur_glass` | 玻璃模糊 |
| blur | `blur_lens` | 镜头模糊（径向） |
| blur | `blur_zoom` | 缩放模糊 |
| blur | `blur_jitter` | 抖动模糊 |
| noise | `noise_gaussian_RGB` | RGB 通道独立高斯噪声 |
| noise | `noise_gaussian_YCrCb` | YCrCb 空间高斯噪声（亮度权重不同） |
| noise | `noise_speckle` | 散斑噪声 |
| noise | `noise_spatially_correlated` | 空间相关噪声 |
| noise | `noise_poisson` | 泊松噪声 |
| noise | `noise_impulse` | 椒盐/脉冲噪声 |
| compression | `compression_jpeg` | JPEG 压缩伪影 |
| compression | `compression_jpeg_2000` | JPEG 2000 压缩伪影 |
| brighten | `brightness_brighten_shift_HSV` | HSV 空间亮度提升 |
| brighten | `brightness_brighten_shift_RGB` | RGB 空间亮度提升 |
| brighten | `brightness_brighten_gamma_HSV` | HSV 空间 Gamma 提亮 |
| brighten | `brightness_brighten_gamma_RGB` | RGB 空间 Gamma 提亮 |
| darken | `brightness_darken_shift_HSV` | HSV 空间亮度降低 |
| darken | `brightness_darken_shift_RGB` | RGB 空间亮度降低 |
| darken | `brightness_darken_gamma_HSV` | HSV 空间 Gamma 压暗 |
| darken | `brightness_darken_gamma_RGB` | RGB 空间 Gamma 压暗 |
| contrast | `contrast_strengthen_scale` | 对比度增强（缩放） |
| contrast | `contrast_strengthen_stretch` | 对比度增强（拉伸） |
| contrast | `contrast_weaken_scale` | 对比度减弱（缩放） |
| contrast | `contrast_weaken_stretch` | 对比度减弱（拉伸） |
| saturation | `saturate_strengthen_HSV` | HSV 空间饱和度增强 |
| saturation | `saturate_strengthen_YCrCb` | YCrCb 空间饱和度增强 |
| saturation | `saturate_weaken_HSV` | HSV 空间饱和度减弱 |
| saturation | `saturate_weaken_YCrCb` | YCrCb 空间饱和度减弱 |
| other | `oversharpen` | 过度锐化 |
| other | `pixelate` | 像素化 |
| other | `quantization_otsu` | 量化（Otsu 阈值） |
| other | `quantization_median` | 量化（中值分割） |
| other | `quantization_hist` | 量化（直方图分割） |

### Severity → 物理参数映射

#### Blur

| 函数 | 参数 | sev=1 | sev=2 | sev=3 | sev=4 | sev=5 |
|------|------|-------|-------|-------|-------|-------|
| `blur_gaussian` | sigma | 1 | 2 | 3 | 4 | 5 |
| `blur_motion` | (radius, sigma) | (5,3) | (10,5) | (15,7) | (15,9) | (20,12) |
| `blur_glass` | (sigma, shift, iter) | (0.7,1,1) | (0.9,2,1) | (1.2,2,2) | (1.4,3,2) | (1.6,4,2) |
| `blur_lens` | radius | 2 | 3 | 4 | 6 | 8 |
| `blur_zoom` | zoom_range | [1,1.03] | [1,1.06] | [1,1.10] | [1,1.15] | [1,1.21] |
| `blur_jitter` | shift | 1 | 2 | 3 | 4 | 5 |

#### Noise

| 函数 | 参数 | sev=1 | sev=2 | sev=3 | sev=4 | sev=5 |
|------|------|-------|-------|-------|-------|-------|
| `noise_gaussian_RGB` | sigma | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 |
| `noise_gaussian_YCrCb` | sigma_l / sigma_r / sigma_b | 0.05/0.05/0.05 | 0.06/0.087/0.087 | 0.07/0.133/0.133 | 0.08/0.188/0.188 | 0.09/0.252/0.252 |
| `noise_speckle` | sigma | 0.14 | 0.21 | 0.28 | 0.35 | 0.42 |
| `noise_spatially_correlated` | sigma | 0.08 | 0.11 | 0.14 | 0.18 | 0.22 |
| `noise_poisson` | c | 80 | 60 | 40 | 25 | 15 |
| `noise_impulse` | amount | 0.01 | 0.03 | 0.05 | 0.07 | 0.10 |

#### Compression

| 函数 | 参数 | sev=1 | sev=2 | sev=3 | sev=4 | sev=5 |
|------|------|-------|-------|-------|-------|-------|
| `compression_jpeg` | quality | 25 | 18 | 12 | 8 | 5 |
| `compression_jpeg_2000` | quality (dB) | 29 | 27.5 | 26 | 24.5 | 23 |

#### Brightness / Contrast / Saturation / Other

| 函数 | 参数 | sev=1 | sev=2 | sev=3 | sev=4 | sev=5 |
|------|------|-------|-------|-------|-------|-------|
| `brightness_brighten_gamma_RGB` | gamma | 0.8 | 0.7 | 0.6 | 0.45 | 0.3 |
| `brightness_darken_gamma_RGB` | gamma | 1.4 | 1.7 | 2.1 | 2.6 | 3.2 |
| `contrast_strengthen_scale` | scale | 1.4 | 1.7 | 2.1 | 2.6 | 4.0 |
| `contrast_weaken_scale` | scale | 0.75 | 0.6 | 0.45 | 0.3 | 0.2 |
| `saturate_strengthen_HSV` | scale | 3.0 | 6.0 | 12.0 | 20.0 | 64.0 |
| `saturate_weaken_HSV` | scale | 0.7 | 0.55 | 0.4 | 0.2 | 0.0 |
| `oversharpen` | amount | 2 | 2.8 | 4 | 6 | 8 |
| `pixelate` | scale_factor | 0.5 | 0.4 | 0.3 | 0.25 | 0.2 |
| `quantization_otsu` | num_classes | 15 | 11 | 8 | 5 | 3 |
| `quantization_median` | num_classes | 20 | 15 | 10 | 6 | 3 |
| `quantization_hist` | num_classes | 24 | 16 | 8 | 6 | 4 |

完整参数参见 `.claude/skills/image-degradation-simulator/references/severity_mappings.md`。

## DataLoader 使用

```python
from prepare import make_dataloader_restoration

loader = make_dataloader_restoration(
    params_path='path/to/params.json',           # degradation-simulator 输出
    shards_url='datasets/DIV2K/DIV2K_train_HR_wds/train-*.tar',
    batch_size=16,
)

for inputs, targets, epoch in loader:
    # inputs: [B, 3, H, W] 退化图像 (float32, [0,1])
    # targets: [B, 3, H, W] 干净原图 (float32, [0,1])
    # epoch: 当前 epoch 编号（无穷循环，递增）
    ...
```

## License

MIT
