| # | 轮次 | 假设管线 | 仿真 PSNR | CI | 决策 | 归因类别 | 依据/备注 |
|---:|:--:|---|---:|---:|---|---|---|
| 1 | R0 | `blur_gaussian:2` | 34.83 | 8/10 | ADJUST_SEVERITY | R+L | gaussian 主假设；gm_ratio=0.532；PSNR 模式；30-40dB → UNCERTAIN |
| 2 | R0 | `blur_gaussian:3` | 32.89 | 7/10 | ADJUST_SEVERITY | R | severity 扫描（SKILL 强制） |
| 3 | R0 | `blur_gaussian:1` | 32.91 | 8/10 | ADJUST_SEVERITY | R | severity 扫描（SKILL 强制） |
| 4 | R0 | `blur_gaussian:2,compression_jpeg:3` | 29.76 | 4/10 | TRY_ALTERNATIVE | L | dct_zero_ratio=0.624 触发 JPEG 假设；PSNR 29.76 拒绝 |
| 5 | R0 | `blur_gaussian:4` | 30.55 | 5/10 | ADJUST_SEVERITY | R | severity 扫描；更重 blur 不优 |
| 6 | R0 | `blur_motion:2` | 25.76 | 8/10 | TRY_ALTERNATIVE | R | SKILL 要求至少测试 3 种 blur；motion 仅 25.76 |
| 7 | R0 | `blur_gaussian:2,noise_gaussian_YCrCb:1` | 21.4 | 3/10 | TRY_ALTERNATIVE | L | noise 假设（sigma/spatial_corr）；PSNR 21.40 拒绝 |
| 8 | R0 | `blur_lens:2` | 34.61 | 8/10 | ADJUST_SEVERITY | R | blur 跨族测试（lens）；34.61 接近 gaussian |
| 9 | R0 | `compression_jpeg:3,blur_gaussian:2` | 31.2 | 7/10 | ADJUST_SEVERITY | L | 违反物理顺序的对照（JPEG→blur）；31.20 拒绝 |
| 10 | R0 | `blur_gaussian:2,compression_jpeg:1` | 32.04 | 5/10 | ADJUST_SEVERITY | L | JPEG severity 细化；32.04 拒绝 |
| 11 | R1 | `blur_glass:2` | 36.65 | 10/10 | ADJUST_SEVERITY | L | R1 自主补测未测过的 glass：36.65/10/10 = GT 类型 |
| 12 | R1 | `blur_glass:1` | 35.8 | 8/10 | ADJUST_SEVERITY | R | glass severity 扫描 |
| 13 | R1 | `blur_glass:3` | 33.05 | 8/10 | ADJUST_SEVERITY | R | glass severity 扫描 |
| 14 | R1 | `blur_glass:2` |  | MATCH | ACCEPT | R | verify_signals 强制验证 → MATCH(2✓) |
| 15 | R2 | `brightness_brighten_gamma_RGB:1,blur_gaussian:2` | 22.79 | 7/10 | TRY_ALTERNATIVE | L | gamma 假设（cross_channel_corr=0.731）；22.79 拒绝 |
| 16 | R2 | `blur_gaussian:2,brightness_brighten_gamma_RGB:1` | 22.71 | 7/10 | TRY_ALTERNATIVE | L | gamma 在 blur 后；22.71 拒绝 |
| 17 | R2 | `brightness_brighten_gamma_RGB:1,blur_glass:2` | 22.88 | 9/10 | TRY_ALTERNATIVE | L | gamma+glass；22.88 拒绝 |
| 18 | R2 | `blur_gaussian:2,compression_jpeg:2` | 31.16 | 4/10 | ADJUST_SEVERITY | L | JPEG sev=2 补测；31.16 拒绝 |
| 19 | R2 | `blur_gaussian:2,compression_jpeg:4` | 28.09 | 4/10 | TRY_ALTERNATIVE | L | JPEG sev=4 补测；28.09 拒绝 |
| 20 | R2 | `contrast_weaken_stretch:1,blur_gaussian:2` | 16.84 | 3/10 | REJECT | L | contrast 假设（variance_ratio=0.954）；16.84 拒绝 |
| 21 | R4 | `blur_glass:2` | 36.58 | 10/10 | ADJUST_SEVERITY | L | R4 重新验证 glass 基线；36.58/10/10 |
| 22 | R4 | `blur_glass:2,compression_jpeg:1` | 33.1 | 5/10 | ADJUST_SEVERITY | L | glass+JPEG:1；33.10 拒绝 |
| 23 | R4 | `blur_glass:2,compression_jpeg:2` | 32.02 | 4/10 | ADJUST_SEVERITY | L | glass+JPEG:2；32.02 拒绝 |
| 24 | R4 | `blur_glass:2,compression_jpeg:3` | 30.38 | 4/10 | ADJUST_SEVERITY | L | glass+JPEG:3；30.38 拒绝 |
| 25 | R4 | `blur_glass:2,compression_jpeg:4` | 28.52 | 4/10 | TRY_ALTERNATIVE | L | glass+JPEG:4；28.52 拒绝 |
| 26 | R4 | `blur_glass:2,compression_jpeg:5` | 26.04 | 4/10 | TRY_ALTERNATIVE | L | glass+JPEG:5；26.04 拒绝，JPEG 假阳性封闭 |
| 27 | R4 | `blur_glass:4` | 30.45 | 6/10 | ADJUST_SEVERITY | L | glass severity 4；30.45 拒绝 |
| 28 | R4 | `blur_glass:5` | 28.6 | 6/10 | TRY_ALTERNATIVE | L | glass severity 5；28.60 拒绝 |
| 29 | R4 | `blur_glass:2,noise_gaussian_YCrCb:1` | 21.47 | 3/10 | TRY_ALTERNATIVE | L | glass+YCrCb 噪声；21.47 拒绝 |
| 30 | R4 | `blur_glass:2,noise_gaussian_RGB:1` | 25.72 | 3/10 | TRY_ALTERNATIVE | L | glass+RGB 噪声；25.72 拒绝 |
| 31 | R4 | `blur_lens:3` | 34.6 | 9/10 | ADJUST_SEVERITY | L | lens:3；34.60 低于 glass |
| 32 | R4 | `blur_motion:1` | 26.7 | 6/10 | TRY_ALTERNATIVE | L | motion:1；26.70 拒绝 |
| 33 | R4 | `blur_glass:2,blur_gaussian:1` | 37.3 | 6/10 | ADJUST_SEVERITY | L! | 双 blur glass:2+gaussian:1；37.30 但 CI 6/10；违反每次最多一次 blur 的管线规则 |
| 34 | R4 | `blur_gaussian:1,blur_glass:2` | 36.75 | 7/10 | ADJUST_SEVERITY | L! | 反向双 blur；36.75 CI 7/10；同样违反管线规则 |
| 35 | R4 | `blur_lens:2` | 34.61 | 8/10 | ADJUST_SEVERITY | L | lens:2 复测；34.61 |
