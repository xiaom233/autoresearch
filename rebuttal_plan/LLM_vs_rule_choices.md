# LLM/AutoIR vs rule controller：逐场景退化选择对比

> GT 为事后标注；LLM 为 AutoIR 选中轮；rule 为 E1 选中轮。DIV2K/LSDIR 为固定 LQ test（同 A0/E1 指标）。`exact` = 函数+严重度+顺序全匹配；`type` = 函数+顺序匹配但 severity 不同；`category` = 类别顺序匹配；`order` = 类别相同但顺序不同；`same_len` = 步数相同但类别不同；`no` = 其他。

- 24 场景中，LLM 与 rule 选中管线相同 **4** 个，不同 **20** 个。
- LQ 配对：17/24 场景 A0/E1 的 LQ hash 完全一致；7/24 非确定性场景（0003/0005/0017/0018/0020/0022/0023）原始 A0/E1 的 `lq_sha256` 不一致（8,400 行中 2,450 行 = 7×350），原因是 `noise_impulse`/`blur_glass` 使用私有 RNG，同一 GT 参数会采到不同 realization；GT 管线没有变。已使用 `E1E2_fixed_lq` 的同一份固定 LQ 重新评估。下表 test 数字已对 7 个场景使用 fixed-LQ 值。
- GT exact：LLM **7**，rule **6**；GT type-only（函数+顺序匹配、severity 不同）：LLM **2**，rule **3**；GT order-only（类别相同、顺序不同）：LLM **1**，rule **3**。
- 差异标签计数：severity **17**、subtype **12**、step_count **10**、order **1**；LLM 步数更多 **6** 个场景，rule 更多 **4** 个。
- DIV2K test：LLM 胜 **16**、rule 胜 **7**、平 **1**；平均 Δ(LLM−rule) **+0.697 dB**。
- LSDIR test：LLM 胜 **15**、rule 胜 **6**、平 **3**；平均 Δ **+0.825 dB**。

## 表 1：选择差异（最终选中轮）

| 场景 | GT | LLM 选中 | rule 选中 | 差异类型/关键差异 | GT match (LLM / rule) |
|---|---|---|---|---|---|
| blind_0001 | `noise_poisson:5` | **R0**: `blur_gaussian:1` → `noise_poisson:4` | **R1**: `noise_poisson:5` | LLM 多 blur_gaussian:1；severity: noise_poisson 4 vs 5 | no / exact |
| blind_0002 | `noise_gaussian_YCrCb:2` | **R1**: `noise_gaussian_YCrCb:2` | **R0_fixed**: `noise_gaussian_YCrCb:2` | 相同 | exact / exact |
| blind_0003 | `blur_glass:2` | **R1**: `blur_gaussian:2` | **R1**: `blur_glass:2` | LLM 多 blur_gaussian:1；rule 多 blur_glass:1；子类型: blur: blur_gaussian vs blur_glass | category / exact |
| blind_0004 | `blur_motion:1` | **R2**: `blur_motion:1` | **R1**: `blur_motion:1` | 相同 | exact / exact |
| blind_0005 | `noise_impulse:5` | **R1**: `brightness_brighten_shfit_HSV:1` → `noise_impulse:3` | **R1**: `noise_impulse:5` | LLM 多 brightness_brighten_shfit_HSV:1；severity: noise_impulse 3 vs 5 | no / exact |
| blind_0006 | `compression_jpeg:1` | **R0**: `compression_jpeg:1` | **R1**: `compression_jpeg:3` | severity: compression_jpeg 1 vs 3 | exact / type |
| blind_0007 | `noise_speckle:1` | **R1**: `noise_speckle:1` | **R1**: `noise_speckle:1` | 相同 | exact / exact |
| blind_0008 | `saturate_weaken_HSV:4` | **R2**: `saturate_weaken_HSV:4` | **R1**: `saturate_weaken_HSV:5` | severity: saturate_weaken_HSV 4 vs 5 | exact / type |
| blind_0009 | `noise_poisson:2` → `saturate_strengthen_YCrCb:3` | **R2**: `noise_gaussian_RGB:5` → `brightness_darken_shfit_HSV:2` → `saturate_strengthen_HSV:5` | **R0**: `noise_gaussian_RGB:2` → `brightness_darken_shfit_RGB:2` → `contrast_strengthen_scale:5` | LLM 多 brightness_darken_shfit_HSV:1, saturate_strengthen_HSV:1；rule 多 brightness_darken_shfit_RGB:1, contrast_strengthen_scale:1；severity: noise_gaussian_RGB 5 vs 2；子类型: brightness: brightness_darken_shfit_HSV vs brightness_darken_shfit_RGB | no / no |
| blind_0010 | `blur_gaussian:1` → `brightness_darken_gamma_HSV:5` | **R0**: `brightness_darken_gamma_HSV:5` → `blur_gaussian:1` | **R1**: `brightness_darken_shfit_RGB:5` → `blur_gaussian:2` | LLM 多 brightness_darken_gamma_HSV:1；rule 多 brightness_darken_shfit_RGB:1；severity: blur_gaussian 1 vs 2；子类型: brightness: brightness_darken_gamma_HSV vs brightness_darken_shfit_RGB | order / order |
| blind_0011 | `noise_poisson:3` → `blur_gaussian:4` | **R0**: `blur_gaussian:4` | **R1**: `blur_gaussian:4` | 相同 | no / no |
| blind_0012 | `blur_motion:5` → `compression_jpeg:3` | **R2**: `noise_gaussian_RGB:4` → `blur_motion:5` → `compression_jpeg:2` | **R1**: `blur_gaussian:3` → `compression_jpeg:5` | LLM 多 noise_gaussian_RGB:1, blur_motion:1；rule 多 blur_gaussian:1；severity: compression_jpeg 2 vs 5；子类型: blur: blur_motion vs blur_gaussian | no / category |
| blind_0013 | `noise_speckle:3` → `compression_jpeg:2` | **R2**: `noise_impulse:3` → `compression_jpeg:2` | **R0**: `noise_gaussian_RGB:3` → `compression_jpeg:4` | LLM 多 noise_impulse:1；rule 多 noise_gaussian_RGB:1；severity: compression_jpeg 2 vs 4；子类型: noise: noise_impulse vs noise_gaussian_RGB | category / category |
| blind_0014 | `brightness_brighten_shfit_HSV:3` → `compression_jpeg:2` | **R2**: `brightness_brighten_shfit_HSV:3` → `compression_jpeg:2` | **R0**: `brightness_brighten_shfit_RGB:4` → `compression_jpeg:5` | LLM 多 brightness_brighten_shfit_HSV:1；rule 多 brightness_brighten_shfit_RGB:1；severity: compression_jpeg 2 vs 5；子类型: brightness: brightness_brighten_shfit_HSV vs brightness_brighten_shfit_RGB | exact / category |
| blind_0015 | `blur_motion:1` → `brightness_darken_shfit_RGB:3` | **R2**: `brightness_darken_gamma_RGB:3` | **R1**: `brightness_darken_shfit_RGB:3` → `blur_gaussian:2` | LLM 多 brightness_darken_gamma_RGB:1；rule 多 brightness_darken_shfit_RGB:1, blur_gaussian:1；子类型: brightness: brightness_darken_gamma_RGB vs brightness_darken_shfit_RGB | no / order |
| blind_0016 | `blur_motion:3` → `compression_jpeg:2` | **R1**: `blur_motion:3` → `compression_jpeg:4` | **R1**: `blur_motion:2` → `compression_jpeg:5` | severity: blur_motion 3 vs 2; compression_jpeg 4 vs 5 | type / type |
| blind_0017 | `noise_impulse:5` → `blur_gaussian:3` → `compression_jpeg:4` | **R2**: `noise_impulse:3` → `blur_gaussian:4` → `compression_jpeg:4` | **R0**: `blur_gaussian:3` → `compression_jpeg:5` | LLM 多 noise_impulse:1；severity: blur_gaussian 4 vs 3; compression_jpeg 4 vs 5 | type / no |
| blind_0018 | `noise_impulse:1` → `contrast_strengthen_stretch:4` → `compression_jpeg:5` | **R1**: `contrast_strengthen_stretch:4` → `compression_jpeg:5` | **R1**: `contrast_strengthen_scale:5` → `saturate_strengthen_HSV:4` → `compression_jpeg:5` | LLM 多 contrast_strengthen_stretch:1；rule 多 contrast_strengthen_scale:1, saturate_strengthen_HSV:1；子类型: contrast: contrast_strengthen_stretch vs contrast_strengthen_scale | no / same_len |
| blind_0019 | `blur_lens:4` → `compression_jpeg:4` → `saturate_weaken_HSV:5` | **R1**: `blur_gaussian:3` → `compression_jpeg:4` → `saturate_weaken_HSV:5` | **R1**: `blur_gaussian:3` → `saturate_weaken_HSV:3` → `compression_jpeg:5` | severity: compression_jpeg 4 vs 5; saturate_weaken_HSV 5 vs 3；顺序不同 | category / order |
| blind_0020 | `noise_poisson:3` → `blur_glass:4` → `contrast_weaken_stretch:4` | **R2**: `blur_glass:4` → `contrast_weaken_stretch:4` | **R1**: `blur_gaussian:5` → `contrast_weaken_stretch:5` → `saturate_weaken_HSV:5` | LLM 多 blur_glass:1；rule 多 blur_gaussian:1, saturate_weaken_HSV:1；severity: contrast_weaken_stretch 4 vs 5；子类型: blur: blur_glass vs blur_gaussian | no / same_len |
| blind_0021 | `blur_gaussian:3` → `brightness_brighten_shfit_RGB:1` → `compression_jpeg:4` | **R1**: `noise_impulse:1` → `blur_motion:5` → `compression_jpeg:4` | **R1**: `blur_gaussian:4` → `compression_jpeg:5` | LLM 多 noise_impulse:1, blur_motion:1；rule 多 blur_gaussian:1；severity: compression_jpeg 4 vs 5；子类型: blur: blur_motion vs blur_gaussian | same_len / no |
| blind_0022 | `noise_impulse:2` → `blur_glass:3` → `compression_jpeg:1` | **R2**: `blur_glass:3` → `compression_jpeg:1` | **R1**: `blur_glass:3` → `compression_jpeg:5` | severity: compression_jpeg 1 vs 5 | no / no |
| blind_0023 | `noise_gaussian_YCrCb:4` → `blur_glass:5` → `compression_jpeg:4` | **R2**: `noise_gaussian_YCrCb:4` → `blur_glass:5` → `compression_jpeg:4` | **R1**: `blur_gaussian:3` → `compression_jpeg:5` | LLM 多 noise_gaussian_YCrCb:1, blur_glass:1；rule 多 blur_gaussian:1；severity: compression_jpeg 4 vs 5；子类型: blur: blur_glass vs blur_gaussian | exact / no |
| blind_0024 | `noise_gaussian_RGB:4` → `blur_gaussian:3` → `saturate_strengthen_HSV:2` | **R0**: `brightness_darken_gamma_RGB:3` → `blur_gaussian:5` | **R0**: `brightness_darken_shfit_RGB:4` → `blur_gaussian:2` → `saturate_strengthen_HSV:5` | LLM 多 brightness_darken_gamma_RGB:1；rule 多 brightness_darken_shfit_RGB:1, saturate_strengthen_HSV:1；severity: blur_gaussian 5 vs 2；子类型: brightness: brightness_darken_gamma_RGB vs brightness_darken_shfit_RGB | no / same_len |

## 表 2：选择与性能（最终选中轮）

| 场景 | LLM 轮/架构 | rule 轮/架构 | support Δ | DIV2K LLM | DIV2K rule | Δ | 胜者 | LSDIR LLM | LSDIR rule | Δ | 胜者 |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| blind_0001 | R0/Swin | R1/Swin+SG | -4.79 | 25.85 | 30.44 | -4.59 | rule | 23.54 | 28.26 | -4.73 | rule |
| blind_0002 | R1/Swin+SG | R0_fixed/Swin+SG | -0.04 | 32.08 | 32.10 | -0.01 | tie | 30.02 | 30.03 | -0.01 | tie |
| blind_0003 | R1/Swin+SG | R1/Swin+SG | -2.56 | 25.53 | 27.94 | -2.41 | rule | 22.33 | 24.19 | -1.86 | rule |
| blind_0004 | R2/Swin+SG | R1/Swin+SG | +0.31 | 34.47 | 34.73 | -0.26 | rule | 30.31 | 30.64 | -0.32 | rule |
| blind_0005 | R1/Swin+SG | R1/Swin+SG | -21.00 | 21.92 | 44.91 | -22.99 | rule | 22.17 | 41.15 | -18.98 | rule |
| blind_0006 | R0/Swin | R1/Swin+SG | +0.39 | 31.78 | 31.47 | +0.31 | LLM | 28.69 | 28.37 | +0.32 | LLM |
| blind_0007 | R1/Swin+SG | R1/Swin+SG | +0.77 | 35.09 | 34.36 | +0.73 | LLM | 33.27 | 32.49 | +0.78 | LLM |
| blind_0008 | R2/Swin+SG | R1/Swin+SG | +21.46 | 43.18 | 22.65 | +20.53 | LLM | 41.41 | 20.20 | +21.21 | LLM |
| blind_0009 | R2/Swin+SG | R0/Swin+SG | -7.32 | 11.27 | 17.80 | -6.54 | rule | 11.04 | 16.75 | -5.71 | rule |
| blind_0010 | R0/Swin | R1/Swin+SG | +9.17 | 27.04 | 16.47 | +10.57 | LLM | 24.09 | 14.95 | +9.14 | LLM |
| blind_0011 | R0/Swin | R1/Swin+SG | -0.01 | 25.58 | 25.51 | +0.06 | LLM | 22.29 | 22.25 | +0.04 | tie |
| blind_0012 | R2/OCAB+SG | R1/Swin+SG | +3.25 | 22.28 | 19.64 | +2.64 | LLM | 20.47 | 17.65 | +2.82 | LLM |
| blind_0013 | R2/Swin+SG | R0/Swin+SG | +0.46 | 27.46 | 27.00 | +0.46 | LLM | 25.43 | 25.14 | +0.30 | LLM |
| blind_0014 | R2/Swin+SG | R0/Swin+SG | +3.91 | 26.98 | 23.64 | +3.34 | LLM | 24.91 | 21.96 | +2.94 | LLM |
| blind_0015 | R2/Swin+SG | R1/Swin+SG | +0.05 | 21.47 | 21.55 | -0.08 | rule | 20.10 | 19.44 | +0.67 | LLM |
| blind_0016 | R1/Swin+SG | R1/Swin+SG | +1.59 | 24.71 | 23.31 | +1.39 | LLM | 21.99 | 20.82 | +1.17 | LLM |
| blind_0017 | R2/Swin+SG | R0/Swin+SG | +0.41 | 23.51 | 23.41 | +0.10 | LLM | 20.88 | 20.92 | -0.04 | tie |
| blind_0018 | R1/Swin+SG+GCM | R1/Swin+SG | +2.78 | 19.56 | 15.98 | +3.58 | LLM | 18.89 | 15.91 | +2.98 | LLM |
| blind_0019 | R1/Swin+SG | R1/Swin+SG | +1.74 | 20.01 | 17.95 | +2.06 | LLM | 17.83 | 15.60 | +2.23 | LLM |
| blind_0020 | R2/Swin+SG+ColorPre | R1/Swin+SG | +2.12 | 18.08 | 15.99 | +2.09 | LLM | 17.41 | 15.23 | +2.18 | LLM |
| blind_0021 | R1/OCAB+SG | R1/Swin+SG | -0.98 | 17.61 | 18.88 | -1.26 | rule | 16.53 | 17.78 | -1.25 | rule |
| blind_0022 | R2/Swin+SG | R1/Swin+SG | +0.42 | 25.48 | 25.05 | +0.43 | LLM | 22.34 | 21.97 | +0.37 | LLM |
| blind_0023 | R2/Swin+SG | R1/Swin+SG+GCM | +2.17 | 23.06 | 20.76 | +2.30 | LLM | 20.40 | 18.72 | +1.68 | LLM |
| blind_0024 | R0/Swin | R0/Swin+SG | +3.30 | 15.25 | 10.99 | +4.27 | LLM | 14.61 | 10.73 | +3.87 | LLM |

## 表 3：初始 R0 选择（无反思/初始预测）

| 场景 | GT | LLM R0 | rule R0（R0_fixed 优先） | 差异 | GT match (LLM / rule) |
|---|---|---|---|---|---|
| blind_0001 | `noise_poisson:5` | `blur_gaussian:1` → `noise_poisson:4` | `noise_poisson:4` | LLM 多 blur_gaussian:1 | no / type |
| blind_0002 | `noise_gaussian_YCrCb:2` | `noise_gaussian_YCrCb:2` | `noise_gaussian_YCrCb:2` | 相同 | exact / exact |
| blind_0003 | `blur_glass:2` | `blur_gaussian:2` | `blur_gaussian:2` | 相同 | category / category |
| blind_0004 | `blur_motion:1` | `blur_motion:1` | `blur_gaussian:1` | LLM 多 blur_motion:1；rule 多 blur_gaussian:1；子类型: blur: blur_motion vs blur_gaussian | exact / category |
| blind_0005 | `noise_impulse:5` | `brightness_brighten_shfit_HSV:1` → `noise_impulse:3` | `noise_impulse:3` | LLM 多 brightness_brighten_shfit_HSV:1 | no / type |
| blind_0006 | `compression_jpeg:1` | `compression_jpeg:1` | `compression_jpeg:5` | severity: compression_jpeg 1 vs 5 | exact / type |
| blind_0007 | `noise_speckle:1` | `noise_speckle:1` | `noise_speckle:2` | severity: noise_speckle 1 vs 2 | exact / type |
| blind_0008 | `saturate_weaken_HSV:4` | `saturate_weaken_HSV:4` | `brightness_brighten_shfit_RGB:2` → `saturate_weaken_HSV:5` | rule 多 brightness_brighten_shfit_RGB:1；severity: saturate_weaken_HSV 4 vs 5 | exact / no |
| blind_0009 | `noise_poisson:2` → `saturate_strengthen_YCrCb:3` | `noise_gaussian_YCrCb:2` → `brightness_darken_shfit_HSV:2` → `saturate_strengthen_HSV:5` | `noise_gaussian_YCrCb:2` → `brightness_darken_shfit_RGB:2` → `contrast_strengthen_scale:5` | LLM 多 brightness_darken_shfit_HSV:1, saturate_strengthen_HSV:1；rule 多 brightness_darken_shfit_RGB:1, contrast_strengthen_scale:1；子类型: brightness: brightness_darken_shfit_HSV vs brightness_darken_shfit_RGB | no / no |
| blind_0010 | `blur_gaussian:1` → `brightness_darken_gamma_HSV:5` | `brightness_darken_gamma_HSV:5` → `blur_gaussian:1` | `brightness_darken_shfit_RGB:5` → `blur_gaussian:2` → `contrast_weaken_scale:5` | LLM 多 brightness_darken_gamma_HSV:1；rule 多 brightness_darken_shfit_RGB:1, contrast_weaken_scale:1；severity: blur_gaussian 1 vs 2；子类型: brightness: brightness_darken_gamma_HSV vs brightness_darken_shfit_RGB | order / no |
| blind_0011 | `noise_poisson:3` → `blur_gaussian:4` | `blur_gaussian:4` | `blur_gaussian:3` | severity: blur_gaussian 4 vs 3 | no / no |
| blind_0012 | `blur_motion:5` → `compression_jpeg:3` | `noise_gaussian_RGB:4` → `blur_motion:5` → `compression_jpeg:2` | `noise_gaussian_YCrCb:3` → `blur_gaussian:3` → `compression_jpeg:5` | LLM 多 noise_gaussian_RGB:1, blur_motion:1；rule 多 noise_gaussian_YCrCb:1, blur_gaussian:1；severity: compression_jpeg 2 vs 5；子类型: blur: blur_motion vs blur_gaussian; noise: noise_gaussian_RGB vs noise_gaussian_YCrCb | no / no |
| blind_0013 | `noise_speckle:3` → `compression_jpeg:2` | `noise_impulse:3` → `compression_jpeg:2` | `noise_gaussian_YCrCb:3` → `compression_jpeg:4` | LLM 多 noise_impulse:1；rule 多 noise_gaussian_YCrCb:1；severity: compression_jpeg 2 vs 4；子类型: noise: noise_impulse vs noise_gaussian_YCrCb | category / category |
| blind_0014 | `brightness_brighten_shfit_HSV:3` → `compression_jpeg:2` | `blur_glass:1` → `compression_jpeg:1` | `brightness_brighten_shfit_RGB:4` → `compression_jpeg:5` | LLM 多 blur_glass:1；rule 多 brightness_brighten_shfit_RGB:1；severity: compression_jpeg 1 vs 5 | same_len / category |
| blind_0015 | `blur_motion:1` → `brightness_darken_shfit_RGB:3` | `brightness_darken_gamma_RGB:3` | `brightness_darken_shfit_RGB:4` → `blur_gaussian:2` | LLM 多 brightness_darken_gamma_RGB:1；rule 多 brightness_darken_shfit_RGB:1, blur_gaussian:1；子类型: brightness: brightness_darken_gamma_RGB vs brightness_darken_shfit_RGB | no / order |
| blind_0016 | `blur_motion:3` → `compression_jpeg:2` | `blur_motion:3` → `compression_jpeg:4` | `blur_gaussian:2` → `compression_jpeg:5` | LLM 多 blur_motion:1；rule 多 blur_gaussian:1；severity: compression_jpeg 4 vs 5；子类型: blur: blur_motion vs blur_gaussian | type / category |
| blind_0017 | `noise_impulse:5` → `blur_gaussian:3` → `compression_jpeg:4` | `blur_gaussian:4` → `compression_jpeg:4` | `blur_gaussian:3` → `compression_jpeg:5` | severity: blur_gaussian 4 vs 3; compression_jpeg 4 vs 5 | no / no |
| blind_0018 | `noise_impulse:1` → `contrast_strengthen_stretch:4` → `compression_jpeg:5` | `contrast_strengthen_stretch:4` → `compression_jpeg:5` | `contrast_strengthen_scale:5` → `saturate_strengthen_HSV:4` → `compression_jpeg:5` | LLM 多 contrast_strengthen_stretch:1；rule 多 contrast_strengthen_scale:1, saturate_strengthen_HSV:1；子类型: contrast: contrast_strengthen_stretch vs contrast_strengthen_scale | no / same_len |
| blind_0019 | `blur_lens:4` → `compression_jpeg:4` → `saturate_weaken_HSV:5` | `compression_jpeg:4` → `blur_gaussian:5` → `saturate_weaken_HSV:5` | `blur_gaussian:3` → `saturate_weaken_HSV:1` → `compression_jpeg:5` | severity: blur_gaussian 5 vs 3; compression_jpeg 4 vs 5; saturate_weaken_HSV 5 vs 1；顺序不同 | order / order |
| blind_0020 | `noise_poisson:3` → `blur_glass:4` → `contrast_weaken_stretch:4` | `blur_gaussian:4` → `contrast_weaken_stretch:4` | `blur_gaussian:5` → `contrast_weaken_scale:5` → `saturate_weaken_HSV:5` | LLM 多 contrast_weaken_stretch:1；rule 多 contrast_weaken_scale:1, saturate_weaken_HSV:1；severity: blur_gaussian 4 vs 5；子类型: contrast: contrast_weaken_stretch vs contrast_weaken_scale | no / same_len |
| blind_0021 | `blur_gaussian:3` → `brightness_brighten_shfit_RGB:1` → `compression_jpeg:4` | `compression_jpeg:4` → `blur_motion:5` → `noise_impulse:1` | `noise_gaussian_YCrCb:3` → `blur_gaussian:4` → `compression_jpeg:5` | LLM 多 blur_motion:1, noise_impulse:1；rule 多 noise_gaussian_YCrCb:1, blur_gaussian:1；severity: compression_jpeg 4 vs 5；子类型: blur: blur_motion vs blur_gaussian; noise: noise_impulse vs noise_gaussian_YCrCb | same_len / same_len |
| blind_0022 | `noise_impulse:2` → `blur_glass:3` → `compression_jpeg:1` | `blur_gaussian:3` → `compression_jpeg:1` | `blur_gaussian:3` → `compression_jpeg:5` | severity: compression_jpeg 1 vs 5 | no / no |
| blind_0023 | `noise_gaussian_YCrCb:4` → `blur_glass:5` → `compression_jpeg:4` | `noise_gaussian_RGB:3` → `blur_gaussian:5` → `compression_jpeg:4` | `blur_gaussian:3` → `contrast_weaken_scale:4` → `compression_jpeg:5` | LLM 多 noise_gaussian_RGB:1；rule 多 contrast_weaken_scale:1；severity: blur_gaussian 5 vs 3; compression_jpeg 4 vs 5 | category / same_len |
| blind_0024 | `noise_gaussian_RGB:4` → `blur_gaussian:3` → `saturate_strengthen_HSV:2` | `brightness_darken_gamma_RGB:3` → `blur_gaussian:5` | `brightness_darken_shfit_RGB:4` → `blur_gaussian:2` → `saturate_strengthen_HSV:5` | LLM 多 brightness_darken_gamma_RGB:1；rule 多 brightness_darken_shfit_RGB:1, saturate_strengthen_HSV:1；severity: blur_gaussian 5 vs 2；子类型: brightness: brightness_darken_gamma_RGB vs brightness_darken_shfit_RGB | no / same_len |

## 模式总结

- **LLM 常见的额外步骤**：brightness、blur、saturation、noise；例如 0001 多 blur、0009 多 brightness、0017 多 impulse、0021 多 motion/impulse。
- **rule 常见的简单管线**：在多数场景只保留 1-2 步（如 0001 poisson:5、0005 impulse:5、0008 saturate:5），severity 往往取 5。
- **子类型分歧**：0003 blur gaussian vs glass；0010 brightness gamma/HSV vs shift/RGB；0018 contrast stretch vs scale；0019 blur gaussian vs lens（两者都错，但 LLM 保留 JPEG/sat）。
- **顺序分歧**：0010 LLM gamma→blur，GT 是 blur→gamma；0019 rule sat→JPEG，GT/LLM 是 JPEG→sat。
- **LLM 大胜场景**：0008（severity 4 vs 5）、0014（LLM 精确 GT）、0005（rule 修正 impulse:5）、0001（rule poisson:5）等——注意这些并不全是“LLM 反思”单独贡献，有的是一开始 R0 就选对。
- **rule 大胜场景**：0005（impulse:5 vs brightness+impulse:3）、0003（glass vs gaussian）、0009（rule 17.8 vs LLM 11.27）。
- **两边都失败**：0009/0011/0017/0024 等仍有漏检/子类型错误；exact GT 不是性能的充分条件，性能也不是 GT 正确的充分条件。

## 关键 caveat

- 选中轮不同（LLM 有 5 R0/8 R1/11 R2；rule 有 5 R0/1 R0_fixed/18 R1/0 R2），因此表 1/2 的差异同时包含 controller 和 reflection 的差异；初始 R0 差异见表 3。
- test 指标：17 个确定性场景 A0/E1 的 LQ hash 一致；7 个非确定性场景使用 E1E2_fixed_lq 的同一份固定 LQ。checkpoint 选择规则仍不同；不要把 Δ 全部归因于管线。
- GT 是单场景单次的隐藏参数；exact/type 是描述性匹配，不是因果结论。
- blind_0020 的 rule R0 训练 NaN；表 3 的 rule R0 管线仍可读，但其模型不可用。
- 本表只覆盖 24 个 exp41 场景，不外推到新场景。
