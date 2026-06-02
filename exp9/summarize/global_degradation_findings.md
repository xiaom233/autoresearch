# 全局退化 (Brightness/Contrast/Saturation) 实验总结

## 实验设计

6 组全新退化，包含全局退化（brightness/contrast/saturation）与局部退化的组合。

### 双退化

| ID | 退化 | 类型 |
|----|------|------|
| L1 | blur_gaussian(3) + brightness_darken_shfit_HSV(3) | blur + 暗化 |
| L2 | noise_gaussian_RGB(3) + contrast_weaken_scale(3) | noise + 低对比度 |
| L3 | compression_jpeg(3) + brightness_brighten_shfit_RGB(2) | JPEG + 亮化 |

### 三退化

| ID | 退化 | 类型 |
|----|------|------|
| L4 | blur_gaussian(3) + noise_gaussian_RGB(3) + brightness_darken_shfit_HSV(3) | 标准 + 暗化 |
| L5 | blur_motion(3) + compression_jpeg(3) + contrast_weaken_scale(3) | motion+JPEG+低对比度 |
| L6 | noise_impulse(3) + blur_lens(3) + saturate_weaken_HSV(3) | impulse+lens+去饱和 |

每组 4 策略：Direct / Ft / Curric(fwd) / Curric(rev)

## 已完成结果 (L1-L2)

### L1: blur_gaussian(3) + brightness_darken(3)

| 策略 | PSNR |
|------|:--:|
| Direct | 27.66 |
| Ft | 28.07 |
| Curric(fwd) | 24.95 |
| **Curric(rev)** | **28.06** |

- Ft > Direct +0.41 (微弱)
- Rev >> Fwd +3.11 (brightness作为外层先剥离天然合理)
- **Rev ≈ Ft**：全局退化下 Curric(rev) 可以追平 Ft

### L2: noise_gaussian(3) + contrast_weaken(3)

| 策略 | PSNR |
|------|:--:|
| **Direct** | **27.30** |
| Ft | 22.78 |
| Curric(fwd) | 22.73 |
| Curric(rev) | 22.76 |

- **Ft 崩了 -4.51！** Direct 远优于 Ft
- 所有 Curric 也崩溃，策略差异 < 0.1 dB

## 关键发现

### L1：全局退化 + blur — 标准规律仍成立

- Ft 微弱优于 Direct (+0.41)，与 blur+noise 规律一致
- Rev 明显优于 Fwd (+3.11)：brightness 在外层，自然先剥离
- Curric(rev) ≈ Ft：为全局退化单独使用 Curric 有竞争力

### L2：全局退化 + noise — **Ft 崩溃**

- Ft 比 Direct 差 -4.51 dB — 这是极其罕见的
- 原因假设：盲预训练中 **从未见过 contrast_weaken** 这种全局退化，盲预训练的特征对对比度变化产生了负迁移
- 对比度是全局像素级变换，盲预训练的局部修复特征不仅无用，反而干扰

## 待完成

L3-L6 正在训练中，完成后补充。

## 初步结论

1. **全局退化的耦合规律与局部退化不同** — Ft 不是安全的默认选择
2. **brightness 与 blur 交互弱** — 类似 noise+blur，标准规律适用
3. **contrast 与 noise 交互特殊** — Ft 严重负迁移（-4.51），Direct 最优
4. **全局退化需要独立研究** — 不能简单套用局部退化的经验
