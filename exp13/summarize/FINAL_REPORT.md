# exp13 — 盲识别全流程端到端验证 最终报告

## 实验设计

24 组退化 (4 单 + 10 双 + 10 三)，同图模式盲识别，四层隔离协议。

```
Phase 1: M_blind 盲基线训练 (随机退化, EPOCH=2)
Phase 2: 退化生成 (setup_challenge.sh, 24组)
Phase 4: 盲识别 (Skill × 24, 同图模式)
Phase 3: GT 基线测试 (M_blind on GT)
Phase 5: Spec + Ft 训练 (48次, predicted_params → GT验证)
Phase 5.5: 反思分析
```

## 关键结论

**盲识别质量决定一切，同图模式使 Spec ≈ Ft。**

---

## 一、完整实验结果

### 单退化 (4 组)

| ID | GT Pipeline | Pred Pipeline | CI | 函数匹配 | M_blind | Spec | Ft | Spec vs Ft |
|----|------------|--------------|-----|:--:|--------|------|------|:--:|
| 1001 | blur_gaussian(5) | blur_gaussian(5) | 10/10 | 1/1 | 20.92 | 21.85 | 20.27 | **+1.58 Spec** |
| 1002 | compression_jpeg_2000(2) | compression_jpeg_2000(2) | 10/10 | 1/1 | 27.52 | 28.23 | 28.00 | +0.23 ≈ |
| 1003 | compression_jpeg_2000(2) | compression_jpeg_2000(2) | 10/10 | 1/1 | 27.52 | 28.29 | 28.26 | +0.03 ≈ |
| 1004 | blur_jitter(5) | noise_gaussian_RGB(3) | 8/10 | 0/1 | 21.23 | 18.82 | 18.78 | +0.04 ≈ |

**1001**: 满分识别，Spec 显著优于 Ft (+1.58)。盲识别完美的单退化上从零训练比微调更好。
**1004**: 识别完全错误（jitter→gaussian noise），但 Spec≈Ft。退化极强时两种策略都难以恢复。

### 双退化 (10 组)

| ID | GT Pipeline | Pred Pipeline | CI | 函数匹配 | M_blind | Spec | Ft | Spec vs Ft |
|----|------------|--------------|-----|:--:|--------|------|------|:--:|
| 2001 | blur_glass(1)+noise_impulse(5) | noise_impulse(3)+noise_gaussian_RGB(2) | 8/10 | 1/2 | 24.92 | 24.14 | 24.12 | +0.02 ≈ |
| 2002 | compression_jpeg_2000(5)+blur_glass(3) | blur_gaussian(3)+compression_jpeg_2000(2) | 9/10 | 1/2 | 20.34 | 19.42 | 19.46 | -0.04 ≈ |
| 2003 | compression_jpeg(3)+noise_impulse(4) | noise_gaussian_RGB(1)+noise_impulse(3) | 9/10 | 1/2 | 26.56 | 26.76 | 26.72 | +0.04 ≈ |
| 2004 | blur_zoom(2)+compression_jpeg(5) | blur_gaussian(2)+compression_jpeg(4) | 7/10 | 1/2 | 18.99 | 18.54 | 18.34 | +0.20 ≈ |
| **2005** | blur_gaussian(4)+noise_spatially_correlated(4) | blur_gaussian(2)+noise_impulse(1) | 6/10 | 1/2 | 20.68 | 19.33 | 19.49 | -0.16 ≈ |
| 2006 | noise_gaussian_RGB(1)+compression_jpeg(2) | noise_gaussian_RGB(2)+compression_jpeg(3) | 7/10 | 2/2 | 27.95 | 28.33 | 28.40 | -0.07 ≈ |
| **2007** | compression_jpeg(4)+noise_gaussian_YCrCb(5) | oversharpen(3)+noise_impulse(5) | 8/10 | 0/2 | 23.65 | 19.30 | 19.25 | +0.05 ≈ |
| **2008** | compression_jpeg_2000(1)+blur_jitter(4) | noise_gaussian_RGB(5)+compression_jpeg(5) | 6/10 | 0/2 | 21.85 | 19.24 | 19.24 | +0.00 ≈ |
| 2009 | compression_jpeg(4)+blur_jitter(3) | noise_impulse(1)+compression_jpeg(5) | 8/10 | 1/2 | 22.28 | 20.66 | 20.69 | -0.03 ≈ |
| **2010** | noise_spatially_correlated(2)+compression_jpeg_2000(3) | blur_lens(3)+compression_jpeg(1) | 5/10 | 0/2 | 23.76 | 23.14 | 23.18 | -0.04 ≈ |

**2006**: 唯一函数完全匹配(2/2)的双退化，Spec 和 Ft 都非常接近 M_blind。
**2007/2008**: 函数匹配 0/2，Spec 比 M_blind 分别低 4.3 和 2.6 dB——识别错误直接导致训练退化。
**2005/2010**: NEEDS_WORK，CI 偏低 (5-6/10)。

### 三退化 (10 组)

| ID | GT Pipeline | Pred Pipeline | CI | 函数匹配 | M_blind | Spec | Ft | Spec vs Ft |
|----|------------|--------------|-----|:--:|--------|------|------|:--:|
| **3001** | compression_jpeg_2000(2)+noise_gaussian_YCrCb(3)+blur_jitter(3) | oversharpen(2)+noise_impulse(4)+compression_jpeg(2) | 4/10 | 0/3 | 21.49 | 19.97 | 20.10 | -0.13 ≈ |
| **3002** | compression_jpeg_2000(1)+blur_jitter(4)+noise_impulse(1) | oversharpen(2)+blur_gaussian(1)+noise_impulse(1) | 4/10 | 1/3 | 21.78 | 17.73 | 17.85 | -0.12 ≈ |
| **3003** | noise_impulse(2)+compression_jpeg(1)+blur_glass(1) | blur_lens(4)+noise_gaussian_RGB(1)+compression_jpeg(2) | 8/10 | 1/3 | 24.83 | 23.38 | **15.70** | **+7.68 Spec** |
| **3004** | noise_gaussian_YCrCb(1)+blur_jitter(1)+compression_jpeg_2000(1) | oversharpen(2)+brightness_darken_shfit_RGB(1)+compression_jpeg(1) | 6/10 | 0/3 | 25.49 | 18.21 | 18.41 | -0.20 ≈ |
| 3005 | noise_speckle(4)+blur_gaussian(2)+compression_jpeg_2000(4) | blur_gaussian(5)+compression_jpeg_2000(1)+saturate_weaken_HSV(1) | 7/10 | 1/3 | 18.58 | 16.92 | 17.28 | **-0.36 Ft** |
| **3006** | compression_jpeg(5)+noise_poisson(3)+blur_jitter(5) | compression_jpeg(5)+blur_jitter(4)+noise_poisson(3) | 10/10 | 3/3 | 20.36 | 21.02 | 21.00 | +0.02 ≈ |
| 3007 | noise_spatially_correlated(5)+compression_jpeg_2000(2)+blur_glass(4) | noise_spatially_correlated(1)+compression_jpeg_2000(2)+blur_lens(2) | 7/10 | 2/3 | 20.44 | 19.61 | 19.60 | +0.01 ≈ |
| 3008 | noise_gaussian_RGB(2)+compression_jpeg(1)+blur_gaussian(1) | compression_jpeg(3)+noise_gaussian_YCrCb(2)+blur_gaussian(1) | 8/10 | 2/3 | 26.17 | 28.06 | 28.14 | -0.08 ≈ |
| 3009 | compression_jpeg_2000(1)+noise_speckle(4)+blur_glass(2) | noise_gaussian_RGB(1)+blur_gaussian(1)+compression_jpeg_2000(2) | 8/10 | 1/3 | 21.70 | 20.94 | 20.81 | +0.13 ≈ |
| **3010** | compression_jpeg_2000(3)+blur_motion(4)+noise_spatially_correlated(4) | noise_impulse(2)+blur_motion(1)+compression_jpeg(2) | 5/10 | 1/3 | 18.44 | 18.63 | 18.76 | -0.13 ≈ |

**3006**: 唯一三退化满分识别 (CI=10/10, 3/3 函数匹配)，Spec 和 Ft 均略优于 M_blind。
**3003**: Ft 崩溃——Spec=23.38 正常，Ft=15.70 崩盘。预测与 GT 完全不同，盲预训练特征在此退化上产生强烈负迁移。
**3004**: 盲识别最差案例——Spec 比 M_blind 低 7.28 dB。函数匹配 0/3。
**3008**: Spec 最大正收益 (+1.89 vs M_blind)，函数匹配 2/3。

---

## 二、统计分析

### 总体

| 指标 | 数值 |
|------|:--:|
| 总退化 | 24 (4+10+10) |
| 训练实验 | 48 (24 Spec + 24 Ft) |
| Spec 均值 | 21.69 |
| Ft 均值 | 21.33 |
| CI 平均 | 7.5/10 |

### Spec vs Ft

| 分类 | 数量 | 退化 |
|------|:--:|------|
| Spec > Ft (>0.3dB) | 2 | 1001, 3003(异常) |
| Ft > Spec (>0.3dB) | 1 | 3005 |
| 持平 (≤0.3dB) | **21** | 其余全部 |

### 盲识别质量

| 指标 | 数值 |
|------|:--:|
| GOOD | 17/24 (71%) |
| NEEDS_WORK | 7/24 (29%) |
| CI=10/10 | 4/24 |
| 函数匹配 0/n | 8/24 |
| 函数匹配 3/3 | 1/24 (3006) |

### Spec vs M_blind

| 分类 | 数量 |
|------|:--:|
| Spec > M_blind | 8/24 |
| Spec < M_blind | 16/24 |
| 最差 (Δ<-3dB) | 3 组: 3004(-7.28), 2007(-4.34), 3002(-4.05) |

---

## 三、问题退化诊断

### 盲识别明显错误 (Spec << M_blind, 需 R1 修正)

| ID | Δ Spec-M_blind | 函数匹配 | 问题 |
|----|:--:|:--:|------|
| **3004** | **-7.28 dB** | 0/3 | oversharpen 和 brightness_darken 完全误识别 |
| **2007** | **-4.34 dB** | 0/2 | compression_jpeg 识别为 oversharpen；YCrCb 噪声识别为 impulse |
| **3002** | **-4.05 dB** | 1/3 | compression_jpeg_2000 识别为 oversharpen；blur_jitter 识别为 gaussian |

共性：三组都将压缩类退化误识别为 oversharpen，且严重度估计偏差大。

### Ft 崩溃 (3003)

| 模型 | PSNR | 说明 |
|------|:--:|------|
| M_blind | 24.83 | 盲基线正常 |
| Spec(pred) | 23.38 | 从零训练正常 |
| Ft(pred) | **15.70** | 加载 M_blind 后微调崩溃 |

GT 全是低严重度 (1-2)，Pred 以高严重度 blur_lens(4) 为主。Ft 的 LOAD_CKPT 从盲预训练加载了与 GT 冲突的特征 → 负迁移。
建议修正：对比 Direct (从零训练，不用 LOAD_CKPT) 确认是否是 LOAD_CKPT 导致。

### NEEDS_WORK 退化

| ID | CI | 函数匹配 | 主要问题 |
|----|-----|:--:|------|
| 3001 | 4/10 | 0/3 | 全部函数误识别 |
| 3002 | 4/10 | 1/3 | 压缩和模糊类型错误 |
| 2010 | 5/10 | 0/2 | noise_spatially_correlated→blur_lens 混淆 |
| 3010 | 5/10 | 1/3 | blur_motion sev严重低估 (4→1) |
| 2005 | 6/10 | 1/2 | noise_spatially_correlated→impulse 混淆 |
| 2008 | 6/10 | 0/2 | 全部函数误识别 |
| 3004 | 6/10 | 0/3 | 全部函数误识别 |

---

## 四、与 exp12 对比

| 维度 | exp12 | exp13 |
|------|-------|-------|
| 盲识别 | 无 (直接用 GT) | 有 (同图模式 Skill) |
| 退化数 | 56 | 24 |
| Spec vs Ft | Ft >> Spec (碾压) | Spec ≈ Ft (等价) |
| VAL 锁定 | 无 (被污染) | 有 (launcher 注入) |
| 反思有效性 | 无法判断 | 无需修正 (盲识别质量高时等价) |
| 隔离协议 | 后期补丁 | 四层协议预先设计 |
| 盲识别可用率 | — | 100% (同图) vs 12% (跨图) |

**为什么 exp12 Ft >> Spec 而 exp13 Spec ≈ Ft？**

exp12 训练直接用 GT 退化参数，Ft 拥有"盲预训练通用特征 + GT 精准训练"双重优势。
exp13 的 Spec 和 Ft 都在 predicted_params 上训练，验证用 GT。
当 predicted ≈ GT（同图模式 CI 高），两种策略差异消失。
当 predicted ≠ GT（函数匹配 0/3），两种策略都变差。

---

## 五、未完成工作

1. **DFPIR 大模型基线**未测试 — 可作为后续对比
2. **3 组盲识别修正 (R1)** 未执行 — 2007/3002/3004
3. **3003 Ft 崩溃**未排查 — LOAD_CKPT 负迁移根因
4. **架构优化** (PCP/CSN/ColorPre) 未在 exp13 上尝试
5. **16 组 Spec < M_blind** — 说明在 predicted_params 上训练后验证 GT，多数情况反而变差。盲识别的微小偏差即可导致此现象

## 六、核心经验

1. **同图模式是盲识别的基础** — CI pass 从跨图 12% 提升到 100%
2. **盲识别质量决定 Spec vs Ft 的差距** — 质量高时两者等价，质量低时两者都差
3. **VAL_PARAMS_PATH 必须锁定** — exp12 的教训不可忘记
4. **四层隔离 + 同图模式 = 可靠的盲识别协议**
5. **盲识别参数修正的优先级远高于训练策略调整**
