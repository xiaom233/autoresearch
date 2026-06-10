# exp11：误识别退化 → 失败模式 → 修正

## 目标

给定一个 GT 退化管线，Phase 4 的盲识别可能产生错误（类型/严重度/遗漏/顺序）。在**错误退化上训练的模型**，在 GT 退化上测试时，会产生**特定的失败模式**。

目标：系统刻画"识别误差类型 → 模型失败模式"的映射关系，构建可操作的修正规则。

## 核心方法论

```
GT 退化 P_gt
    │
Phase 4 识别 → P_pred (可能有误差)
    │
训练 M_spec(P_pred)
    │
在 P_gt 上测试 M_spec → 残差图 + 逐数据集PSNR
    │
分析失败模式 → 推断 P_pred 的误差 → 修正 P_pred
```

**关键**：我们不直接访问 P_gt（那是盲识别的目标），而是通过**模型在多个测试集上的表现模式**间接推断。

## 实验矩阵

### 矩阵 A：类型误识别（Type Mismatch）— 新实验 10 组

以 L6 (impulse+lens+saturate, sev=3) 为 GT 基准：

| ID | P_pred（错误训练退化） | 误差类型 | 验证什么 |
|----|------|:--:|------|
| A1 | gaussian_noise+lens+saturate | blur类型错(impulse→gaussian) | 噪声类型误判的残差模式 |
| A2 | impulse+gaussian_blur+saturate | blur类型错(lens→gaussian) | 模糊类型误判的残差模式 |
| A3 | impulse+lens+contrast_scale | 全局类型错(saturate→contrast) | 全局退化误判的残差模式 |
| A4 | impulse+lens | **遗漏**全局退化 | 遗漏 saturate 的残差模式 |
| A5 | lens+saturate | **遗漏**噪声 | 遗漏 impulse 的残差模式 |

以 L2 (noise+contrast, sev=3) 为 GT 基准：

| ID | P_pred | 误差类型 | 验证什么 |
|----|------|:--:|------|
| A6 | impulse_noise+contrast | noise子类型错 | 噪声子类型误判 |
| A7 | gaussian_noise+brightness_shift | 全局类型错 | contrast→brightness 误判 |
| A8 | gaussian_noise | **遗漏**全局 | 遗漏 contrast |

以 L5 (motion+jpeg+contrast, sev=3) 为 GT 基准：

| ID | P_pred | 误差类型 | 验证什么 |
|----|------|:--:|------|
| A9 | gaussian_blur+jpeg+contrast | blur类型错(motion→gaussian) | 方向性模糊被误判为各向同性 |
| A10 | motion+jpeg | **遗漏**全局 | 遗漏 contrast |

### 矩阵 B：严重度误识别（Severity Mismatch）— 利用已有数据

已有实验提供了天然的 severity mismatch：

| 退化 | 训练 sev | GT sev | 已有? |
|------|:--:|:--:|:--:|
| L2 contrast | 1 | 3 | ✅ P3_L2s1_Swin |
| L2 contrast | 5 | 3 | ✅ P3_L2s5_Swin |
| L2 contrast | 3 | 1 | 需交叉测试 |
| L2 contrast | 3 | 5 | 需交叉测试 |

只需交叉测试：将已有 sev=1/5 的模型在 sev=3 的 GT 上评估。

### 矩阵 C：遗漏退化（Missing Degradation）— 利用 Cascade 数据

Cascade Phase A 实验正是"遗漏全局退化"的完美案例：

| 模型训练退化 | GT 退化 | 已有? |
|------|------|:--:|
| L6_spatial (impulse+lens) | L6_full (impulse+lens+saturate) | ✅ Phase B 已测 |
| L2_spatial (noise) | L2_full (noise+contrast) | ✅ Phase B 已测 |
| L5_spatial (motion+jpeg) | L5_full (motion+jpeg+contrast) | ✅ Phase B 已测 |
| L1_spatial (blur) | L1_full (blur+brightness_HSV) | ✅ Phase B 已测 |
| L4_spatial (blur+noise) | L4_full (blur+noise+brightness_HSV) | ✅ Phase B 已测 |

### 矩阵 D：顺序误识别（Order Mismatch）— 利用已有数据

reversed 退化实验正是顺序误识别的案例：

| 模型训练退化 | GT 退化 | 已有? |
|------|------|:--:|
| L6_rev (saturate→impulse→lens) | L6 (impulse→lens→saturate) | 需交叉测试 |
| L2_rev (contrast→noise) | L2 (noise→contrast) | 需交叉测试 |
| L5_rev (contrast→motion→jpeg) | L5 (motion→jpeg→contrast) | 需交叉测试 |

## 实施计划

### Step 1：交叉测试脚本（~1h 开发）

创建 `exp11/scripts/cross_test.py`：
- 加载任意 checkpoint
- 在任意退化上评估
- 输出：per-dataset PSNR + 残差图统计 + 频域分析

```python
# 用法
uv run exp11/scripts/cross_test.py \
  --ckpt exp10/experiments/G9r_L6_Swin_lr5e4/checkpoints/G9r_L6_Swin_lr5e4_step15092.pt \
  --params exp10/degradation/L6_spatial.json \
  --output exp11/results/L6_Swin_on_L6spatial.json
```

### Step 2：矩阵 A 新实验（10 组 × 86 min ≈ 15 GPU-hours）

优先执行 A1-A5（L6 基准），最丰富的失败模式。

### Step 3：交叉测试已有模型（利用 571 checkpoint）

无需新训练——直接用交叉测试脚本在已有 checkpoint 上运行不同退化评估。

| 交叉测试 | 利用数据 | 组数 |
|------|------|:--:|
| Severity mismatch | P3_L2s1/S5 Swin | 4 |
| Missing degradation | Cascade Phase A spatial Swin | 5 |
| Order mismatch | reversed Swin | 3 |
| Type mismatch cross-test | 随机采样 10 个已有模型 | 10 |
| **合计** | | **22** |

### Step 4：构建诊断规则

汇总所有交叉测试和矩阵 A 的结果，提取特征→误差映射：

```
诊断特征向量:
  - per_dataset_psnr: [Set5, Set14, B100, Urban100, Manga109, DIV2K]
  - residual_stats: {edge_energy, flat_energy, chroma_shift, directional_bias}
  - freq_stats: {hf_loss, lf_loss, phase_distortion}

→ 推断误差: {missing_types: [...], severity_bias: ±N, order_error: bool}
```

## 预期产出

1. **误差-症状对照表**：每种识别误差类型对应的 failure signature
2. **修正决策树**：观察症状 → 推断误差 → 修正动作
3. **验证准确率**：在留出的 20% 已有数据上验证诊断规则的准确率
4. **更新 REFLECTION_MECHANISM.md**：将实验验证的规则补充进去

## 需要的资源

| 类型 | 数量 | 时间 |
|------|:--:|------|
| 新训练实验（矩阵 A） | 10 组 | ~2h (8 GPU) |
| 交叉测试（矩阵 B/C/D） | 22 组 | ~10min (纯推理) |
| 脚本开发 | — | ~1h |
| 数据分析 | — | ~2h |
| **合计** | | **~5h** |
