# exp38 实验结果总结

> 状态: 25/32 已验证 + 7 组重跑中（MDTA/OCAB）
> 评估: Set14 checkpoint PSNR_RGB, 同环境公平对比

---

## 一、策略对比

### RandomCurric vs Direct

| 管线 | 退化类型 | Direct | RandomCurric | Δ |
|------|------|:--:|:--:|:--:|
| D1 | blur+noise (纯局部) | 23.37 | 23.32 | **-0.05** |
| D3 | JPEG+blur (梯度干扰) | 21.38 | 25.71 | **+4.33** |
| L2 | contrast+noise (全局) | 28.19 | 28.04 | **-0.15** |
| S5 | motion+noise+JPEG | 22.25 | 21.78 | -0.47 |

**结论**:
- 纯局部退化: RandomCurric ≈ Direct (安全)
- 梯度干扰 (JPEG+blur): RandomCurric 大幅胜出 (+4.33)
- 含 contrast 全局退化: RandomCurric 轻微劣化 (-0.15), 未复现 exp9 的 -4.51
- 三退化: RandomCurric 略差 (-0.47)

### True Ft vs RandomCurric vs Direct（D3 管线）

| 策略 | PSNR | vs Direct | 机制 |
|------|:--:|:--:|------|
| Direct | 21.45 | — | 从头训练 |
| True Ft (AR_LOAD_CKPT) | 21.48 | **+0.03** | 预训练特征迁移 |
| RandomCurric | 25.80 | **+4.35** | 梯度干扰减少 |

**结论**: 2-epoch 预算下 True Ft ≈ Direct。RandomCurric 收益来自**梯度干扰减少**，非预训练迁移。两者机制完全不同，不可混用术语。

### Curric 方向（S5 motion sev=5, Swin）

| 策略 | PSNR | vs Direct |
|------|:--:|:--:|
| Direct | 22.25 | — |
| RandomCurric | 21.78 | -0.47 |
| CurricFwd (motion first) | 22.63 | **+0.38** |
| CurricRev (JPEG first) | 22.20 | -0.05 |

**结论**: Swin 下 Fwd > Rev (+0.43 dB)。Motion sev=5 时 Fwd 仍胜出，"Fwd 翻转" 不适用于 Swin。CurricFwd > Direct (+0.38)，课程学习对 motion 有益。

---

## 二、架构组件

### Contrast (L2) 组件

| 架构 | PSNR | Δ vs Swin |
|------|:--:|:--:|
| Swin | 28.19 | — |
| +FiLM-GCM | 28.66 | **+0.47** |
| +ColorPre | 28.81 | **+0.62** |

> 趋势与 exp10 一致（FiLM +0.79, ColorPre +0.81），幅度较小。两者均有效。

### FiLM-GCM 跨全局退化泛化

| 退化 | 类型 | Swin | FiLM-GCM | ColorPre | FiLM Δ | 判定 |
|------|------|:--:|:--:|:--:|:--:|------|
| L2 | contrast_scale | 28.19 | 28.66 | 28.81 | **+0.47** | ✅ 有效 |
| BS | brightness_shift | 28.62 | 28.60 | — | -0.02 | ≈0 安全 |
| G1 | gamma_RGB | 28.58 | 28.63 | 28.57 | +0.05 | ≈0 安全 |
| G2 | stretch | 27.79 | 28.14 | 27.89 | **+0.35** | 🟡 微益 |
| SF7 | saturation | — | 28.78 | — | — | ✅ 安全 |
| SF5 | gamma triple | — | 23.26 | — | +0.30 | 🟡 微益 |

**结论**: FiLM-GCM 在所有非 contrast 全局退化上**安全**（无 NaN/无崩溃）。contrast & stretch 有效，brightness/gamma/saturation 无害（≈0）。**不是 "contrast 专用"**。

---

## 三、关键修正

### 已确认需修正的 finetune_strategy.md 条目

| 条目 | 旧结论 | 新结论 | 来源 |
|------|------|------|------|
| MDTA/OCAB vs contrast | "避免 MDTA/OCAB" | **待 MDTA/OCAB 重跑完成** | Phase A |
| Motion sev≥5 Fwd 翻转 | "Fwd 翻转" | **Swin 下 Fwd 仍胜出 (+0.43)** | Phase B |
| RandomCurric 安全边界 | "contrast 崩溃 -4.51" | **仅 -0.15，未复现崩溃** | Phase A |
| True Ft vs RandomCurric | exp9 "Ft" = Fine-tune | **True Ft ≈ Direct, RandomCurric 是梯度干扰减少** | Phase D |
| FiLM-GCM 适用范围 | "contrast 专用" | **contrast+stretch 有效，其他全局安全** | Phase F |
| T2 实验 | 引用为策略依据 | **含已禁用函数，不可引用** | 审计发现 |
| exp9/10 "Ft" 术语 | Fine-tune | **RandomCurric** | 审计确认 |

### 未复现的 exp9 现象

| 现象 | exp9 | exp38 | 可能原因 |
|------|:--:|:--:|------|
| L2 RandomCurric 崩溃 | -4.51 dB | -0.15 dB | 数据集/LR/代码版本差异 |
| L2 MDTA crash | -9.78 dB | ⏳ 待确认 | exp9 含 RandomCurric 耦合 |
| L2 OCAB crash | -10.38 dB | ⏳ 待确认 | exp9 含 RandomCurric 耦合 |

---

## 四、发现的代码 Bug

| Bug | 影响 | 状态 |
|------|------|:--:|
| `AR_ATTENTION_TYPE` 不在 for 循环 | 所有 MDTA/OCAB 实验实际用 Swin | ✅ 已修复 (train.py) |
| OCAB rearrange 缺 `c=` 参数 | OCAB 启动崩溃 | ✅ 已修复 (model.py) |
| `VAL_DIRS=[]` | 最终验证 0 图像 | ⚠️ 不影响 (用 Set14 ckpt PSNR) |

---

## 五、MDTA/OCAB 公平参数对比（EMBED_DIM=30, ~470K）

| ID | 架构 | 策略 | L2 PSNR | vs Swin |
|------|------|------|:--:|:--:|
| A1 | Swin (dim=64, 454K) | Direct | 28.19 | — |
| A2 | **MDTA** (dim=30, 473K) | Direct | 28.53 | **+0.34** |
| A3 | OCAB (dim=30, 467K) | Direct | 28.02 | -0.17 |

| ID | 架构 | 策略 | S5 PSNR | vs Swin |
|------|------|------|:--:|:--:|
| B1 | Swin (dim=64, 454K) | Direct | 22.25 | — |
| B3 | Swin (dim=64) | CurricFwd | 22.63 | +0.38 |
| B5 | OCAB (dim=30) | Direct | 22.23 | -0.02 |
| B6 | OCAB (dim=30) | CurricFwd | 22.23 | -0.02 |
| B7 | OCAB (dim=30) | CurricRev | 22.36 | +0.11 |

**结论**:
- **MDTA 在 L2 上优于 Swin (+0.34 dB)**，通道注意力在 contrast 上有参数效率优势
- **OCAB 在 L2 和 S5 上均不优于 Swin**，空间注意力无优势
- exp9 中 OCAB 的 +1.72 dB 来自参数量膨胀 (2.1M, 4.7×)，公平对比下消失
- **Swin CurricFwd 仍是 S5 motion 最优**

## 六、待完成

| 项目 | 状态 |
|------|:--:|
| GT 重评估 (多数据集) | ⏳ 需补充 B100/Urban100/Manga109 |
