# exp39: 0参数新组件探索 — 退化特异性 vs 跨退化泛化

> 来源: NAFNet (ECCV 2022) SimpleGate + SCA
> 对比基线: Direct + Swin (454K)
> 核心问题: 0参数组件能否在特定退化上提供收益，且跨退化无害？

---

## 一、实验目的

### 1.1 核心假设

| 组件 | 假设 | 机理解释 |
|------|------|------|
| **SimpleGate** | 多步退化 (≥2步) 受益 | 通道对半分→相乘 = 隐式门控，减少梯度干扰。单步退化无干扰故无效 |
| **SCA** | 全局退化 (contrast/stretch) 受益 | GAP→L2 Norm 通道注意力增强全局统计量感知。contrast = 通道级仿射变换 |
| **SG+SCA** | 叠加收益 | SimpleGate 减少干扰 + SCA 增强通道交互，可能1+1>2 |

### 1.2 验证维度

1. **退化特异性**: 组件在匹配退化上 vs Swin 的 Δ PSNR
2. **跨退化泛化安全性**: 组件在不匹配退化上的 Worst Δ — 是否有害
3. **与 exp38 组件对比**: SimpleGate/SCA (0参数) vs ColorPre (+0.8K) / FiLM (+26K) 的性价比

---

## 二、测试组件

| # | 组件 | 来源 | Δ 参数 | 机制 | 环境变量 |
|:--:|------|------|:--:|------|------|
| 1 | **SimpleGate** | NAFNet (ECCV 2022) | **-32K** | 通道对半分 → 逐元素相乘，替代 GELU | `AR_USE_SIMPLE_GATE=1` |
| 2 | **SCA** | NAFNet (ECCV 2022) | **+512** | GAP → L2 Norm → 可学习通道缩放 | `AR_USE_SCA=1` |
| 3 | **SG+SCA** | 组合 | **-32K** | SimpleGate + SCA 联合 | `AR_USE_SIMPLE_GATE=1 AR_USE_SCA=1` |

### 2.1 实现细节

**SimpleGate** (`model.py:425`):
```python
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)  # 通道对半分
        return x1 * x2                # 逐元素门控
```
- 替换 SwinBlock MLP 中的 GELU
- MLP 结构: Linear(dim→hidden) → SimpleGate → Linear(hidden//2→dim)
- 第二层 Linear 输入减半，反而减少 32K 参数

**SCA** (`model.py:437`):
```python
class SCA(nn.Module):
    def __init__(self, dim):
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        gap = x.mean(dim=1, keepdim=True)       # 空间池化
        gap = gap * gap.norm(p=2).reciprocal()   # L2 归一化
        return x * self.scale * gap               # 通道缩放
```
- 插入 SwinBlock MLP 之后
- 无激活函数、无 Sigmoid、无可学习 bias

### 2.2 参数统计

| 架构 | 参数量 | vs Swin |
|------|:--:|:--:|
| Swin (baseline) | 454,531 | — |
| +SimpleGate | 421,763 | **-32,768** |
| +SCA | 455,043 | +512 |
| +SG+SCA | 422,275 | -32,256 |

---

## 三、退化矩阵

### 3.1 退化管线

| ID | 路径 | 步数 | 管线内容 |
|:--:|------|:--:|------|
| **C** | `C_pure.json` | 1 | `contrast_weaken_scale(3)` |
| **N** | `N_pure.json` | 1 | `noise_gaussian_RGB(3)` |
| **B** | `B_pure.json` | 1 | `blur_gaussian(3)` |
| **L2** | `L2_dual.json` | 2 | `noise_gaussian_RGB(3)` + `contrast_weaken_scale(3)` |
| **D3** | `D3_dual.json` | 2 | `compression_jpeg(3)` + `blur_lens(4)` |
| **G2** | `G2_dual.json` | 2 | `noise_gaussian_RGB(3)` + `contrast_weaken_stretch(3)` |
| **BS** | `BS_dual.json` | 2 | `noise_gaussian_RGB(3)` + `brightness_darken_shfit_RGB(3)` |
| **S5** | `S5_triple.json` | 3 | `blur_motion(5)` + `noise_gaussian_RGB(1)` + `compression_jpeg(1)` |
| **SF5** | `SF5_triple.json` | 3 | `blur_gaussian(3)` + `noise_gaussian_RGB(3)` + `brightness_darken_gamma_RGB(3)` |

### 3.2 退化选择逻辑

```
单步 (3): 验证组件是否对简单退化也有效
  C: contrast → SCA 最可能受益 (通道级变换)
  N: noise    → 预期 ≈0 (无通道/梯度结构)
  B: blur     → 预期 ≈0 (空间退化)

双步 (4): 验证组件在混合退化上的表现
  L2: contrast+noise → 全局+随机, SCA 预期有效
  D3: JPEG+blur → 梯度干扰, SimpleGate 预期有效
  G2: stretch+noise → 非contrast全局, SCA 边界测试
  BS: brightness+noise → 简单全局, 预期 ≈0

三步 (3): 验证组件在复杂场景是否退化
  S5: motion+noise+JPEG → 三退化, SimpleGate 预期有效
  SF5: gamma+blur+noise → gamma三退化, 预期 ≈0
```

---

## 四、实验矩阵

### 4.1 统一训练配置

```
EPOCH_BUDGET=2        # 公平对比
LEARNING_RATE=0.0005   # 5e-4
BATCH_SIZE=16
EMBED_DIM=64
WINDOW_SIZE=8
ATTENTION_TYPE=swin
LOSS_FN=l1
LR_SCHEDULE=constant
AMP_DTYPE=bfloat16
所有实验 Direct 训练 (无 CURRICULUM_CONFIG, 无 LOAD_CKPT)
```

### 4.2 36 组完整矩阵

#### Phase D: Swin 基线 (9 组) — `AR_USE_SIMPLE_GATE=0 AR_USE_SCA=0`

| ID | 退化 | 环境变量 | exp38 参考 PSNR |
|------|:--:|------|:--:|
| SW_C | C | `AR_PARAMS_PATH=exp39/degradation/C_pure.json AR_VAL_PARAMS_PATH=exp39/degradation/C_pure.json` | — |
| SW_N | N | `... N_pure.json` | — |
| SW_B | B | `... B_pure.json` | — |
| SW_L2 | L2 | `... L2_dual.json` | 28.19 |
| SW_D3 | D3 | `... D3_dual.json` | 21.38 |
| SW_G2 | G2 | `... G2_dual.json` | 27.79 |
| SW_BS | BS | `... BS_dual.json` | 28.62 |
| SW_S5 | S5 | `... S5_triple.json` | 22.25 |
| SW_SF5 | SF5 | `... SF5_triple.json` | — |

#### Phase A: SimpleGate (9 组) — `AR_USE_SIMPLE_GATE=1 AR_USE_SCA=0`

| ID | 退化 | 预期 Δ vs Swin | 假设 |
|------|:--:|:--:|------|
| SG_C | C | ≈0 | 单步无梯度干扰 |
| SG_N | N | ≈0 | 单步无干扰 |
| SG_B | B | ≈0 | 单步无干扰 |
| SG_L2 | L2 | ≈0 | global+noise=弱干扰 |
| SG_D3 | D3 | **+0.3~0.5** | JPEG+blur=强梯度干扰 |
| SG_G2 | G2 | ≈0 | stretch+noise=弱干扰 |
| SG_BS | BS | ≈0 | shift=太简单 |
| SG_S5 | S5 | **+0.3~0.5** | 三退化=最强干扰 |
| SG_SF5 | SF5 | ≈0 | gamma=MLP已能处理 |

#### Phase B: SCA (9 组) — `AR_USE_SIMPLE_GATE=0 AR_USE_SCA=1`

| ID | 退化 | 预期 Δ vs Swin | 假设 |
|------|:--:|:--:|------|
| SCA_C | C | **+0.3~0.5** | 纯contrast=通道变换 |
| SCA_N | N | ≈0 | noise=无通道结构 |
| SCA_B | B | ≈0 | blur=空间退化 |
| SCA_L2 | L2 | **+0.3~0.5** | contrast=通道级 |
| SCA_D3 | D3 | ≈0 | JPEG+blur=空间退化 |
| SCA_G2 | G2 | **+0.2~0.3** | stretch=通道级但弱于contrast |
| SCA_BS | BS | ≈0 | shift=太简单 |
| SCA_S5 | S5 | ≈0 | 主空间退化 |
| SCA_SF5 | SF5 | ≈0 | gamma=MLP已能处理 |

#### Phase C: SG+SCA (9 组) — `AR_USE_SIMPLE_GATE=1 AR_USE_SCA=1`

| ID | 退化 | 预期 Δ vs Swin | 假设 |
|------|:--:|:--:|------|
| SSC_C | C | SCA 主导 | 单步contrast |
| SSC_N | N | ≈0 | 无收益 |
| SSC_B | B | ≈0 | 无收益 |
| SSC_L2 | L2 | SCA 主导 | contrast |
| SSC_D3 | D3 | SG 主导 | 梯度干扰 |
| SSC_G2 | G2 | SCA+SG? | 可能叠加 |
| SSC_BS | BS | ≈0 | 无收益 |
| SSC_S5 | S5 | SG 主导 | 三退化 |
| SSC_SF5 | SF5 | ≈0 | 无收益 |

---

## 五、训练脚本入口

### 5.1 任务生成

```bash
python3 exp39/scripts/generate.py
# 输出: exp39/scripts/tasks.txt (gpu_runner.sh 格式, 36 行)
```

### 5.2 启动命令

```bash
cd /data/zyli/projects/autoresearch
bash scripts/exp_launcher.sh start exp39/scripts/tasks.txt 1,2,3,4,5,6,7
# 跳过 GPU0 (温度问题)
```

### 5.3 任务格式示例

```
CUDA_VISIBLE_DEVICES=GPU_ID AR_EMBED_DIM=64 AR_EPOCH_BUDGET=2 ... AR_USE_SIMPLE_GATE=1 AR_PARAMS_PATH=exp39/degradation/D3_dual.json AR_VAL_PARAMS_PATH=exp39/degradation/D3_dual.json AR_CKPT_PREFIX=exp39/experiments/SG_D3 .venv/bin/python3 train.py > exp39/logs/SG_D3.log 2>&1|SG_D3
```

### 5.4 监控

```bash
watch -n 30 'echo "剩余: $(wc -l < exp39/scripts/tasks.txt)"; grep "^val_psnr_db:\|Ckpt.*PSNR_RGB" exp39/logs/*.log 2>/dev/null | tail -20'
```

---

## 六、预期结果

### 6.1 退化特异性矩阵 (预期)

| 退化 | 步数 | SimpleGate | SCA | SG+SCA | 最佳 |
|------|:--:|:--:|:--:|:--:|:--:|
| C (contrast) | 1 | ≈0 | **+0.3~0.5** | +0.3~0.5 | SCA |
| N (noise) | 1 | ≈0 | ≈0 | ≈0 | Swin |
| B (blur) | 1 | ≈0 | ≈0 | ≈0 | Swin |
| L2 (contrast+noise) | 2 | ≈0 | **+0.3~0.5** | +0.3~0.5 | SCA |
| D3 (JPEG+blur) | 2 | **+0.3~0.5** | ≈0 | +0.3~0.5 | SG |
| G2 (stretch+noise) | 2 | ≈0 | **+0.2~0.3** | +0.2~0.3 | SCA |
| BS (brightness+noise) | 2 | ≈0 | ≈0 | ≈0 | Swin |
| S5 (motion+noise+JPEG) | 3 | **+0.3~0.5** | ≈0 | +0.3~0.5 | SG |
| SF5 (gamma+blur+noise) | 3 | ≈0 | ≈0 | ≈0 | Swin |

### 6.2 跨退化泛化安全性 (预期)

| 组件 | Best Δ | Worst Δ | Win Rate | 安全? |
|------|:--:|:--:|:--:|:--:|
| SimpleGate | +0.3~0.5 | ≈0 | 2/9 (22%) | ✅ 无害 |
| SCA | +0.3~0.5 | ≈0 | 3/9 (33%) | ✅ 无害 |
| SG+SCA | 叠加 | ≈0 | 3/9 (33%) | ✅ 无害 |

### 6.3 与 exp38 组件对比 (预期)

| 组件 | 参数量 | L2 Δ | D3 Δ | 有害退化 | 盲识别可用? |
|------|:--:|:--:|:--:|------|:--:|
| ColorPre | +0.8K | +0.62 | — | saturation | ❌ |
| FiLM-GCM | +26K | +0.47 | — | — (安全) | ❌ (盲识别不确定) |
| **SimpleGate** | **-32K** | ≈0 | **+0.3~0.5** | **无** | **✅ 可常开** |
| **SCA** | **+512** | **+0.3~0.5** | ≈0 | **无** | **可常开** |

> 关键差异化: SimpleGate 解决梯度干扰 (D3/S5)，SCA 解决通道退化 (L2/G2/C)，互补且 0 参数。

---

## 七、判定标准

- **有效**: Δ > 0.3 dB 且 ≥2 退化
- **安全**: Worst Δ ≥ -0.2 (无显著有害)
- **常开推荐**: 有效 + 安全 → 可设为训练默认值
- **信号门控**: 有效 + 仅在特定退化 → 需盲识别信号触发
- **不推荐**: Δ ≈ 0 且无特殊价值

---

## 八、执行步骤

```bash
# 1. 确认环境
.venv/bin/python3 -c "from train import RestoreNet; m=RestoreNet(use_simple_gate=True,use_sca=True); print('OK')"

# 2. 生成任务
python3 exp39/scripts/generate.py

# 3. 清理 + 启动 (跳过 GPU0)
bash scripts/exp_launcher.sh start exp39/scripts/tasks.txt 1,2,3,4,5,6,7

# 4. 监控
watch -n 30 'wc -l < exp39/scripts/tasks.txt; grep "Ckpt.*PSNR_RGB" exp39/logs/*.log 2>/dev/null | wc -l'

# 5. 汇总
python3 exp39/scripts/collect.py
```

**预计墙钟**: 36 组 × 5 min / 7 GPU ≈ 25-30 min
