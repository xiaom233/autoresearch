# 当前实验计划：退化类型感知的训练策略

## 目标

找到能够根据退化属性自适应调整训练策略的变量，使得 Curric 可以在更多退化上超越 Ft。

## 实验 1：自适应 Phase 切换 (ADP)

### 机制
用 EMA loss 斜率检测 Phase1 是否收敛。收敛→切 Phase2，未收敛→继续学。
退化修复难度天然决定了 Phase1 需要多长时间。

### 参数扫描（18 组）
- 退化：D3 (comp+blur, 中等难度)
- ADAPTIVE_SWITCH=500
- Threshold ∈ {0.001, 0.005, 0.01}（相对改善率）
- Patience ∈ {2, 3, 5}（连续平坦次数）
- 2 方向 (fwd/rev)

### 验证（8 组）
最优参数在以下退化上验证：
- S5 (motion P1, Ft不好, Curric>Ft) — ADP 应进一步改善
- M1 (motion P1 sev=3) — ADP 可能帮助
- M2 (lens P1, 困难) — ADP 不应帮助
- T2 (策略饱和) — ADP 不应帮助

### 代码
`train.py`: ADAPTIVE_SWITCH, ADAPTIVE_THRESHOLD, ADAPTIVE_PATIENCE, ADAPTIVE_WARMUP

## 实验 2：Phase1 类型感知学习率

### 机制
结构化退化 (blur, compression) 需要更大探索→高 LR (2e-3)
随机退化 (noise) 容易过拟合→低 LR (5e-4)

### 实验（12 组）
| 退化 | Phase1 类型 | 优化 LR | 方向 |
|------|:--:|:--:|:--:|
| S5 motion(5)→noise(1)→jpeg(1) | structured | 2e-3 | fwd/rev |
| M1 motion(3)→noise(3)→jpeg(3) | structured | 2e-3 | fwd/rev |
| M2 lens(3)→noise(3)→jpeg(3) | structured | 2e-3 | fwd/rev |
| N4 noise(3)→blur(3) | random | 5e-4 | fwd/rev |
| S6 motion(1)→noise(5)→jpeg(1) | random | 5e-4 | fwd/rev |
| T2 noise_corr(4)→motion(3)→jpeg2000(3) | random | 5e-4 | fwd/rev |

基线：已有默认 LR=1e-3 的 Curric(cf/cr) 结果

## 实验 3：Phase1 类型感知 Loss

### 机制
L1=拉普拉斯先验(稀疏), MSE=高斯先验(平滑)
Curric Phase1 只修复单一退化→Loss 应匹配退化类型

### 实验（9 组）
| 退化 | Phase1 退化 | 优化 Loss | 原因 |
|------|:--:|:--:|------|
| S5 | motion blur | l1+edge | 边缘重建 |
| M1 | motion blur | l1+edge | 边缘重建 |
| M2 | lens blur | l1+edge | 边缘重建 |
| D3 | compression | l1+fft | 频域伪影 |
| N6 | compression | l1+fft | 频域伪影 |
| N4 | gaussian noise | MSE | 高斯先验最优 |
| S6 | gaussian noise | MSE | 高斯先验最优 |
| T2 | correlated noise | MSE | 高斯先验最优 |
| D2 | impulse noise | huber | 抗离群值 |

基线：已有 LOSS_FN=l1 的 Curric(cf/cr) 结果

## 实验时间

| 批次 | 组数 | 状态 |
|------|:--:|------|
| ADP 参数扫描 (D3) | 18 | 8/18 done, 7 GPU 在跑 |
| ADP 验证 | 8 | 待最优参数确定后启动 |
| 类型感知 LR | 12 | 已排队 |
| 类型感知 Loss | 9 | 已排队 |
| **合计** | **47** | ~67 GPU hours |

## 预期结论

1. ADP 是否优于固定 split？
2. 结构化退化是否受益于高 LR？
3. Phase1 Loss 是否应匹配退化类型？
4. 以上三个策略是否可以叠加使用？
