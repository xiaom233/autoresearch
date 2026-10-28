# E5c blind_0005 4-cell 失败归因

> 目的：直接回答 reviewer Q2 的一部分：blind_0005 的低 PSNR 是“DC 错误”还是“restorer 不足”？  
> 这是一个 **targeted 4-cell oracle 诊断**，不是完整 E2-RS（24 场景四格）。

## 1. 设计

| Cell | 训练 DC | 架构 | 说明 |
|---|---|---|---|
| A: no-fix | `brightness:1 + noise_impulse:3` | Swin+SG | 复用 AutoIR R1（错误 DC） |
| B: DC-only | `noise_impulse:5` | Swin+SG | 复用 E1 规则 R1（正确 DC） |
| C: RS-only | `brightness:1 + noise_impulse:3` | Swin+SG+DualBranch | 新训练：只改架构，保留错误 DC |
| D: both | `noise_impulse:5` | Swin+SG+DualBranch | 新训练：正确 DC + DualBranch |

- 训练：Direct、2 epoch、LR5e-4、batch16、seed42、同 1.5 GPU-h 预算；C/D 在 GPU0 顺序训练。
- 评估：blind_0005 固定 LQ（`E1E2_fixed_lq/blind_0005`），DIV2K 100 张；PSNR/SSIM。
- A/B 复用已有模型，不是同一代码/seed 的新重跑；C/D 是新增训练。DualBranch 是“只改架构”的代表性选择，不是完整架构搜索。

## 2. 结果（DIV2K 100，fixed-LQ）

| Cell | DC | 架构 | PSNR | SSIM |
|---|---|---|---:|---:|
| A no-fix | `brightness:1 + impulse:3` | Swin+SG | 21.915 | — |
| B DC-only | `impulse:5` | Swin+SG | **44.908** | — |
| C RS-only | `brightness:1 + impulse:3` | DualBranch+SG | 21.976 | 0.8274 |
| D both | `impulse:5` | DualBranch+SG | 41.620 | 0.9807 |

参考：A 为 AutoIR R1 的 fixed-LQ 21.915；B 为规则 R1 的 fixed-LQ 44.908。

## 3. 主效应

| 效应 | 计算 | ΔPSNR |
|---|---|---:|
| **DC effect @ Swin+SG** | B − A = 44.908 − 21.915 | **+22.992** |
| DC effect @ DualBranch | D − C = 41.620 − 21.976 | **+19.644** |
| **RS effect @ wrong DC** | C − A = 21.976 − 21.915 | **+0.061** |
| RS effect @ correct DC | D − B = 41.620 − 44.908 | **−3.288** |
| Interaction | (D−C) − (B−A) | **−3.348** |

## 4. 解释

1. **DC 是主导因素**：只把 `impulse:3` 改成 `impulse:5` 并去掉 brightness，PSNR +23.0 dB；这是 blind_0005 低分的根因。
2. **只改架构几乎无效**：DualBranch 在错误 DC 上只 +0.06 dB，不能把 brightness/severity 错误救回来。
3. **架构改错还会伤害正确 DC**：正确 DC 下 DualBranch 比 Swin+SG 低 3.29 dB；说明 RS 不是“加模块就更好”，需要信号门控。
4. **restorer 容量解释不成立**：同一 0.46M 模型、同预算，正确 DC 能到 44.9 dB；21.9 dB 不是模型容量上限。
5. 与规则控制器对照一致：规则控制器只改 DC 得到 44.92/41.14（24 场景 test）；E5c 在固定 LQ 0005 上复现了同一结论。

## 5. 对 reviewer Q2 的直接回答

> blind_0005 的失败不是 restorer 不足，而是 DC 错误。在固定 LQ 0005 上，只改 DC（`impulse:3+brightness:1 → impulse:5`）带来 **+23.0 dB**；只改架构（加 DualBranch）在错误 DC 上只有 **+0.06 dB**，在正确 DC 上反而 **−3.29 dB**。因此 CLR 应优先修正 DC（参数级 severity/假阳性检查），而不是把低 PSNR 归因于模型容量或盲目加架构。

## 6. Caveat

- A/B 复用已有模型，C/D 新训练；四格不是同一代码/seed 的严格 4 次重跑。A/B 的 Swin+SG 与 C/D 的 DualBranch 差异可比较，但训练 run 差异仍在。
- DualBranch 只是“RS-only”的一个代表；其他架构（OCAM/GCM 等）未测。
- 只做 blind_0005 一个场景、DIV2K 100 张；不是完整 E2-RS 24 场景四格。
- `blind_0005` 的非确定性 LQ 使用 `E1E2_fixed_lq` 的同一份固定 LQ；A/B 的 fixed-LQ 值来自 `E1E2_fixed_lq_summary.json`。
- 不能据此声称“所有 RS 都无效”；只能声称“在本案例、DualBranch 这个 RS 选择下，DC 是主导，RS 单独无效/有害”。

## 7. 产物

```text
rebuttal_runs/E5c/selection.jsonl
rebuttal_runs/E5c/test_metrics.jsonl
rebuttal_runs/E5c/blind_0005_RSonly/checkpoints/
rebuttal_runs/E5c/blind_0005_both/checkpoints/
rebuttal_runs/E5c/logs/blind_0005_RSonly.log
rebuttal_runs/E5c/logs/blind_0005_both.log
```
