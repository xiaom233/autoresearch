# exp37 训练策略与资源报告

## 训练配置

| 参数 | 值 |
|------|------|
| EPOCH_BUDGET | 2 |
| LR | 5e-4 |
| Loss | L1 |
| Batch size | 16 |
| AMP | bfloat16 |
| GPU 时间/挑战 | ~1.5h |

## 逐挑战策略与架构

| Challenge | 策略 | 架构 | R0? | R1? |
|-----------|------|------|:--:|:--:|
| blind_0001 | Direct | Swin+DualBranch | ✅ | ✅ |
| blind_0002 | Direct | Swin+DualBranch | ✅ | — |
| blind_0003 | Ft | Swin | ✅ | — |
| blind_0004 | Direct | Swin+DualBranch | ✅ | — |
| blind_0005 | Direct | Swin+DualBranch | ✅ | ✅ |
| blind_0006 | Ft | Swin | ✅ | — |
| blind_0007 | Direct | Swin+DualBranch | ✅ | — |
| blind_0008 | Ft | Swin | ✅ | — |
| blind_0009 | Direct | Swin+DualBranch | ✅ | ✅ |
| blind_0010 | Ft | Swin | ✅ | — |
| blind_0011 | Ft | Swin | ✅ | — |
| blind_0012 | Direct | Swin+DualBranch | ✅ | ✅ |
| blind_0013 | Direct | Swin+DualBranch | ✅ | — |
| blind_0014 | Direct | Swin+DualBranch | ✅ | ✅ |
| blind_0015 | Direct | Swin+DualBranch | ✅ | — |
| blind_0016 | Ft | Swin | ✅ | — |
| blind_0017 | Direct | Swin+DualBranch | ✅ | — |
| blind_0018 | Direct | Swin+FiLM-GCM | ✅ | ✅ |
| blind_0019 | Direct | Swin+DualBranch | ✅ | ✅ |
| blind_0020 | Direct | Swin+FiLM-GCM | ✅ | — |
| blind_0021 | Direct | Swin+DualBranch | ✅ | ✅ |
| blind_0022 | Direct | Swin+DualBranch | ✅ | — |
| blind_0023 | Direct | Swin+FiLM-GCM | ✅ | — |
| blind_0024 | Direct | Swin+DualBranch | ✅ | — |

## GPU 资源统计

- R0 训练: 8 GPU × 3 任务 × 1.5h = 36 GPU-h (~4.5h 墙钟)
- R1 训练: 8 GPU × 1 任务 × 1.5h = 12 GPU-h (~1.5h 墙钟)
- DFPIR-ft: 8 GPU × 3 任务 × 1.5h = 36 GPU-h (~4.5h 墙钟)
- 总 GPU 时间: ~84 GPU-h

