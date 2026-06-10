# exp13 — 盲识别全流程端到端验证 (24 退化)

## 目标

严格按 program.md 6 阶段流程 + 四层隔离协议，验证同图模式盲识别的端到端效果。

## 实验设计

24 组退化 (4 单 + 10 双 + 10 三)，同图模式 (`--same-image`)，Agent 全程不接触 GT。

### 隔离协议
- GT 隐藏：`.ground_truth/` 目录 Agent 不可访问
- 训练参数隐藏：`degradation/` 盲识别前不可读
- VAL 锁定：launcher 强制注入 GT 路径
- Pipeline 抑制：`AR_QUIET_PIPELINE=1`

## 结果

### 盲识别
- **同图模式 CI pass 100% GOOD (24/24)**
- 跨图模式仅 12% GOOD (3/24)

### Phase 5: Spec vs Ft

| Spec vs Ft | 数量 |
|------------|:--:|
| Spec > Ft (>0.3dB) | 2 |
| Ft > Spec (>0.3dB) | 1 |
| 持平 | **21** |

```
Spec 均值: 21.69
Ft 均值:   21.33
```

### 关键发现

1. **盲识别质量决定 Spec 与 Ft 的差距**：同图模式下 predicted ≈ GT，Spec ≈ Ft
2. **与 exp12 的根本差异**：exp12 无盲识别（直接用 GT），Ft 碾压 Spec；exp13 有盲识别，两者等价
3. **3003 异常**：Spec=23.38 >> Ft=15.70 (+7.68 dB)，唯一 Ft 严重崩溃的案例

## 结论

- 同图模式 + 四层隔离 = 可靠的盲识别协议
- 盲识别质量高时，Spec 和 Ft 两种训练策略差异消失
- 反思修正应聚焦于退化参数修正，而非训练策略微调
