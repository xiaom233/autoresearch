# exp9 实验总结

## 文件导航

| 文件 | 内容 | 适合 |
|------|------|------|
| `strategy_guide.md` | 策略选择指南、抽象原则、经验教训 | 日常参考 |
| `experiment_log.md` | 所有实验的完整记录和数据表 | 查阅细节 |
| `coupling_rules.md` | 退化→策略决策树速查 | 快速决策 |

## 核心结论

1. **Ft 永远是安全默认** — 30+ 退化组从未显著输
2. **motion P1 + sev≥3 是唯一 Curric > Ft 的可靠方向** — 4/4
3. **专攻小模型 > 通用大模型** — 12/15 DFPIR 对比
4. **Fixed split 不可被超越** — 四种动态方案均失败
5. **Loss/FtCurr/多专家均不值得投入** — <0.3 dB

详见 `strategy_guide.md`。
