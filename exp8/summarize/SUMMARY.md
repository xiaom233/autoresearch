# exp8 — DFPIR All-in-One 基线实验

## 目标

在本地环境部署并测试 DFPIR (CVPR'25) 31M 参数 all-in-one 盲复原模型，作为后续实验的大模型参考上界。

## 关键结论

1. **DFPIR 成功部署**：使用独立 conda 环境 (torch 2.5.1+cu124)，checkpoint `dfpir_blind_step301920.pt`
2. **多 GPU 并行**：8 GPU spawn 模式，tiled inference (tile=512, overlap=64)
3. **验证集覆盖**：737 张全分辨率图片，6 个标准 benchmark
4. **性能基准**：DFPIR (31M) 作为大模型参考上界，后续 exp9-12 中与小模型 (~0.45M) 对比

## 技术积累

- DFPIR 专用测试脚本：`test_degradation.py` — 支持多 GPU spawn + 单 GPU direct 模式
- 识别了 spawn 死锁问题：GPU 被占用时 spawn 永久挂起，需要 8 GPU 全部空闲
- ThreadPoolExecutor 预加载优化，单 GPU 模式避免 spawn 开销

## 后续影响

DFPIR 基线成为 exp9-13 的标准对比基线，用于衡量小模型专用化训练 vs 大模型泛化能力的差距。
