# IRAgent — Autonomous Image Restoration Agent

给定退化图像，自动识别退化类型与严重程度，在线合成退化数据集，自动优化网络结构并训练专用复原模型。

详见 [program.md](program.md)。

## 快速开始

```bash
uv sync                          # install dependencies
uv run prepare.py                # crop DIV2K + package WebDataset shards
uv run train.py                  # train restoration model
```

## 核心组件

| 文件/目录 | 说明 |
|-----------|------|
| `program.md` | Agent 行为规范 + 完整 pipeline 文档 |
| `train.py` | 复原模型 + 训练循环 (Agent 可编辑) |
| `prepare.py` | 数据准备 + WebDataset + 退化 DataLoader (只读) |
| `x_distortion/` | 35 种退化函数 × 5 级严重度 |
| `resource/.../net/` | DFPIR 大模型 (31M, CVPR'25) |
| `.claude/skills/image-degradation-simulator/` | 退化识别 skill |
| `blind_challenge.py` | 盲识别挑战生成 |
| `setup_challenge.sh` | 防泄露挑战初始化 |
| `run_experiments.py` | Phase 5 实验矩阵 runner |
