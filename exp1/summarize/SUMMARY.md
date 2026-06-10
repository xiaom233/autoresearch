# exp1 — 退化管线下的架构与训练策略探索

## 目标

在固定退化管线 `blur_motion(3) → noise_spatially_correlated(2) → compression_jpeg(4)` 下最大化验证 PSNR。首次系统探索模型架构和训练超参。

## 实验规模

Phase 1-4: 32 组 | Phase 5: 72 组 | 总计 104 组

## 核心结论

### Phase 1-4 发现 (32 组，每实验 10 分钟)

1. **恒定 LR 优于余弦调度** (+0.44 dB) — 短训练下模型未收敛
2. **更宽优于更深** — EMBED_DIM=96 > 增加层数
3. **改进效果不叠加** — 组合优化不如单点优化
4. **梯度裁剪** (max_norm=1.0) 提供 +0.33 dB (vs 初版基线)
5. **Sobel 边缘 loss 与 bfloat16 AMP 不兼容**

### Phase 5 发现 (72 组，每实验 20 分钟)

1. **LR 调度结论反转** — 更长训练下 cosine > constant LR (+0.15 dB)
2. **window_size=16 是最大免费增益** (+0.50 dB，零参数成本)
3. **梯度裁剪增益缩小** — +0.21 dB (vs Phase 5 基线)

### 最佳结果

| 阶段 | 配置 | PSNR |
|------|------|:--:|
| Phase 1-4 | const LR + embed=96 | 20.71 |
| Phase 5 | Y3: w16+embed96+heads8+clip+cosine | **21.59** |

盲基线 M_blind: 18.88 dB。最佳 Specialist: +2.71 dB。

## 教训

- 训练预算影响超参排名：短训练 (10min) 和长训练 (20min) 下最优 LR 调度不同
- 改进效果不线性叠加——需要系统搜索而非贪心组合
