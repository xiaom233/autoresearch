# exp10: 退化感知的协作神经架构搜索

## 方法论来源：X-Restormer 的比较研究范式

X-Restormer (arXiv 2310.11881) 的核心方法论：**系统比较多种架构在同一退化上的表现，识别退化-架构耦合规律。**

关键发现：
1. **没有单一最优架构** — 不同恢复任务偏好不同的架构模式
2. **双分支设计有效** — 通道注意力 (MDTA) + 空间注意力 (OCAB) 互补
3. **重叠窗口空间注意力** — OCAB 通过 overlap 实现跨窗口通信，无需 shift
4. **U-Net skip connection** — 多尺度特征融合对恢复任务普遍有益

## 与传统 NAS 的区别

| 传统 NAS | 本方案 |
|----------|--------|
| 预定义搜索空间 | LLM 自由生成代码级变异 |
| 找一个通用最优架构 | 找退化-架构耦合规律 |
| 搜索算法驱动 | LLM 知识 + 实验结果驱动 |
| 输出：参数配置 | 输出：可执行 Python 代码 + 耦合规律 |

## Round 1：XRestormer 关键组件的退化感知消融

基于 XRestormer 的架构组件，在 3 组代表性退化上做系统消融：

| 退化 | 特征 | XRestormer 相关组件 |
|------|------|-------------------|
| D3 (comp+blur) | 强梯度冲突, 频域伪影+空间模糊 | OCAB空间注意力, 大窗口 |
| N4 (noise+blur) | 噪声主导, 局部处理 | MDTA通道注意力, InstNorm |
| S5 (motion三退化) | 方向性模糊, 长程依赖 | 双分支, 深层, 大感受野 |

### 消融实验矩阵

基于 RestoreNet (SwinIR)，逐步引入 XRestormer 组件：

| 变体 | 注意力 | 归一化 | FFN | 窗口 | 退化目标 |
|------|--------|--------|-----|------|----------|
| **基线** | Swin(W-MSA) | LayerNorm | MLP | 8 | D3/N4/S5 |
| **V1** | Swin → OCAB | LayerNorm | MLP | 12 | D3 (空间注意力) |
| **V2** | Swin → MDTA | LayerNorm | MLP | 4 | N4 (通道注意力) |
| **V3** | OCAB | InstNorm | GDFN | 12 | N4 |
| **V4** | Dual(Swin+GDFN) | LayerNorm | GDFN | 16 | S5 (双分支) |
| **V5** | OCAB | LayerNorm | GDFN | 16 | S5 |

每变体 × 3 退化 × Ft 策略 = 15 组实验 × ~43 min (EPOCH=1) ≈ 11 GPU hours

### 预期产出

1. **退化-注意力耦合**：空间注意力对 blur 更重要？通道注意力对 noise 更重要？
2. **退化-归一化耦合**：InstNorm 是否在噪声退化上优于 LayerNorm？
3. **退化-窗口大小耦合**：大窗口是否在 structured blur 上更重要？
