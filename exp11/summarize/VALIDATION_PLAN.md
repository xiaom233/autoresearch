# exp11 Phase E: 反射修正验证实验

## 目标

验证完整的"识别→训练→检测错误→修正→重训练"流程，证明诊断规则可以实际改善退化识别。

## 两类验证实验

### 实验 1：盲识别修正（类型误差）

**Step 1**：对已知 GT 退化图，用 `image-degradation-simulator` skill 进行盲识别
**Step 2**：在预测退化上训练 Swin (EPOCH_BUDGET=2)
**Step 3**：评估模型 → 应用诊断规则检测误识别
**Step 4**：根据诊断特征推断修正方向 → 更新 params.json
**Step 5**：在修正退化上重新训练 → 对比修正前后 PSNR

**测试案例**（选 4 个代表性 GT 退化）：
- GT1: L2 (noise+contrast) — 仿射型全局，skill 可能误判
- GT2: L6 (impulse+lens+saturate) — HSV 型全局，skill 常见遗漏 saturation
- GT3: L5 (motion+jpeg+contrast) — 复杂三退化
- GT4: bHSV_pure (纯 brightness_HSV) — 纯全局，skill 可能误判类型

### 实验 2：严重度修正（强度误差）

**目标**：验证在退化类型正确但严重度偏差 1-2 级时，能否通过模型表现检测并修正。

**设计**：
- GT = L2, contrast sev=3
- 故意用 sev=1, 2, 4, 5 训练（4 组）
- 测试每个模型的诊断特征
- 验证能否从残差图中推断正确的严重度

**关键假设**：
- sev 偏差 1 级（如 2 vs 3）：PSNR 下降较小，需要图像特征辅助
- sev 偏差 2 级（如 1 vs 3）：PSNR 下降明显，标量特征足够
- sev 高估（4→3）：过度锐化，edge_frac 异常高
- sev 低估（1→3）：残留噪声，entropy 高，edge_frac 低

### 实验 3：接近严重度修正（最难）

**目标**：验证在退化正确但严重度非常接近（偏差仅 1 级）时，能否检测。

**设计**：
- GT = L6, saturate sev=3
- 故意用 sev=2 和 sev=4 训练（2 组）
- 专门验证图像特征能否区分 ±1 严重度偏差

**预期**：这是最难的情况——PSNR 差异可能仅 1-2 dB，标量特征不可靠。必须依赖图像级特征（entropy, color_asymmetry）和残差图的视觉分析。

## 实验矩阵

| 实验 | GT 退化 | 预测退化 | 误差类型 | 验证什么 |
|:--:|------|------|:--:|------|
| V1 | L2 sev=3 | L2 sev=1 | severity -2 | 严重度低估检测 |
| V2 | L2 sev=3 | L2 sev=2 | severity -1 | 接近严重度检测 |
| V3 | L2 sev=3 | L2 sev=4 | severity +1 | 接近严重度检测 |
| V4 | L2 sev=3 | L2 sev=5 | severity +2 | 严重度高估检测 |
| V5 | L6 sev=3 | L6 sev=2 | severity -1 (saturate) | HSV 严重度检测 |
| V6 | L6 sev=3 | L6 sev=4 | severity +1 (saturate) | HSV 严重度检测 |
| V7 | L2 sev=3 | L1 (wrong type) | 类型错误 | 类型检测 + 修正 |
| V8 | L6 sev=3 | L2 (wrong type) | 类型错误 | 类型检测 + 修正 |

## 评估标准

对每个实验：
1. 记录预测模型在 GT 退化上的 PSNR
2. 应用诊断规则 → 输出修正建议
3. 修正后重训练 → 记录新 PSNR
4. 计算 Δ = PSNR_修正后 - PSNR_修正前
5. 统计修正成功率

**成功标准**：
- 修正后 PSNR > 修正前 PSNR + 0.5 dB → 修正有效
- 修正后 PSNR > 修正前 PSNR → 部分有效
- 修正后 PSNR ≤ 修正前 → 修正失败

## 时间估算

8 组实验 × 2 epochs × 86 min ≈ 23 GPU-hours ≈ 3h on 8 GPUs

## 与 Skill 的协同

Step 1: Skill 盲识别 → predicted_params.json
Step 2: 训练 M_spec(predicted_params)
Step 3: 交叉测试 → 数值诊断 + 图像特征
Step 4: 推断修正 → corrected_params.json  
Step 5: 训练 M_spec(corrected_params) → 验证改善
