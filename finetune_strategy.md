# 退化类型如何影响训练策略

> 基于 exp9 (~500 组实验)。核心转变：从"什么退化用什么策略" → "什么机制决定什么策略"。
> 详见 `exp9/summarize/SUMMARY.md`。

---

## 一、七个抽象原则

### 原则 1：梯度干扰强度决定 Direct 是否可用

**退化 A 和 B 对模型参数发出不同方向的梯度。方向越冲突，同时训练的干扰越强，Direct 越不可用。**

- 弱干扰（梯度方向一致）→ Direct 或 Ft 即可。典型：noise+blur（都指向边缘恢复）
- 强干扰（梯度方向冲突）→ 必须 Curric 解耦。典型：comp+blur（去压缩平滑 vs 去模糊锐化）

**⚡ 待验证预测**：任意两个退化方向冲突的组合都应该 Curric > Direct，不限于 comp+blur。测试 oversharpen+blur、pixelate+noise。

### 原则 2：Phase1 迁移价值决定 Curric 方向

**不是"施加顺序"或"外层内层"决定方向——是哪个退化单独修复后的特征对后续任务最有帮助。**

- 高迁移价值：去压缩（块效应识别通用）、去 motion blur（1D反卷积可迁移）、去噪声（信号分离通用）
- 低迁移价值：去 lens blur（径向PSF太专一）、去 impulse noise（模式太特殊）

**⚡ 待验证预测**：Phase1=高迁移价值退化时，Curric 方向不重要（Fwd/Rev 都行）；Phase1=低迁移价值退化时，Fwd/Rev 差异大。

### 原则 3：修复难度 ≠ 破坏程度，严重度非线性

**严重的噪声（破坏重但极易修）≠ 严重的模糊（破坏轻但极难修）。**

- noise_impulse(4)：damage PSNR=16.7（极重），expert PSNR=43.1（极易修）
- blur_lens(3)：damage PSNR=21.5（较轻），expert PSNR=24.5（难修）

**⚡ 待验证预测**：按"修复难度"分配步数优于按"破坏程度"或"严重度标签"分配。

### 原则 4：预训练通用性与 Curric 特化互斥

**盲预训练给的是通用特征。Curric 的单退化阶段会把通用特征特化→不可逆。**

- Ft：通用→精调 ✅
- Curric：从零渐进 ✅
- FtCurr：通用→特化破坏→重建 ❌ (11/12 崩)

**⚡ 待验证预测**：如果 Curric Phase 间退化类型不变（只改 LR/loss/severity），不再破坏通用特征，FtCurr 可能有效。

### 原则 5：修复顺序 ≠ 退化施加的逆序

**修复时的最优顺序由迁移价值决定，不由"外层先剥"决定。**

**⚡ 待验证预测**：存在退化组合，最优修复顺序与施加顺序相同（先修内层迁移价值高）。

### 原则 6：训练预算的边际效用因退化类型而异

**给已"饱和"的退化（noise_impulse）多步数是浪费。给"上升中"的退化（blur_motion）多步数有效。**

**⚡ 待验证预测**：动态 split 只在瓶颈退化处于"上升区"时有效（如 motion blur），在饱和退化瓶颈（如 noise）时无效。

### 原则 7：策略效应量级的分层

```
梯度干扰     → 决定 Direct vs Curric       (0~+5 dB)
迁移价值     → 决定 Curric 方向            (0~+2 dB)
修复难度差   → 决定是否需要动态 split      (0~+0.5 dB，有时无效)
LR/loss微调  → 影响最小                    (< 0.3 dB)
```

---

## 二、策略选择框架

```
输入退化管线 → 

Step 1: 评估梯度干扰
  退化之间梯度方向冲突? → 需要 Curric
  否则 → Direct 或 Ft 即可

Step 2: 是否用盲预训练?
  sev ≤ 3 → Ft (优势 +0.3~+5 dB)
  sev ≥ 4 → Ft ≈ Direct (选简单的)
  用 Curric → 不要预训练 (原则4)

Step 3: 如果用 Curric, Phase1 选哪个?
  迁移价值最高的退化 → Fwd/Rev 均可
  迁移价值接近 → Rev 略优
  不确定 → Rev (22/42 > Fwd 6/42)

Step 4: 步数分配
  默认 Fixed 5000/5000/5094
  仅在明确瓶颈 + 上升区退化时考虑动态 split
```

---

## 三、实操建议

| 场景 | 策略 | 原因 |
|------|------|------|
| 不确定 | Ft | 永远不会显著差 |
| 退化间明显冲突 | Curric | 解耦梯度干扰 |
| Phase1 迁移价值高 | 方向不重要 | Fwd/Rev 都行 |
| 严重度 ≥ 4 | Direct 或 Ft | 预训练优势消失 |
| 噪声瓶颈 | Fixed split | 去噪迅速饱和，多给步数无用 |
| **全局退化 (contrast/brightness)** | **Direct** | **Ft 可能崩溃 (-4.51 dB)** |

### 原则 8：全局退化不遵循局部退化规律

> 来源：exp9 Phase 11-15，已验证于 L1-L6 全局退化

| 属性 | 局部退化 (blur/noise/comp) | 全局退化 (brightness/contrast/saturation) |
|------|:--:|:--:|
| Ft 是否安全？ | ✅ 是 | ❌ 否 (L2 noise+contrast: Ft-Direct = **-4.51 dB**) |
| Curric 是否有效？ | 有时 | 未知 |
| 盲预训练覆盖？ | ✅ | ❌ (随机管线未包含) |
| 推荐策略 | Ft (EPOCH=2, LR=5e-4) | **Direct** (EPOCH=2, LR=5e-4) |

**判定规则**：若管线包含 contrast_strengthen/weaken、brightness_brighten/darken、saturate_strengthen/weaken、gamma_HSV/RGB → 使用 Direct，不用 Ft。

---

## 四、已完成的验证实验

### 实验 1：梯度干扰泛化 ✅ 已完成
- O1 (oversharpen+blur): **Ft > Direct +3.55 dB**，强梯度干扰确认。但 Curric 未超越 Ft。
- O2 (oversharpen(4)+motion): Ft 微弱最优，策略差异小。
- P1 (pixelate+noise): 策略差异 < 0.5 dB，弱梯度干扰。
- G1 (oversharpen三退): 所有策略 ~27 dB，退化太轻。
- G2 (pixelate+blur+jpeg): 全部 ~22.8 dB，完全平手。
- **结论**：原则1部分成立——梯度干扰确实存在（O1 Ft>Direct +3.55），但 Curric 不能因此超越 Ft。

### 实验 2：Phase1 迁移价值 ✅ 已完成
- V1 (gauss+motion+jpeg): Ft≈Direct≈Curric，无差异。
- V2 (gauss+lens+jpeg): Ft > Curric by 1.6 dB。
- V3 (motion+gauss+jpeg): Direct > Ft > Curric。
- **结论**：原则2未验证——Phase1迁移价值差异未转化为Curric方向差异。V1/V2/V3的Fwd/Rev最多差0.7dB。

### 实验 3-4：待实现

### MD 实验（边际难度分配）✅ 已完成
- 14 组，MD ≈ Fixed 在大多数退化上
- S5_rev_MD +1.28, S8_fwd_MD +0.98（少数有效case）
- **结论**：非普适改进，split微调不应作为主要方向

### DFPIR 基线对比 ✅ 已完成
- 15 组退化，**小模型(0.45M) 12/15 超越 DFPIR(31M)**
- G1 (oversharpen三退) 小模型 27.40 vs DFPIR 20.90 (-6.45!)
- **结论**：核心假设验证——专攻小模型 > 通用大模型
