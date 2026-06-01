# 退化与训练策略的耦合机制 —— 271 组实验的深度理解

## 一、核心原理：三个独立机制

退化与训练策略的耦合由三个独立机制驱动。它们不是同一现象的三个方面——是三个**不同**的因果链条。

```
机制 1: 梯度干扰 → Curric vs Direct
机制 2: Phase1 迁移价值 → Fwd vs Rev  
机制 3: Pretrain 通用性 → FtCurr 为什么崩
```

---

## 二、机制 1：Standalone Curric 有效 = 梯度干扰减少

### 问题：Direct 为什么差？

多退化同时训练时，每个退化的修复目标给模型发送**不同方向的梯度更新**：

```
Direct 训练: 去 blur、去 noise、去 compression 同时学

去模糊:  梯度要求参数 → "让边缘更锐利"
去噪声:  梯度要求参数 → "让平坦区域更平滑"
去压缩:  梯度要求参数 → "消除 8×8 块边界"

三个梯度方向不同 → 同一个参数被三个信号撕扯 → 收敛慢、局部最优
```

### Curric 如何解决？

```
Curric Phase 1: 只有 1 个退化 → 梯度方向纯粹 → 快速建立好基础
Curric Phase 2: 2 个退化 → 已有一个基础，梯度干扰减半
Curric Phase 3: 3 个退化 → 已有两个基础，梯度干扰已是"叠加"而非"撕扯"
```

**Curric 不是在"教模型更简单的东西"——它是在减少训练早期的梯度干扰。** 这和传统课程学习的"从易到难"有本质区别：这里的"简单"不是更容易的任务，而是**更少方向的梯度更新**。

### 证据

D3 (comp+blur) 是最强的梯度干扰案例：
- Direct = 20.99（被梯度撕扯崩溃）
- Curric(fwd) = 25.26（+4.27 dB, Phase1 干净的 compression 梯度建立基础）

---

## 三、机制 2：Fwd vs Rev = Phase1 哪个退化迁移价值更高

### 问题：Phase1 应该学哪个退化？

给定 N 步退化管线，Curric Phase1 只学一个退化。**选哪个？**

| 退化类型 | Phase1 学到什么 | 对后续 Phase 的迁移价值 | 原因 |
|----------|---------------|:--:|------|
| **motion blur** | 1D 线性反卷积 | **极高** | 边缘重建是所有修复任务的基础 |
| **compression** | 结构化伪影识别 | **高** | 块效应模式识别是通用感知能力 |
| **gaussian noise** | 信号/噪声分离 | **高** | 与方向无关的通用技能 |
| **lens blur** | 径向 PSF 反卷积 | **低** | 太过专一，只对 lens 本身有用 |
| **impulse noise** | 离群值检测 | **低** | 脉冲模式太特殊，无法迁移 |

### 决策规则

```
Phase1 应该选"迁移价值最高的退化"

motion → Phase1(Fwd): motion的1D反卷积是所有修复的基础 → Fwd
lens   → Phase1(Rev): lens的径向PSF太专一, 换成compression → Rev
noise  → Phase1: 迁移价值高但与方向无关 → Fwd/Rev都可以

Fwd: Phase1=第一步退化, Rev: Phase1=最后一步退化
```

### 证据

- motion: 不论配 gauss 还是 impulse，Fwd 始终优（-0.58, -1.80）
- lens: 不论配 gauss 还是 poisson，Rev 始终优（+0.81, +1.75）
- compression: Phase1 迁移价值高，所以在 Rev 中当 Phase1 也有效

---

## 四、严重度：调节"迁移价值能否在预算内兑现"

### 问题：为什么 sev=5 时 motion→Fwd 翻转？

迁移价值高，但**需要足够的训练预算才能兑现**。如果 Phase1 的 5000 步不够学好 motion(sev=5)，那迁移价值就无法兑现——不如选另一个退化。

```
motion sev=3: 5000步 ✅ 能学好1D反卷积 → 迁移价值兑现 → Fwd 优
motion sev=5: 5000步 ❌ 学不好 → 迁移价值无法兑现 → 换 Rev

lens sev=3:   5000步 ❌ 学不好径向反卷积 → 从未兑现 → 始终 Rev
lens sev=5:   同上 → 始终 Rev
```

### 不同退化的"严重度-可学性"曲线

```
             sev=1   sev=2   sev=3   sev=4   sev=5
motion:       ✅      ✅      ✅      ✅      ❌   (在 sev=5 不可学)
lens:         ❌      ❌      ❌      ❌      ❌   (从未可学)
gaussian:     ✅      ✅      ✅      ✅      ✅   (始终可学)
compression:  ✅      ✅      ✅      ✅      ✅   (始终可学)
noise:        ✅      ✅      ✅      ✅      ✅   (始终可学)
```

**严重度不直接决定策略——它通过改变"给定预算下能否学好"的阈值来间接影响。**

### 不平衡严重度的解释

```
S8: lens(1)+noise(5)+jpeg(1), Rev-Fwd=+2.47
  → noise(5) 是唯一"难"的退化，迁移价值极高
  → Rev: Phase1=jpeg → Phase2=jpeg+noise(5) → 先打底再攻关
  → Rev 优势极度强化

S7: lens(5)+noise(1)+jpeg(1), Rev-Fwd=+0.01
  → lens(5) 不可学（始终如此），noise(1) 可学但太简单（梯度过弱）
  → 两个方向都缺乏高迁移价值的 Phase1
  → 差异消失
```

---

## 五、机制 3：为什么 Pretrain + Curric 一定崩

### 数据

12 组 FtCurr 中 11 组差于 Ft。唯一"平手"是 T1fcr(+0.05)。

### 原因：Pretrain 已经提供了通用解

```
Pretrain (随机退化): 模型学到的不是"去某个退化"的技能
                    而是"修复图像的通用能力"
                    边缘重建、纹理生成、平滑去噪——全部都有

Ft Phase2 (直接专攻): 在通用能力基础上精调 → 好用
FtCurr Phase2 (单退化): 把通用能力特化成专用能力 → 自废武功
FtCurr Phase3 (全退化): 专用能力无法重新泛化 → 崩
```

**Pretrain 的通用特征已经被 Curric 的梯度干扰破坏了。** Curric 有效的前提是从零开始（无通用特征可破坏）——但 Pretrain 已经给了通用特征，Curric 只是在摧毁它。

这和 NLP 中 BERT fine-tune 到窄任务后失去广泛语言理解是同一个现象。

### 这意味着什么

**"Ft + Curric"不应该是决策树的一个选项。** 如果用 Ft，就不要加 Curric。如果用 Curric，就不要 Pretrain。两者互斥。

---

## 六、统一框架

```
决策树 (修正版):

Step 1: 用不用盲预训练？
  ├── sev ≤ 3 → Ft (盲预训练优势大)
  │   之后不加 Curric ← 关键!
  └── sev ≥ 4, 或退化组合特殊 → 考虑 Standalone Curric
      ├── Phase1 = 迁移价值最高的退化
      │   motion/compression/noise: 高迁移价值
      │   lens: 低迁移价值, 换别的
      └── 严重度影响: Phase1的退化必须在预算内可学好

Step 2: 课程方向？
  ├── 迁移价值最高的退化在第一步 → Fwd (Phase1=motion/noise)
  ├── 迁移价值最高的退化在最后 → Rev (Phase1=compression)
  └── 两者迁移价值接近 → 方向无所谓
```

### 什么不重要（始终 < 0.3 dB）

- Loss 函数选择
- Fine-tune Phase2 的微调方式
- 多专家架构
- 混合预热 (MW) 在双退化上

---

## 七、耦合强度：为什么 compression + blur 是断层式最强

### 数据

| 退化 | Direct→Curric Δ |
|------|:--:|
| **compression_jpeg(3) + blur_lens(4)** | **+4.09** |
| blur_gaussian(4) + noise_speckle(4) | +0.36 |
| noise_impulse(4) + blur_motion(3) | +0.05 |
| 其他 | < 0.3 |

D3 的耦合强度是第二名的 **11 倍**。这不是量变，是质变。

### 原因：compression + blur 创造了"第三种伪影"

**独立退化（noise + blur）**：
```
noise(img) → 随机像素翻转
blur(noise(img)) → 模糊把噪声抹平了
→ 视觉上 ≈ blur(clean_img)
→ 修复任务 = 去模糊（噪声信号已被覆盖）
```
噪声和模糊是**加性/独立的**——一个不做的事不影响另一个做的事。梯度方向一致（都要恢复边缘）。

**交互退化（compression + blur）**：
```
jpeg(img) → 8×8块效应 + DCT量化 + 高频细节丢失
blur(jpeg(img)) → 块边界被模糊扩散 → "被模糊的块效应"
→ 视觉上 ≠ blur(clean_img), ≠ jpeg(img), ≠ jpeg(blur(img))
→ 修复任务 = 去"同时被两种退化污染"的伪影，这种伪影在任何单一退化中都不存在
```

**compression + blur 不是两个问题的叠加，而是一个新的第三种问题。** Direct 训练时，模型收到三组梯度：
- "消除块效应"（需要平滑块边界区域的梯度）
- "去模糊"（需要锐化边缘区域的梯度）
- "消除被模糊扩散后的块效应"（既不能纯平滑也不能纯锐化）

第三组梯度与前两组**互相矛盾**——这在噪声+模糊中不存在。

### 为什么 Curric 能解决

Curric Phase1 只学 compression——先建立了"什么是块效应、怎么去"的认知。Phase2 叠加 blur 时，模型面对的是"在已去块的基础上再去模糊"——梯度干扰被分解成了顺序的两个单任务。

**Curric 不是让任务变简单——是把"第三种伪影"的不可解梯度冲突，分解成两个顺序的可解任务。**

### severity 放大效应

为什么 D3(sev=3,4) 比 N1(sev=4,4) 强 11 倍？因为严重的 compression + 严重的 blur = 第三种伪影被**极度放大**。模糊扩散得更远，块效应污染更大的区域——梯度冲突的区域更大。

### 判断"强耦合"的标准

```
问: 退化A的产物，在退化B的作用下，会产生新的、在任何单一退化中都不存在的伪影模式吗？
  → 不会: 弱耦合，Direct 够用，Curric 收益 < 0.5 dB
  → 会: 强耦合，Direct 不可用，Curric 可能拯救数 dB
```

---

## 八、退化属性 → 训练决策 精确映射

以下是从 271 组实验中提取的退化属性与最优训练策略的对应关系。

### 退化属性有哪些

影响训练策略的退化属性共 6 个：

| 属性 | 取值 | 影响什么决策 |
|------|------|-------------|
| **退化类别** | blur / noise / compression | Ft vs Direct, 迁移价值 |
| **blur 子类型** | motion / lens / gaussian / jitter / zoom / glass | Curric 方向 |
| **严重度** | 1-5 | Ft 优势大小, 方向是否翻转 |
| **退化步数** | 2 (双退化) / 3 (三退化) | 是否用 Curric |
| **顺序** | blur-first / noise-first / comp-first / comp-last | Curric 方向 |
| **类别组合** | blur+noise / comp+blur / comp+noise / blur+comp | 耦合强度 |

### 属性 → 决策映射

#### 退化类别 → Ft vs Direct

```
noise-first  → Ft 优势最大 (+2.66 均值) 
              盲预训练中 noise-first 最少见, curriculum 效应最强

blur-first   → Ft 优势中等 (+0.28)
              盲预训练中 blur-first 最常见, 迁移收益小

compression 相关 → Ft ≈ Direct (+0.05 到 -0.03)
                  压缩伪影在盲预训练中充分覆盖
```

#### blur 子类型 → Curric 方向

```
motion → Fwd (Rev-Fwd = -1.19 均值)
        1D 线性反卷积: 5000步可学好, 迁移价值极高

lens   → Rev (Rev-Fwd = +1.28 均值)
        径向 PSF: 5000步学不好, 换 compression 作为 Phase1

jitter/zoom/glass/gaussian → ≈平 (|Rev-Fwd| < 0.2)
        各向同性或随机模糊, 方向无所谓
```

#### 严重度 → 调节上述两个决策

```
对 Ft vs Direct:
  sev 1→5: Ft 优势 +2.55 → -0.03 (严格单调递减)
  机制: 盲预训练与目标退化的难度差决定了 curriculum 收益

对 Curric 方向:
  motion sev=5: Fwd→Rev 翻转 (Rev-Fwd: -0.58 → +0.39)
  lens: 不受严重度影响 (始终 Rev)
  机制: 严重度改变"Phase1 能否在预算内学好"的阈值

对不平衡严重度:
  严重步骤主导 Phase1 选择
  noise/comp 严重 → Rev (先剥外层)
  blur 严重 → motion 强化 Fwd, lens 取消 Rev
```

#### 退化步数 → 是否用 Curric

```
双退化: Curric 收益因退化类型而异 (0 到 +4.27 dB)
三退化: Curric 中位收益 +1.22 dB, 71% ≥ 0.5 dB
       除 T2 (策略饱和) 外, 三退化普遍受益于 Curric
```

#### 顺序 → Curric 方向

```
compression 在外层 (最后施加) → Rev
  先剥外层 compression, 再修内层退化

compression 在内层 (最先施加) → Fwd 或 Ft
  D3: Fwd, N7: Direct

noise-first 双退化 → Fwd (如用 Curric)
  noise 是内层, blur 是外层, 先用 noise 打底

blur-first 双退化 → Rev (如用 Curric)
  外层剥离更自然, 但严重度可能翻转这个偏好
```

#### 类别组合 → 耦合强度

```
compression+blur → 极强耦合 (Direct→Curric +4.09 dB)
  第三种伪影: 块效应被模糊扩散, 梯度互斥

compression+noise → 弱耦合 (所有策略 ≈平手)
  两者相互独立, 梯度不冲突

blur+noise → 弱-中耦合 (0 到 +2.3 dB)
  取决于严重度和顺序, 噪声被模糊"抹平"

blur+compression → 中耦合 (取决于 compression 位置)
  comp 在外层: Rev 有效；comp 在内层: 需具体分析
```

### 一句话决策树

```
输入退化 → 
  含 compression? → 耦合强, 考虑 Curric
    comp 在外层 → Rev
    comp 在内层+blur → Fwd
  noise+blur? → Ft 优势大, 如需 Curric→Rev
  三退化? → 几乎总是 Curric 优于 Direct
    blur=motion → Fwd
    blur=lens → Rev
    其余 → Ft 或 MW
  不确定 → Ft (30+组从未显著输)
```

---

## 九、可验证预测

1. **增加 Phase1 预算 → motion×sev=5 从 Rev 回到 Fwd**
   如果给 motion(sev=5) 10000 步而不是 5000 步，能学好 → 迁移价值兑现 → Fwd 恢复优势

2. **减少 Phase1 预算 → gaussian 可能出现方向偏好**
   如果只给 2000 步，gaussian(sev=3) 也学不好 → 需要换方向

3. **lens 在任意预算下都不可学 → Rev 永远不被翻转**
   即使给 15000 步，径向 PSF 反卷积也不会在单个退化上学好

4. **混合预热 (MW) 在 Pretrain 后也不应该有用**
   MW 本质是软化的 Curric，如果 Curric 在 Pretrain 后崩，MW 也会崩
# 退化与训练策略的抽象规律

> 不是"comp+blur 用 Curric(fwd)"——而是"当退化之间存在梯度干扰时，需要解耦训练"。

---

## 一、退化之间的梯度干扰强度

**退化 A 和 B 对模型参数发出不同方向的梯度更新。方向差异越大，同时训练的干扰越强。**

```
噪声+模糊: 梯度方向一致 → 都指向"恢复干净边缘" → 弱干扰，Direct 即可
压缩+模糊: 梯度方向相反 → 去压缩要平滑块边界，去模糊要锐化边缘 → 强干扰，Direct 崩溃
```

**规律**：两个退化的梯度方向越趋于正交或相反，越需要 Curric 解耦。

**可推广**：不限 comp+blur。任何两个目标冲突的退化都应考虑 Curric。例如：
- oversharpen + blur（过度锐化产生鬼影→模糊扩散鬼影→梯度冲突）
- pixelate + noise（像素化产生块→噪声掩盖块边界→梯度冲突）

---

## 二、Phase1 的迁移价值

**Phase1 不是"学容易的退化"，而是学"迁移价值最高的退化"。**

迁移价值 = Phase1 学到的特征在后续 Phase 中是否可直接复用。

```
高迁移价值:  去压缩（块效应识别→任何后续任务都可复用）
            去 motion blur（1D反卷积→边缘重建是基础能力）
            去噪声（信号/噪声分离→与具体退化无关的通用能力）

低迁移价值:  去 lens blur（径向PSF→只对lens自身有用，无法迁移）
            去 impulse noise（离群值检测→太特殊，无法迁移）
```

**规律**：Phase1 应选迁移价值最高的退化，而非"第一个施加的退化"或"最严重的退化"。

**可推广**：对于新的退化类型，问"学会修复这个退化后，学到的特征对其他退化有用吗？"→ 有用则作为 Phase1。

---

## 三、严重度非线性

**sev=5 ≠ 5倍难度。**

```
noise_impulse(4): 破坏严重(damage PSNR=16.7), 但极易修复(expert PSNR=43.1)
blur_lens(3):     破坏较轻(damage PSNR=21.5), 但极难修复(expert PSNR=24.5)
```

**规律**：修复难度 ≠ 破坏程度。严重的噪声 5000 步就能学好，轻微的 lens blur 10000 步也学不好。

**可推广**：分配训练预算时，应按"修复难度"而非"破坏程度"或"严重度标签"。

---

## 四、预训练与课程学习的互斥

**盲预训练已经给了通用特征。再加 Curric 会破坏它们。**

```
Ft:     [盲预训练: 通用] → [专攻: 精调] = 通用+专攻 ✅
FtCurr: [盲预训练: 通用] → [单退化: 特化破坏通用] → [全退化: 无法重新泛化] = 崩 ❌
Curric: [随机权重: 从零开始] → [渐进叠加: 无通用特征可破坏] = 从零构建 ✅
```

**规律**：盲预训练后的特征是"脆弱通用性"——一旦被单任务特化，就无法恢复。

**可推广**：如果要用 Curric，不要用预训练。如果用了预训练，不要加 Curric。两者不可兼得。

---

## 五、退化施加顺序与修复顺序

**修复顺序 ≠ 退化施加的逆序。修复顺序应由迁移价值决定。**

```
施加: clean → [A] → [A+B] = degraded
修复: 不应总是先剥B（外层），再修A（内层）

应先修"迁移价值高"的那个，不管它在内层还是外层
```

**规律**：修复时，先处理"学了对后续有帮助"的退化，而非"最后施加"的退化。

---

## 六、训练预算的边际效用

**给一个退化更多训练步数的收益，取决于该退化是否接近"可学性饱和点"。**

```
饱和前: noise_impulse → 2000步就饱和（再多步数无用）
上升中: blur_motion → 5000步还在上升，给更多步数有效
永远不饱和: blur_lens → 10000步仍在上升，但斜率很低
```

**规律**：动态分配步数的效果取决于退化是否在"可学性上升区"。已饱和的退化多给步数是浪费。

---

## 七、综合决策框架

```
给定退化管线 P

1. 评估梯度干扰: P中退化两两之间的梯度方向是否冲突?
   → 强冲突: 必须 Curric (Direct 不可用)
   → 弱冲突: Direct 或 Ft 即可

2. 选择 Phase1: 哪个退化单独修复后的特征最有迁移价值?
   → 不要问"哪个是第一/最后施加的"
   → 要问"学会修这个后, 对其他退化的修复有帮助吗?"

3. 是否需要预训练?
   → 如果用 Curric: 不要预训练
   → 如果用 Ft: 预训练是核心

4. 分配步数:
   → 给"还在上升区"的退化更多步数
   → 已饱和的退化少给步数
   → 不改变退化类型 (防止灾难性遗忘)
```

## 八、验证状态

### 已验证 ✅

| 原则 | 验证结果 | 证据 |
|------|----------|------|
| 梯度干扰：oversharpen+blur Ft>Direct +3.55 | ✅ 成立 | O1 实验 |
| 梯度干扰：pixelate+noise 弱干扰 | ✅ 成立 | P1 实验 |
| 小模型专攻 > 通用大模型 | ✅ 成立 | 12/15 DFPIR对比 |
| motion在P1+sev≥3→Fwd>Ft | ✅ 成立 | 4/4 满足条件的case |

### 未验证 ❌

| 原则 | 验证结果 | 证据 |
|------|----------|------|
| 梯度干扰→Curric>Ft | ❌ 未成立 | O1 Ft +3.55但Curric未超越 |
| Phase1迁移价值决定方向 | ❌ 未成立 | V1-V3 Fwd/Rev差 <0.7dB |
| 动态split是普适改进 | ❌ 证伪 | MD≈Fixed, TA≤Fixed |
| 修复难度>破坏程度分配 | ⏳ 待验证 | 实验3未完成 |

### 修正后的核心规律

1. **梯度干扰存在但不保证Curric>Ft** — Ft已经很好地处理了干扰
2. **motion P1 + sev≥3 是唯一Curric>Ft的可靠方向** — 4/4 case
3. **DFPIR大模型不如专攻小模型** — 核心假设验证
4. **split微调不值得投入** — MD/SL/TA/PR全部 ≤ Fixed
4. **修复难度相似的双退化 → 固定 split 最优**
# 实验教训总结

## 一、有效的发现

### 1. Ft 是最安全的默认策略（30+组验证）

盲预训练→专攻永远不会显著输给任何策略。sev=1 时优势最大(+2.55 dB)，sev=5 时 Direct 微弱反超。

### 2. Standalone Curric 在特定退化上显著优于 Direct

D3 (comp+blur): Curric +4.09 dB。机制是梯度干扰减少——comp 和 blur 产生"第三种伪影"，Direct 的梯度互相冲突。

### 3. blur 子类型决定 Curric 方向（motion→Fwd, lens→Rev）

交叉验证通过（换 noise 类型偏好不变）。机制是 Phase1 迁移价值：motion 的 1D 反卷积是可迁移的基础技能。

### 4. FtCurr（盲预训练+课程）几乎总是最差

12 组中 11 组差于 Ft。原因：盲预训练已给通用特征，再插单退化阶段破坏通用性。

## 二、被证伪的假设

### 1. "三退化策略差异 < 0.5 dB" → 证伪

17 组三退化中位 Δ = 1.22 dB，71% ≥ 0.5 dB。策略在三退化上同样重要。

### 2. "PR 峰值比能零成本预测最优分配" → 证伪

PR 全部差于 Fixed split。峰值比过度偏好结构化退化（motion peak=92），饿死其他 phase。

### 3. "TA 动态 split 能系统提升 Curric" → 证伪

Phase 13 系统验证：TA 在 B1-B9 上全部 ≤ Fixed。Phase 11 的 T2/N12 收益（+2.91/+3.97）是特例，无法复现。

### 4. "动态 split 是普适改进" → 证伪

Fixed 5000/5000/5094 在 13/13 组上都是最稳健的。动态 split 有时更好（T2,N12），但无法提前预测。

### 5. "多专家级联能超越单模型 Ft" → 证伪

15 专家 + 6 联合微调，从未超越 Ft。误差级联放大。

### 6. "Loss 函数选择对结果有显著影响" → 证伪

l1/mse/huber/l1+edge/l1+fft 之间 Δ < 0.25 dB。l1 是最安全默认。

### 7. "Mixed Warmup 是 Curric 的升级版" → 部分证伪

三退化上 MW 优于 Curric(fwd)，但双退化上 Curric 更好。MW 不是普适改进。

### 8. "Ft 优势来自盲预训练见过类似退化" → 证伪

Ft 优势在 sev=1 时最大(+2.55)，在 sev=5 时消失。如果"见过类似"是对的，应该 sev=5 时优势最大。真正机制是盲预训练→目标的难度差作 curriculum 效应。

## 三、仍需探索的未知

### 1. 边际难度分配（Phase 13b）

当前 TA 按累计严重度分配——Phase3 总是最多步数。但 Phase1 才是打地基的阶段。改为按新增退化的边际难度分配。待验证。

### 2. 严重度×预算的交互

EPOCH=0.5/1/2 的 budget-aware 实验（Phase 12）尚未完成。低预算下策略差异可能更大。

### 3. 为什么 T2/N12 的 TA 收益不能复现

这两个是唯一 TA > Fixed 的案例。需要理解是 noise_corr + blur_motion + jpeg2000 这个特殊组合的什么属性导致了 TA 有效。

## 四、实验方法教训

### 1. 每 GPU 最多 1 个实验

违反导致 OOM。已写入 CLAUDE.md。

### 2. 需要 Direct 和 Ft 基线

只比较 Curric 内部变体不够——需要知道 Curric 是否优于 Direct/Ft。

### 3. gaussian blur 不能代表所有 blur

Phase 13 用 gaussian 作为中性基线，但 TA 在 gaussian 上无效不等于在其他 blur 上无效。motion/lens 的行为不同。

### 4. 严重度非线性

sev=5 的 noise_impulse (PSNR 43) 比 sev=3 的 blur_lens (PSNR 24) 容易得多。严重度标签本身不反映修复难度。

### 5. 固定 split 是强基线

5000/5000/5094 经过大量实验检验。MD/SL/TA/PR 四种动态方案均未系统超越 Fixed。

### 6. motion P1 + sev≥3 是唯一 Curric > Ft 的可靠方向

4/4 满足条件：S5(+1.29), M1(+0.22), X2(+0.09), M13(+0.08)。其他条件均 Ft 胜。

### 7. DFPIR 大模型不如专攻小模型

15 组退化对比，小模型(0.45M) 12/15 超越 DFPIR(31M)。G1 oversharpen 三退化 -6.45 dB。

### 8. 梯度干扰存在但 Curric 无法利用

O1 oversharpen+blur Ft > Direct +3.55，但 Curric 未超越 Ft。梯度干扰确实存在但 Ft 已经解决了。

### 9. split 微调不应作为主要方向

TA/SL/PR/MD 四种动态 split 方案，无一种系统优于 Fixed split。效应量级 < 0.5 dB，不值得投入。
