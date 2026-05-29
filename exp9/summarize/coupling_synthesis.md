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
