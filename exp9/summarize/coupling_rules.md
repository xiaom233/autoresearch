# 退化类型和强度与训练策略的耦合规律

> 基于 227 组实验 (Phase 1-8)，EMBED_DIM=64，EPOCH_BUDGET=2

---

## 一、退化类型决定是否迁移学习

```
退化分布"常见" → Ft (盲预训练→专攻)
退化分布"特殊" → Direct 可能更好
```

"常见" = 退化在盲预训练随机分布中出现频率高；"特殊" = 严重度极端、顺序罕见。

| 退化特征 | Ft vs Direct 优势 | 机制 |
|----------|:--:|------|
| noise-first 双退化 | **+0.5~+5 dB** | 盲预训练中 noise-first 少见，先验差距最大 |
| blur-first 双退化 | +0.3 dB | 盲预训练中 blur-first 常见，先验已够用 |
| compression 相关 | ~0 dB | 压缩伪影在盲预训练中充分覆盖 |
| 三退化 | +0.1~+0.4 dB | 复杂度本身是主要瓶颈 |

**规则**：永远先跑 Ft。只有 Ft≈Direct (<0.1 dB) 时才考虑 Direct。

---

## 二、退化交互强度决定课程顺序

**判断标准**：前一个退化是否会改变后一个退化的统计分布？

| 退化组合 | 交互强度 | 机制 | 课程方向 |
|----------|:--:|------|:--:|
| noise + blur | **弱** | blur 抹平 noise：blur(noise)≈blur(clean) | **Rev** (逆序剥离) |
| compression + blur | **强** | JPEG 块效应被 blur 扩散成新伪影，去 blur 统计分布改变 | **Fwd** (正序渐进) |
| compression + noise | **弱** | 两者相对独立 | 无所谓 |
| blur + compression | **中** | 取决于 compression 位置 | comp 在外层→Rev |

**证据**：
- D2 (noise→blur, 弱交互): Rev > Fwd +1.95
- D3 (comp→blur, 强交互): Fwd > Rev +4.27
- N6 (jpeg2000+motion): Rev > Fwd +1.34
- N8 (blur+jpeg): Rev > Fwd +1.82

---

## 三、blur 子类型决定三退化的课程方向

| blur | Rev-Fwd 均值 | 方向 | 交叉验证 | 机制 |
|------|:--:|:--:|:--:|------|
| **motion** | **-1.19** | **Fwd** | ✅ impulse 噪声下仍 Fwd (-1.80) | 线性拖影，结构简单近乎可逆，1D 反卷积是好基础 |
| **lens** | **+1.28** | **Rev** | ✅ poisson 噪声下仍 Rev (+1.75) | 径向散焦 PSF 复杂，单独修复困难，先学 JPEG 打基础 |
| gaussian | ~0 | ≈平 | — | 各向同性平滑，无方向偏好 |
| jitter | +0.03 | ≈平 | — | 随机像素位移 |
| zoom | +0.14 | ≈平 | — | 径向缩放 |
| glass | +0.08 | ≈平 | — | 纹理/毛玻璃模糊 |

**规则**：
- blur=motion → **Curric(fwd)** 是唯一正确选择
- blur=lens → **Curric(rev)** 是唯一正确选择
- 其余 blur → Ft 或 Mixed Warmup，方向无所谓
- 不确定 blur 子类型时 → **Mixed Warmup** 最安全（比选错 Fwd 平均 +1.1 dB）

---

## 四、三退化的 Mixed Warmup 策略

用概率混合替代硬切换，消除 Curric 的 phase 边界冲击。

| 场景 | 推荐 | 原因 |
|------|------|------|
| 三退化 + 不确定 blur | **MW** | 6/10 最优，比 Fwd 平均 +1.1 dB |
| 三退化 + 已知 blur=motion | **Curric(fwd)** | Fwd 确定性优于 MW |
| 三退化 + 已知 blur=lens | **Curric(rev)** | Rev 确定性优于 MW |
| 双退化 | **Curric** | MW 在双退化上不占优 (2/8) |

---

## 五、严重度的影响（初步，待验证）

| 严重度 | 观察 |
|--------|------|
| sev=3 (中等) | 策略差异最大，规律最清晰 |
| sev=4 (M8) | 策略差异缩小 (Ft≈Direct)，但课程方向保持 (Rev > Fwd +1.30) |
| sev=5 | 未知——策略饱和还是保持？ |
| sev=1 | 未知——所有策略平手？ |
| 不平衡 | 未知——哪个步骤的严重度主导决策？ |

---

## 六、决策流程

```
输入: 退化管线 P = [(f1,s1), (f2,s2), ...]

Step 1: N = 2 (双退化)?
├── 含 compression?
│   ├── compression 在最外层 → Curric(rev)
│   └── compression 在内部   → Ft (或具体分析)
├── noise-first?  → Ft (优势最大)
└── blur-first?   → Ft (足够)

Step 2: N ≥ 3 (三退化)?
├── blur = motion?   → Curric(fwd)
├── blur = lens?     → Curric(rev)
├── 其余 blur?       → Ft 或 Mixed Warmup
├── 不确定 blur?     → Mixed Warmup
└── sev ≥ 4?        → 策略差异缩小, Ft 即可
```

---

## 七、不要做的事

- 不要在 Loss 函数上花时间（l1 够用，Δ < 0.25 dB）
- 不要尝试 FtCurr（盲预训练后课程，D2 崩 -6.83）
- 不要尝试多专家级联（从未超越单模型 Ft）
- 不要在双退化上用 Mixed Warmup（Curric 更好）
- 不要在三退化上默认用 Ft（中位 Δ = 1.22 dB，策略重要）

---

## 八、证据强度

| 结论 | 样本 | 反例 | 强度 |
|------|:--:|:--:|:--:|
| Ft 是安全默认 | 30+ | 0 | **强** |
| motion→Fwd, lens→Rev | 4 | 0 | **强** (交叉验证) |
| 其余 blur 中性 | 4 | 0 | **中** |
| MW 三退化安全 | 10 | 4 (Rev 胜) | **中** |
| compression 外层→Rev | 2 | D3, N7 | **弱** |
| 严重度效应 | 1 | — | **极弱** (仅 M8) |
| Loss 影响 < 0.25 dB | 6 | D2(l1+fft) | **中** |
| 多专家无效 | 6 | 0 | **中** |
