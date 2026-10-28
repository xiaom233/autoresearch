# E1 one-shot（无反思）对照：rule-only vs LLM reflection

> 目的：回答“目前基于规则的方法会有反思吗？”，并构造一个**没有反思过程、规则预测一次退化、只训练一次**的简化 E1 基线，用来把“LLM 反思”的贡献从“controller 差异”中分离出来。
>
> 相关文档：`controller_comaprison.md`（E1 规则控制器的完整构造）、`case_blind_0003/report.md`、`case_blind_0005/report.md`。

---

## 1. 先回答：现在的 rule-based 方法有反思吗？

**有。** 当前 E1 规则控制器不是“一次预测、一次训练”，而是确定性的 R0→R1→R2 状态机：

- R0：5 对 support 信号 → 3/5 多数投票 → R0 管线 → 训练 R0；
- 如果 R0 低置信 / NaN / support PSNR<25 → 训练 R1（候选由固定枚举产生）；
- 如果 R1 support PSNR 提升 >0.5 dB 且存在 R2 候选 → 训练 R2；
- 用 support PSNR 选择最终 round/checkpoint。

因此现有 E1 规则控制器**已经有 rule-based reflection**，只是反思由标量分数和固定规则驱动，不是 LLM 的自然语言反思。用它和 AutoIR 比较，得到的是“规则反思 vs LLM 反思”，而不是“无反思 vs 有反思”。这确实可能低估 LLM 反思的增量。

为了凸显 LLM 反思，需要增加一个**无反思的 rule-only 基线**：rule 只预测一次退化，只训练一次，不做 R1/R2。

---

## 2. 新的 2×2 设计

|  | 无反思（one-shot） | 有反思（full） |
|---|---|---|
| **规则 controller** | **E1-oneshot**：E1 R0 预测 + 训练一次 | **E1-full**：E1 R0→R1/R2（现有） |
| **LLM controller** | **AutoIR-oneshot**：AutoIR R0 预测 + 训练一次 | **AutoIR-full**：AutoIR R0→R1/R2（现有） |

这样可以把两个因素分开：

| 比较 | 含义 |
|---|---|
| `AutoIR-full − AutoIR-oneshot` | **LLM 反思的增量**（同一 controller、同一初始预测，只差 R1/R2） |
| `E1-full − E1-oneshot` | **规则反思的增量**（同一规则 controller，只差 R0→R1/R2） |
| `AutoIR-oneshot − E1-oneshot` | **无反思时的 controller 差异**（LLM 初始预测 vs 规则初始预测） |
| `AutoIR-full − E1-full` | **有反思时的 controller 差异**（现有 E1 结果） |
| `AutoIR-full − E1-oneshot` | 端到端“LLM+反思 vs 规则无反思”（用户最初的想法，但混合了两个因素） |

**建议主表用前两行**：它们才是“反思贡献”的干净对照。`AutoIR-full − E1-oneshot` 可以作为端到端卖点，但不能单独证明“LLM 反思”的价值，因为它同时包含 controller 和 reflection 差异。

---

## 3. E1-oneshot 的构造

**原则：不改预测、不重训已有模型；能复用就复用。**

1. **规则初始预测**：直接用 E1 规则控制器 `plan` 产生的 R0 管线；不跑 R1/R2。
2. **checkpoint**：
   - 优先使用 E1_fix 的 `R0_fixed`（修正噪声映射 bug 的 5 个场景：0002/0009/0012/0013/0021）；
   - 其余使用 E1 `R0`；
   - 每个场景按 support PSNR 选 best checkpoint；
   - `blind_0020` 的 E1 R0 训练 NaN（support 全 NaN）。**主结果直接把它报告为 no-reflection rule 的 terminal failure（1/24），不静默删除、不做数值替代**；Swin+SG 稳定重训仅作为可选 sensitivity，不进入主表。
3. **训练预算**：与 E1/AutoIR 相同：Direct、2 epoch、LR 5e-4、batch 16、L1、seed 42。
4. **测试**：DIV2K 100（LSDIR 暂未评估）；17 个确定性场景用与 A0/E1 相同的 hash-seeded LQ；6 个非确定性共同场景用 `E1E2_fixed_lq` 的同一份固定 LQ 重评（见 §6）。
5. **不跑**：R1/R2、候选打分、support 选择之外的任何反思。

产物：

```text
rebuttal_runs/E1_oneshot/selection.jsonl              # 23 个有效 R0（0020 NaN）
rebuttal_runs/E1_oneshot/blind_0020_R0_stable/        # 可选 sensitivity，不进入主表
rebuttal_runs/E1_oneshot/test_metrics/                # 固定 LQ 测试
rebuttal_runs/scripts/build_oneshot_selection.py
rebuttal_runs/scripts/aggregate_oneshot.py
```

---

## 4. AutoIR-oneshot 的构造

AutoIR 的初始预测就是 exp41 R0。A0 的 `support_selection.jsonl` 已保存每个场景的 `per_round.R0.selected`，直接复用：

- 24 个场景；
- R0 checkpoint 由 A0 按 support PSNR 选择；
- 不跑 R1/R2；
- 同一 LQ 协议：17 个确定性场景与 A0/E1 hash 一致；6 个共同 fixed 场景用 `E1E2_fixed_lq` 的同一份 LQ 重评。

这样 AutoIR-oneshot 与 AutoIR-full 的差异**只来自 R1/R2 反思**（同一初始 R0 预测、同一 controller、同一 checkpoint 选择规则）。

---

## 5. 结果

> 固定 LQ test 已在 23 个有效共同场景上完成（DIV2K 100 张/模型，PSNR-only；one-shot 跳过 SSIM/LPIPS 以加速）。`blind_0020` 的 E1-oneshot R0 训练 NaN，作为 **1/24 失败**单独报告，不静默删除、不做数值替代；主表是 23 个有效场景的 macro。full 行的数字也限制到同一 23 场景，保证可比。

### 5.1 固定 LQ test（DIV2K，23 个共同场景）

| 方法 | 反思 | DIV2K PSNR | DIV2K SSIM |
|---|---|---:|---:|
| E1-oneshot（rule） | 否 | 22.206 | — |
| E1-full（rule） | 是（规则） | **24.632** | 0.7213 |
| AutoIR-oneshot（LLM） | 否 | 23.989 | — |
| AutoIR-full（LLM） | 是（LLM） | **25.268** | 0.7391 |

> 24 场景的 full 数字（含 blind_0020）是 rule 24.272 / AutoIR 24.968；这里为了和 one-shot 的 23 场景对齐，排除了 blind_0020 后重算为 24.632 / 25.268。不要和 24 场景表混用。

### 5.2 配对比较（场景级 bootstrap，23 场景）

| 比较 | DIV2K ΔPSNR | 95% CI | wins | 解释 |
|---|---:|---:|---:|---|
| **E1-full − E1-oneshot** | **+2.426** | [+1.347, +3.666] | 18/23 | **规则反思增量** |
| **AutoIR-full − AutoIR-oneshot** | **+1.279** | [+0.391, +2.680] | 15/23 | **LLM 反思增量** |
| AutoIR-oneshot − E1-oneshot | +1.783 | [−1.159, +4.919] | 14/23 | 无反思 controller 差异（不显著） |
| AutoIR-full − E1-oneshot | +3.062 | [+0.301, +6.108] | 17/23 | 端到端混合效应 |
| AutoIR-full − E1-full | +0.636 | [−2.442, +3.488] | 15/23 | 有反思 controller 差异（不显著） |

### 5.3 反思增益的分布：规则广、LLM 尾部大

| Controller | 平均 Δ | 中位数 Δ | 正/零/负 | 最大增益 | 最大损失 |
|---|---:|---:|---:|---|---|
| rule | +2.43 | **+1.67** | 18/4/1 | blind_0004 +9.85、0005 +8.97、0023 +6.72 | blind_0018 −0.57 |
| AutoIR | +1.28 | **+0.14** | 15/5/3 | **blind_0014 +14.38**、0019 +3.04、0023 +2.45 | blind_0004 −0.11、0003 −0.01、0005 −0.005 |

- `AutoIR` 的 +1.28 dB 被 **blind_0014 的 +14.38 dB** 主导；去掉它后其余 22 场景平均只有约 +0.68 dB，中位数 +0.14 dB。
- `rule` 的增益更广：18/23 正增益，中位数 +1.67 dB；blind_0005（impulse）+8.97、blind_0004 +9.85。
- 这就是为什么**不能把“LLM 反思”包装成平均优越**：LLM 反思是**尾部驱动**，在个别场景能找到很大的修正；规则反思更稳定、平均更大。

### 5.4 NaN 失败案例（blind_0020）

| 方法 | 反思 | blind_0020 DIV2K PSNR | 说明 |
|---|---|---:|---|
| E1-oneshot（rule） | 否 | **NaN** | GCM R0 训练 NaN；无反思 → terminal failure，计为 1/24 |
| E1-oneshot stable fallback（可选） | 否 | 14.05（fixed-LQ） | Swin+SG 稳定重训、仍用错误 DC；只是避免 NaN，不是 R1 |
| E1-full（rule） | 是（规则） | 15.98（R1 fallback，fixed-LQ） | R0 NaN 后规则状态机切到 R1/Swin+SG，恢复 |
| AutoIR-oneshot（LLM） | 否 | 15.78（fixed-LQ） | R0 正常训练；无反思 |
| AutoIR-full（LLM） | 是（LLM） | 18.09（fixed-LQ） | 反思选中 R2 |

**这条本身就是“反思有效”的证据**：没有反思时，rule 在 blind_0020 直接 NaN；规则反思（R1 fallback）能恢复。它不是 LLM-specific，但说明 one-shot 的 23 场景 macro 必须同时报告 1/24 failure rate，不能只报 23 场景平均值。**可选 stable fallback**（Swin+SG、仍用错误 DC）fixed-LQ 只有 **14.05**，低于 AutoIR-oneshot 15.78 和 E1-full 15.98；它只是避免 NaN，不能替代 R1 反思，也再次说明错误 DC 是主因。

### 5.5 关键结论（诚实版）

1. **反思本身有效**：rule +2.43 dB、LLM +1.28 dB，两个 within-controller 对照的 95% CI 下限都 >0。
2. **LLM 反思并不比规则反思平均更强**：rule 的 +2.43 > LLM 的 +1.28；LLM 的中位数只有 +0.14，主要由 blind_0014 一个场景撑起。
3. **LLM 全系统 vs 规则无反思的端到端 +3.06 dB 是显著的（CI [+0.30,+6.11]）**，但它分解为“初始预测 +1.78（不显著）+ 反思 +1.28（显著）”；不能把这 3.06 全归因于 LLM 反思。
4. **LLM full vs rule full 只有 +0.636 dB（CI 跨 0）**：现有证据不支持“LLM 反思优于规则反思”。
5. 如果论文要凸显 LLM 反思，较稳妥的写法是**案例级**：blind_0014 的 +14.38 dB 说明 LLM 反思能在个别场景找到规则 one-shot 漏掉的大修正；但不能说它在 24 场景平均上优于规则反思。

---

## 6. 解释与 caveat

- **E1-oneshot 不是“规则方法没有反思”的完整代表**：它是对现有 E1 规则控制器的刻意消融。现有 E1-full 本来就有规则反思；本实验只是把反思关掉。
- **AutoIR-oneshot 与 E1-oneshot 的架构不完全一致**：AutoIR R0 多数是纯 Swin，E1 R0 是 Swin+SG；因此跨 controller 的 one-shot 比较包含架构差异。**反思增量应看 within-controller 的 full − oneshot**。
- **E1-oneshot 的 blind_0020**：E1 R0 原始训练 NaN，主结果按 **1/24 terminal failure** 报告；主表是 23 个有效场景的 macro，不做 0 dB 插补或静默删除。Swin+SG 稳定重训只作为可选 sensitivity；若补上，必须明确它不是原始 one-shot，重建 24 场景表也不能掩盖原 NaN 失败。
- **checkpoint 选择**：one-shot 用 best R0 support；full 用各自的选择规则；比较时不要把 checkpoint 选择差异当成反思效应。
- **指标**：one-shot 主评估为 PSNR-only；6 个 fixed-LQ 场景补了 SSIM，但其余 17 个场景没有 SSIM，所以主表 SSIM 仍显示 —。LSDIR one-shot 尚未评估，主表只报 DIV2K。
- **LQ 配对**：17/24 场景 A0/E1/one-shot 的 LQ hash 一致；7/24 非确定性场景（0003/0005/0017/0018/0020/0022/0023）原始 A0/E1 的 `lq_sha256` 不一致（8,400 行中 2,450 行 = 7×350），原因是 `noise_impulse`/`blur_glass` 使用私有 RNG，同一 GT 参数会采到不同 realization；GT 管线没有变。LLM vs rule 主表使用 `E1E2_fixed_lq` 的固定 LQ；one-shot 的 6 个共同 fixed 场景（不含 0020）用 `eval_oneshot_fixed_lq.py` 在同一份固定 LQ 上重评。其余 17 个确定性场景 hash 一致，可直接配对。
- **Support 级预检**（非 test）：AutoIR 23.734→25.058（+1.324）；rule 22.480→24.325（+1.845，23/24 口径不完全一致）。趋势与 test 一致：两个反思都有效，rule 的平均增益更大。
- **对论文的建议**：不要把“LLM 反思 > 规则反思”作为主结论；改为“反思是通用机制；LLM 反思在个别场景有大的尾部收益（blind_0014 +14.38），规则反思更稳定”。如果必须强调 LLM 反思，用 case study，不要用 macro 平均。
- **失败报告规范**：E1-oneshot 是 23/24 有效、1/24 NaN（blind_0020）；所有 one-shot 表格必须带 failure rate，不能只给有效场景均值。E1-full 的 24/24 包含 R1 fallback，应注明“R0 NaN → R1”而不是当成一次成功的 R0。

---

## 7. 复现

```bash
# 1) 构建 one-shot selections
.venv/bin/python3 rebuttal_runs/scripts/build_oneshot_selection.py

# 2) 评估（示例：3 张空闲 GPU；DIV2K-only、PSNR-only 加速）
for g in 0 1 2; do
  CUDA_VISIBLE_DEVICES=2,5,7 .venv/bin/python3 rebuttal_runs/scripts/eval_checkpoint.py test \
    --selection rebuttal_runs/E1_oneshot/selection_combined.jsonl \
    --out rebuttal_runs/E1_oneshot/test_metrics/div2k_gpu${g}.jsonl \
    --gpu-id $g --num-gpus 3 --psnr-only --datasets DIV2K
done

# 3) 汇总（23 个有效场景 + 1 个 NaN failure；stable fallback 仅可选）
.venv/bin/python3 rebuttal_runs/scripts/aggregate_oneshot.py
```
