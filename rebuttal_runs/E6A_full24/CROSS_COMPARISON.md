# E1 / E6-A / E6-B 交叉对比（2026-10-30）

## 1. Test 协议核验

- 共同 test：DIV2K_valid_HR 100 + LSDIR val1/HR 250；E1/E6A/E6B 的逐图 `image_hash` 一致。
- `lq_sha256`：
  - 17/24 确定性场景：E1-rule / A0-AutoIR / E6-A / E6-B 完全一致（0 mismatch）。
  - 7 个私有 RNG 场景（0003/0005/0017/0018/0020/0022/0023）：主评估进程间 LQ realization 不同（7×350 行）；固定 LQ 重评差异 ≤0.02 dB（E1 `fixed_lq_revalidation`；E6-A fixed 敏感性 mean ≤0.002、max 0.018）。
  - E6-B 的 6 场景中只有 0017 属于这 7 个；其余 LQ 一致。
- 结论：test 图与指标口径相同，LQ 差异 ≤0.02 dB，**不能解释 2–8 dB 的差距**。

## 2. 同一 6 场景交叉表（DIV2K，dB）

| 场景 | GT pipeline | E1 AutoIR | E1 rule | E6-A R0 | E6-A final | E6-B set1 | set2 | set3 | E6-B mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| blind_0006 | jpeg:1 | 31.78 | 31.47 | 31.63 | 31.63 | 31.59 | 31.62 | 31.60 | **31.61** |
| blind_0008 | saturate_weaken_HSV:4 | 43.18 | 22.65 | 42.03 | 42.03 | 39.99 | 42.94 | 42.10 | **41.67** |
| blind_0010 | blur_gaussian:1 + darken_gamma_HSV:5 | 27.04 | 16.47 | 25.86 | 31.42 | 31.59 | 31.62 | 31.15 | **31.46** |
| blind_0013 | speckle:3 + jpeg:2 | 27.46 | 27.00 | 28.67 | 28.67 | 29.21 | 29.20 | 29.09 | **29.16** |
| blind_0017 | impulse:5 + blur_gaussian:3 + jpeg:4 | 23.52 | 23.41 | 23.25 | 23.25 | 23.50 | 24.19 | 23.50 | **23.73** |
| blind_0024 | gaussian_RGB:4 + blur_gaussian:3 + sat_HSV:2 | 15.25 | 10.99 | 18.45 | 23.16 | 24.37 | 21.44 | 24.78 | **23.53** |
| **6 场景均值** | | **28.04** | **21.99** | **28.31** | **30.03** | | | | **30.19** |

- **E6-A final（30.03）≈ E6-B LLM（30.19）**：同一批场景、几乎同一 test/LQ，结果一致。
- E6-A R0-only（28.31）较低，是因为 0010/0024 还没有 R1 修复；final 在 0010/0024 走 R1 后升到 31.42/23.16。
- E1 rule 低（21.99）：0010（16.47）和 0024（10.99）出现灾难性 DC 错误；0011/0015/0021 等类似错误也拉低 24 场景 macro。

## 3. 24 场景汇总（DIV2K）

| 方法 | macro | 相对 E6-A final | 逐场景中位差 | GT 整链命中 |
|---|---:|---:|---:|---:|
| E1 AutoIR | 24.968 | −2.69 | −0.19 | — |
| E1 rule | 24.272 | −3.38 | −1.31 | 5/24 场景 |
| E6-A R0 best | 26.817 | −0.84 | — | rep2 15/24 |
| E6-A final | **27.654** | — | — | **29/48 selected runs** |
| E6-B LLM（6 场景） | 30.194 | n/a（6 场景） | n/a | **13/18 selected runs** |

- E6-A final vs AutoIR：mean +2.69，**median +0.19**；vs rule：mean +3.38，median +1.31。
- 增益仍集中在规则/AutoIR 的灾难场景（0008 +19.4、0010 +9.4、0015 +7.8、0024 +7.5、0009 +5.7、0021 +5.3 等）；多数场景接近。

## 4. E1 的 “LLM/AutoIR” 是旧链，不是新跑的 Agent

- E1 的 `AutoIR` 指标复用 **A0/exp41 R0–R2 链**（`A0/test_metrics/`）；A0 `support_selection.jsonl` 的 `exp_meta.timestamp` 为 **2026-06-30**，即预测/反思/训练链来自六月至七月。E1 在十月底只是**重新评测**，没有重跑 Agent。
- A0/AutoIR 的 DC 整链命中 **7/24（29.2%）**，与 E5a 的 exp41 整链 Spec 29.17% 一致；E6-A 新 Agent rep2 命中 **15/24（62.5%）**，final selected runs 29/48。
- E1 AutoIR 24.968 vs E6-A final 27.654：mean **+2.69 dB**，**median +0.19 dB**；差值集中在旧链的灾难 DC：
  - 0005：AutoIR `brightness+impulse:3` vs GT `impulse:5` → **21.92 vs 44.31（+22.4）**
  - 0009：AutoIR `gaussian:5+brightness:2+sat:5` vs GT `poisson:2+satYCrCb:3` → **11.27 vs 23.53（+12.3）**
  - 0024：AutoIR `gamma:3+blur:5` vs GT `gaussian:4+blur:3+sat:2` → **15.25 vs 23.16（+7.9）**
  - 0015：AutoIR `gamma:3` vs GT `motion:1+shift:3` → **21.47 vs 29.36（+7.8）**
  - 0021：AutoIR 三链错 vs GT → **17.61 vs 24.21（+6.6）**
- 其余场景大多接近：0008 AutoIR 43.18 > E6-A 42.03；0004 34.47 > 29.94；0002/0006/0007/0014 基本持平。
- 因此 **E1 的 24.968 是“旧 LLM 链”的数字；E6-A/E6-B 是十月新 Agent 链**。两者不是同一 agent 版本，不能把 24.968 当作 E6-A/E6-B 的同版本对照。

## 5. 为什么 rule-based 特别低

1. **DC 错误**：E1 rule 最终 pipeline 整链命中仅 5/24；同一 6 场景 0/6。典型：
   - 0008 GT `saturate_weaken_HSV:4`，规则选错 → 22.65 vs E6-A 42.03（−19.4）。
   - 0010 GT `blur_gaussian:1+darken_gamma_HSV:5`，规则选错 → 16.47 vs E6-B 31.46。
   - 0024 GT `gaussian_RGB:4+blur_gaussian:3+sat_HSV:2`，规则选错 → 10.99 vs E6-B 23.53。
2. **AutoIR（A0/exp41）DC 也不如新 Agent**：0010 27.04 vs 31.46、0024 15.25 vs 23.53；所以 24 场景 macro 只有 24.97。
3. **不是 test/LQ/时间问题**：协议核验一致；LQ 差异 ≤0.02 dB；三套实验都是最近几天跑的，但 DC 来源、策略、反思轮次不同。
4. **支持集来源不是决定因素**：E1/E6-A 用 exp41 support（DIV2K_valid），E6-B 用 DIV2K_train；E6-B 仍然更高，E6-A final 与 E6-B 在 6 场景上一致。

## 5. 结论

- “27 dB 对不上 rule/LLM” 的直觉来自**把 24 场景 macro（27.65）与 6 个高 PSNR 场景的 macro（30.19）直接比较**；在同一 6 场景上，E6-A final 30.03 与 E6-B 30.19 一致。
- rule-based/AutoIR 低是**DC 错误**（尤其 0008/0010/0024 等），不是协议、时间或 support-test 重叠。
- 24 场景 E6-A final 27.65 是“多数场景接近基线 + 少数场景 DC 修复”的均值，不是每个场景都 27+；median 相对 AutoIR 仅 +0.19 dB。
