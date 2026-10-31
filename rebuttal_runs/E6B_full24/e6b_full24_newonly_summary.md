# E6-B full24 结果（新实验，setA/setB）

24 场景 × {setA,setB}，每组 5 张随机 `DIV2K_train` support；完整 LLM 流程；固定 GT-LQ test（DIV2K 100 + LSDIR 250）。宏平均 = 24 场景等权。

| 数据集 | setA | setB | 两组合并 mean | 场景间 SD |
|---|---:|---:|---:|---:|
| DIV2K | 27.633 | 27.540 | 27.586 | 6.813 |
| LSDIR | 25.177 | 25.178 | 25.177 | 6.494 |

## within-scenario（support-set 差异）

| 数据集 | mean SD | median SD | max SD | mean range | median range | max range |
|---|---:|---:|---:|---:|---:|---:|
| DIV2K | 0.790 | 0.258 | 6.240 | 1.118 | 0.365 | 8.825 |
| LSDIR | 0.724 | 0.142 | 6.396 | 1.023 | 0.200 | 9.046 |

setB − setA（场景配对，描述性）：
- DIV2K: -0.093 dB [-0.879,+0.929]，setB 胜 10/24
- LSDIR: +0.001 dB [-0.706,+1.002]，setB 胜 11/24

## 选点轮次

- setA: 21 R0 / 3 R1 / 0 R2
- setB: 20 R0 / 4 R1 / 0 R2

## E6-A 参考（不同 support 来源）

E6-A 使用 DIV2K_valid support（与 test 同批图），仅作 run-to-run 参考：DIV2K 27.654 / LSDIR 25.240。不能与 setA/setB 无标注合并。

## 分层（按 GT 步数）

- 1 步（n=8）：DIV2K 34.677（range 0.160）；LSDIR 31.870（range 0.267）
- 2 步（n=8）：DIV2K 27.009（range 1.942）；LSDIR 24.512（range 1.696）
- 3 步（n=8）：DIV2K 21.073（range 1.251）；LSDIR 19.149（range 1.107）