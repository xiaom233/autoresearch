# PSNR 口径（E1 / E6-A / E6-B / SIDD）

## 定义

所有主表 PSNR 均为 **RGB joint-MSE PSNR**：

```
pred, clean ∈ [0,1]^3×H×W
MSE = mean over 3 channels and all pixels of (pred - clean)^2
PSNR = 10 * log10(1 / MSE)      (MSE=0 → 100 dB)
```

不是 YCbCr 的 Y 通道 PSNR，也不是逐通道 PSNR 的算术平均。

## 代码证据

- `rebuttal_runs/scripts/eval_checkpoint.py::psnr`：`F.mse_loss(pred, target)` 直接对 RGB 三通道求 MSE。
- 逐图 JSONL 只有 `psnr_rgb` / `ssim_rgb`；**没有 `psnr_y` 字段**。
- 聚合脚本全部取 `psnr_rgb`：
  - `aggregate_e1_final.py`：`x["psnr_rgb"]`
  - `aggregate_e6a_full24.py`：`r["psnr_rgb"]`
  - `aggregate_e6llm.py`：`r["psnr_rgb"]`
  - `aggregate_e4_sidd_full1280.py` / `bootstrap_e4_sidd_full1280.py`：`psnr_rgb`
- `train.py` 同时计算 `psnr_rgb` 与 `psnr_y` 并打印，但：
  - checkpoint 选择比较的是 `psnr_rgb`（`best_checkpoint`）；
  - `val_psnr_db` 写入 `psnr_rgb`；
  - `psnr_y` 不进入任何主表。
- exp41 的 `reeval_all.py` 写 `ov['psnr_rgb']`（旧实验，按用户要求不纳入新统计）。

## 聚合方式

- 逐图 PSNR 先按 dataset 求 mean → 每场景每 rep/set 一个值 → 24 场景等权 macro。
- 不做“所有图像合并 MSE 后再算一个 PSNR”；主表是 **per-image PSNR 的均值**。
- E1 rule 与 E1 new LLM（E6-A final）都用这一口径；E6-A/E6-B/SIDD 同。

## LQ 说明

- 固定 GT-LQ：`eval_checkpoint.py test --seed-salt v1`；17/24 确定性场景的 `lq_sha256` 跨实验一致。
- 7 个私有 RNG 场景（0003/0005/0017/0018/0020/0022/0023）主评估的 LQ realization 可不同；固定 LQ 重评差异 ≤0.02 dB。
- E1 new LLM 对照使用 E1 的 fixed-LQ 值与 E6-A 的 fixed-LQ 值（7 场景），其余 17 场景主值即固定值。

## 结论

E1 的 24.272/24.968/27.654、E6-A 的 26.817/27.654、E6-B 的 30.194、SIDD 的 30.634 等均为 **RGB PSNR**，不存在 RGB/YCbCr 混用。
