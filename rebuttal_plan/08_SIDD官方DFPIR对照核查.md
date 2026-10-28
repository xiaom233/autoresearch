# SIDD官方DFPIR对照：重评后补入W5

2026-10-26。本次核对发现了影响比较有效性的推理设置问题。

## 已有数值

- AutoIR R0：30.532761 dB；1120测试块、35个support之外的场景。
- 官方5D checkpoint在当前评估器下：23.291126 dB；记录说明text_code为随机向量，seed42。seed43/44为23.064/23.461。
- 项目DFPIR-blinded：31.218091 dB。该模型非官方发布权重，不能替代official release对照。
- 当前23.291不能表述为已按官方推荐接口评测，也不能据此写“AutoIR比DFPIR official高7.242dB，证明泛化更好”。

## 官方代码核对

[官方test_5D_DFPIR.py](https://github.com/TxpHome/DFPIR/blob/main/test_5D_DFPIR.py)加载CLIP ViT-B/32（第38行），定义任务prompt（第43–45行），通过clip.tokenize与encode_text构造text_code（第104–109行）传给恢复模型。现有报告中的随机randn(1,512)不符合该流程。改变随机seed不能验证官方text conditioning正确性。

## 重评设置

1. 保持原官方5D checkpoint、SIDD同一1120测试块与指标口径。
2. 使用官方CLIP ViT-B/32 text encoder。原官方脚本没有给出SIDD协议，需明确这是将官方模型用于SIDD的新增评测。
3. 从官方三种denoising prompt（sigma15/25/50）中仅根据五张support的PSNR选一个，固定后评估test；不按test选prompt、不更新权重。明确披露该support选择步骤。也可预先固定一个prompt，但不能以测试结果挑选。
4. 记录prompt、CLIP版本、checkpoint哈希、预处理与输出指标。报告PSNR/SSIM/LPIPS；保留项目blinded为另列参考，不混同官方模型。
5. 将重评PSNR补入05的YRJq W5；若该比较支持优势，给出差值和场景配对CI，再同步bAkj W2/osqx W5。

## 回复定位

SIDD回答真实噪声适应能力；UHDM回答库外退化行为。SIDD单一数据集不能验证任意未知算子。泛化论据应是“原算子库不变，五对support驱动的开发流程用于真实噪声”，官方DFPIR作为重要部署模型对照。
