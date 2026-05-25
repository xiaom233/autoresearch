#!/usr/bin/env python3
"""级联多专家评估：加载 N 个专家 checkpoint，顺序推理，计算 PSNR。

用法:
  python cascaded_eval.py \
    --ckpts ckpt1.pt,ckpt2.pt \
    --params exp9/degradation/D1_dual.json \
    --embed-dim 48
"""

import os, sys, json, math, argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from train import RestoreNet, ValDataset, compute_psnr, compute_ssim, rgb_to_y


class CascadedEval(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.experts = nn.ModuleList(experts)

    def forward(self, x):
        for expert in self.experts:
            x = expert(x)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", required=True, help="逗号分隔的 checkpoint 路径")
    parser.add_argument("--params", required=True, help="验证退化 params.json")
    parser.add_argument("--embed-dim", type=int, default=48)
    parser.add_argument("--depths", default="2,2,2,2")
    parser.add_argument("--num-heads", default="4,4,4,4")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=2)
    parser.add_argument("--output", default=None, help="输出 JSON 路径")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depths = tuple(int(x) for x in args.depths.split(","))
    num_heads = tuple(int(x) for x in args.num_heads.split(","))

    ckpt_paths = [p.strip() for p in args.ckpts.split(",")]
    experts = []
    for ckpt_path in ckpt_paths:
        expert = RestoreNet(in_ch=3, embed_dim=args.embed_dim, depths=depths,
                            num_heads=num_heads, window_size=args.window_size,
                            mlp_ratio=args.mlp_ratio)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        expert.load_state_dict(ckpt["model"])
        expert.to(device)
        expert.eval()
        experts.append(expert)
        print(f"Loaded: {os.path.basename(ckpt_path)}")

    model = CascadedEval(experts)
    model.eval()

    val_dirs = [
        "datasets/Set5/GTmod4", "datasets/Set14/GTmod4",
        "datasets/B100/GTmod4", "datasets/Urban100/GTmod4",
        "datasets/Manga109/GTmod4", "datasets/DIV2K/DIV2K_valid_HR",
    ]

    per_set = {}
    overall = {"psnr_rgb": 0.0, "psnr_y": 0.0, "ssim_rgb": 0.0, "ssim_y": 0.0}
    total_n = 0

    with torch.no_grad():
        for val_dir in val_dirs:
            ds_name = os.path.basename(os.path.dirname(val_dir)) or val_dir.split("/")[-2]
            val_ds = ValDataset([val_dir], count=0, params_path=args.params)
            m = {"psnr_rgb": 0.0, "psnr_y": 0.0, "ssim_rgb": 0.0, "ssim_y": 0.0}
            n = len(val_ds)
            for i in range(n):
                degraded, clean = val_ds[i]
                degraded = degraded.unsqueeze(0).to(device)
                clean = clean.to(device)

                # 级联推理
                pred = model(degraded)

                pred_f = pred.float()
                clean_f = clean.float()
                m["psnr_rgb"] += compute_psnr(pred_f, clean_f)
                pred_y, clean_y = rgb_to_y(pred_f, clean_f)
                m["psnr_y"] += compute_psnr(pred_y, clean_y)
                m["ssim_rgb"] += compute_ssim(pred_f[0], clean_f[0])
                m["ssim_y"] += compute_ssim(pred_y[0], clean_y[0])

            for k in m:
                m[k] /= n
                overall[k] += m[k] * n
            total_n += n
            per_set[ds_name] = m
            print(f"  {ds_name:12s} PSNR_RGB={m['psnr_rgb']:.2f} PSNR_Y={m['psnr_y']:.2f} "
                  f"SSIM_RGB={m['ssim_rgb']:.4f}")

    for k in overall:
        overall[k] /= total_n

    print(f"\n{'Overall':12s} PSNR_RGB={overall['psnr_rgb']:.2f} PSNR_Y={overall['psnr_y']:.2f} "
          f"SSIM_RGB={overall['ssim_rgb']:.4f}")
    print(f"val_psnr_db: {overall['psnr_rgb']:.2f}")

    if args.output:
        result = {
            "ckpts": ckpt_paths,
            "params": args.params,
            "per_dataset": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in per_set.items()},
            "overall": {k: float(v) for k, v in overall.items()},
            "summary": {"val_psnr_db": overall["psnr_rgb"]},
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
