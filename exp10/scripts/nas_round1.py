#!/usr/bin/env python3
"""
NAS Round 1: 基于退化特征的架构变异生成器

对 3 组代表性退化, 生成退化专用架构变体:
  D3: compression_jpeg(3) + blur_lens(4) → 需要空间注意力 + 频域处理
  N4: noise_poisson(3) + blur_lens(3) → 需要通道注意力 + 局部处理
  S5: blur_motion(5) + noise(1) + jpeg(1) → 需要大窗口空间注意力

每个退化生成 2-3 个架构变体, 写入独立的训练脚本。
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# === 退化特征 → 架构偏置映射 ===
DEGRADATION_ARCHITECTURE_MAP = {
    "D3": {
        "desc": "compression_jpeg(3) + blur_lens(4)",
        "features": "强梯度冲突, JPEG块效应+径向散焦",
        "arch_bias": {
            "attention": "空间注意力优先 (OCAB-like)：压缩伪影去块 + lens反卷积都需要空间上下文",
            "norm": "LayerNorm (Transformer风格, 更好处理结构化伪影)",
            "activation": "GELU (平滑, 适合结构化退化)",
            "window": "大窗口 (12-16), 覆盖JPEG的8×8块",
            "channel_mix": "SE (通道重标定, 帮助区分块效应 vs 真实边缘)",
            "depth": "中等 (2,2,2,2), 不需要太深",
        }
    },
    "N4": {
        "desc": "noise_poisson(3) + blur_lens(3)",
        "features": "强随机噪声+镜头模糊, 噪声主导",
        "arch_bias": {
            "attention": "通道注意力 (MDTA-like)：去噪主要靠通道维度的信号/噪声分离",
            "norm": "InstanceNorm (适合去噪, 移除instance-specific噪声统计)",
            "activation": "ReLU (对噪声更鲁棒)",
            "window": "小窗口 (4-8), 噪声是局部像素级问题",
            "channel_mix": "none (通道交互不重要, 噪声独立于通道)",
            "depth": "浅 (1,1,2,2), 容量少, 防过拟合噪声",
        }
    },
    "S5": {
        "desc": "blur_motion(5) + noise_gaussian_RGB(1) + compression_jpeg(1)",
        "features": "严重方向性模糊主导, 噪声和压缩很轻",
        "arch_bias": {
            "attention": "双分支 (Channel+ Spatial)：motion反卷积需要空间, 但channel分支辅助",
            "norm": "LayerNorm",
            "activation": "GELU",
            "window": "大窗口 (16), 覆盖长拖影的motion kernel",
            "channel_mix": "LayerScale (稳定深层训练)",
            "depth": "深 (3,3,3,3), motion反卷积需要更多层",
        }
    },
}

def generate_architecture_variants():
    """为每个退化生成具体架构变体代码"""
    variants = []

    # D3 variants
    variants.append({
        "id": "D3_v1_OCAB",
        "degradation": "D3",
        "desc": "SwinIR + OCAB空间注意力 + 大窗口",
        "params": {
            "EMBED_DIM": 64, "DEPTHS": "(2,2,2,2)", "NUM_HEADS": "(4,4,4,4)",
            "WINDOW_SIZE": 12, "MLP_RATIO": 2, "ACTIVATION": "gelu",
            "NORM_TYPE": "layernorm", "CHANNEL_MIX": "se",
        }
    })
    variants.append({
        "id": "D3_v2_Deep",
        "degradation": "D3",
        "desc": "更深网络 + 更大容量",
        "params": {
            "EMBED_DIM": 96, "DEPTHS": "(3,3,3,3)", "NUM_HEADS": "(6,6,6,6)",
            "WINDOW_SIZE": 8, "MLP_RATIO": 3, "ACTIVATION": "gelu",
        }
    })

    # N4 variants
    variants.append({
        "id": "N4_v1_InstanceNorm",
        "degradation": "N4",
        "desc": "InstanceNorm + ReLU + 浅层",
        "params": {
            "EMBED_DIM": 48, "DEPTHS": "(1,1,2,2)", "NUM_HEADS": "(3,3,3,3)",
            "WINDOW_SIZE": 4, "MLP_RATIO": 2, "ACTIVATION": "relu",
            "NORM_TYPE": "instancenorm",
        }
    })
    variants.append({
        "id": "N4_v2_Small",
        "degradation": "N4",
        "desc": "极小网络, 防过拟合",
        "params": {
            "EMBED_DIM": 32, "DEPTHS": "(1,1,1,1)", "NUM_HEADS": "(2,2,2,2)",
            "WINDOW_SIZE": 4, "ACTIVATION": "relu",
        }
    })

    # S5 variants
    variants.append({
        "id": "S5_v1_LargeWindow",
        "degradation": "S5",
        "desc": "大窗口(16) + 深层 + LayerScale",
        "params": {
            "EMBED_DIM": 64, "DEPTHS": "(3,3,3,3)", "NUM_HEADS": "(4,4,4,4)",
            "WINDOW_SIZE": 16, "MLP_RATIO": 2, "ACTIVATION": "gelu",
            "CHANNEL_MIX": "layerscale",
        }
    })
    variants.append({
        "id": "S5_v2_DeepWide",
        "degradation": "S5",
        "desc": "深+宽, 更多容量处理严重motion",
        "params": {
            "EMBED_DIM": 96, "DEPTHS": "(3,3,3,3)", "NUM_HEADS": "(4,4,4,4)",
            "WINDOW_SIZE": 8, "MLP_RATIO": 3, "ACTIVATION": "gelu",
        }
    })

    return variants


if __name__ == "__main__":
    variants = generate_architecture_variants()
    print(f"Generated {len(variants)} architecture variants:\n")
    for v in variants:
        cmd = (f"AR_EMBED_DIM={v['params']['EMBED_DIM']} "
               f"AR_DEPTHS={v['params'].get('DEPTHS','(2,2,2,2)')} "
               f"AR_WINDOW_SIZE={v['params'].get('WINDOW_SIZE',8)} "
               f"AR_ACTIVATION={v['params'].get('ACTIVATION','gelu')}")
        extra = " ".join([f"AR_{k.upper()}={v['params'][k]}" for k in ['NORM_TYPE','CHANNEL_MIX','MLP_RATIO'] if k in v['params']])
        print(f"  {v['id']}: {v['desc']}")
        print(f"    {cmd} {extra}")
        print()
