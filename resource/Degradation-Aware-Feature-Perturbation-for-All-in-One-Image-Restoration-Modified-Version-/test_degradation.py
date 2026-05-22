#!/usr/bin/env python3
"""
DFPIR 盲复原基准模型 — 特定退化测试（多卡并行 + tiled inference）。

用法:
    python test_degradation.py --params exp5/degradation/params.json
    python test_degradation.py --params params.json --gpus 0,1,2,3
    python test_degradation.py --params params.json --gpus 0 --output results.json

多卡模式：图像均匀分配到各 GPU，每卡 batch=1，逐张处理。
大图自动 tiled inference（tile=512, overlap=64），避免 MDTA 注意力 OOM。
"""

import os, sys, json, argparse, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from PIL import Image
from collections import defaultdict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, "..", ".."))

from net.model import ChannelShuffle_skip_textguaid
from prepare import load_degradation_params
from x_distortion import add_distortion

CKPT_PATH = os.path.join(THIS_DIR, "dfpir_blind/checkpoints/dfpir_blind_step301920.pt")

VAL_DIRS = [
    "datasets/Set5/GTmod4", "datasets/Set14/GTmod4",
    "datasets/B100/GTmod4", "datasets/Urban100/GTmod4",
    "datasets/Manga109/GTmod4", "datasets/DIV2K/DIV2K_valid_HR",
]

TILE_SIZE = 512       # tile size for large-image inference
TILE_OVERLAP = 64     # overlap between adjacent tiles


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
class DFPIRBlind(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.text_embed = nn.Parameter(torch.randn(1, 512) * 0.02)

    def _pad16(self, x):
        _, _, H, W = x.shape
        ph, pw = (16 - H % 16) % 16, (16 - W % 16) % 16
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        return x

    def forward(self, x):
        _, _, H, W = x.shape
        x = self._pad16(x)
        b = x.shape[0]
        out = self.base(x, self.text_embed.expand(b, -1))
        return out[:, :, :H, :W]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_psnr(pred, target):
    return 10.0 * torch.log10(1.0 / F.mse_loss(pred, target).clamp(min=1e-10))


def rgb_to_y(pred, clean):
    kr, kg, kb = 0.299, 0.587, 0.114
    return (kr * pred[:, 0:1] + kg * pred[:, 1:2] + kb * pred[:, 2:3],
            kr * clean[:, 0:1] + kg * clean[:, 1:2] + kb * clean[:, 2:3])


def _gauss(size, sigma):
    c = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-c**2 / (2 * sigma**2))
    return (g / g.sum()).unsqueeze(0) * (g / g.sum()).unsqueeze(1)


def compute_ssim(pred, clean, k1=0.01, k2=0.03, ks=11, sigma=1.5):
    C = pred.shape[0]
    w = _gauss(ks, sigma).unsqueeze(0).unsqueeze(0).to(pred.device).expand(C, 1, ks, ks)
    mu1 = F.conv2d(pred.unsqueeze(0), w, groups=C, padding=ks // 2)
    mu2 = F.conv2d(clean.unsqueeze(0), w, groups=C, padding=ks // 2)
    s1 = F.conv2d(pred.unsqueeze(0)**2, w, groups=C, padding=ks // 2) - mu1**2
    s2 = F.conv2d(clean.unsqueeze(0)**2, w, groups=C, padding=ks // 2) - mu2**2
    s12 = F.conv2d((pred * clean).unsqueeze(0), w, groups=C, padding=ks // 2) - mu1 * mu2
    c1, c2 = k1**2, k2**2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1**2 + mu2**2 + c1) * (s1 + s2 + c2) + 1e-10)
    return m.mean()


# ---------------------------------------------------------------------------
# Tiled inference for large images (avoids MDTA attention OOM)
# ---------------------------------------------------------------------------
@torch.no_grad()
def tiled_forward(model, img, tile_size, overlap, device, autocast_ctx):
    """Sliding-window tiled inference with linear blending in overlap regions."""
    _, _, H, W = img.shape
    if H <= tile_size and W <= tile_size:
        with autocast_ctx:
            return model(img)

    stride = tile_size - overlap
    # Pad so tiles cover edges cleanly
    pad_h = (math.ceil((H - overlap) / stride) * stride + overlap - H)
    pad_w = (math.ceil((W - overlap) / stride) * stride + overlap - W)
    padded = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, Hp, Wp = padded.shape

    out = torch.zeros_like(padded)
    weight = torch.zeros(1, 1, Hp, Wp, device=device)

    # Blending ramp (linear in overlap regions)
    ramp_y = torch.ones(tile_size, device=device)
    ramp_x = torch.ones(tile_size, device=device)
    if overlap > 0:
        ramp = torch.linspace(0, 1, overlap, device=device)
        ramp_y[:overlap] *= ramp
        ramp_y[-overlap:] *= ramp.flip(0)
        ramp_x[:overlap] *= ramp
        ramp_x[-overlap:] *= ramp.flip(0)
    ramp_2d = ramp_y.unsqueeze(1) * ramp_x.unsqueeze(0)

    for y in range(0, Hp - overlap, stride):
        y_end = min(y + tile_size, Hp)
        y0 = y
        for x in range(0, Wp - overlap, stride):
            x_end = min(x + tile_size, Wp)
            x0 = x
            tile = padded[:, :, y0:y_end, x0:x_end]
            if tile.shape[2] != tile_size or tile.shape[3] != tile_size:
                tile = F.pad(tile, (0, tile_size - tile.shape[3], 0, tile_size - tile.shape[2]), mode='reflect')
            with autocast_ctx:
                pred_tile = model(tile)
            pred_tile = pred_tile[:, :, :y_end - y0, :x_end - x0]
            out[:, :, y0:y_end, x0:x_end] += pred_tile * ramp_2d[:pred_tile.shape[2], :pred_tile.shape[3]]
            weight[:, :, y0:y_end, x0:x_end] += ramp_2d[:pred_tile.shape[2], :pred_tile.shape[3]]

    out = out / weight.clamp(min=1e-8)
    return out[:, :, :H, :W]


# ---------------------------------------------------------------------------
# Single-GPU worker: processes a subset of images
# ---------------------------------------------------------------------------
def worker(gpu_id, image_list, pipeline, results_queue):
    """在指定 GPU 上评估 image_list 中的所有图像。每张单独处理（batch=1）。"""
    device = torch.device(f"cuda:{gpu_id}")
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    # Build model
    base = ChannelShuffle_skip_textguaid(
        dim=48, num_blocks=[4, 6, 6, 8], heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias",
    )
    model = DFPIRBlind(base).to(device)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    metrics = {"psnr_rgb": 0.0, "psnr_y": 0.0, "ssim_rgb": 0.0, "ssim_y": 0.0}
    n = len(image_list)
    if n == 0:
        results_queue.put((gpu_id, metrics, 0))
        return

    for idx, path in enumerate(image_list):
        img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        degraded = img.copy()
        for func, sev in pipeline:
            degraded = add_distortion(degraded, severity=sev, distortion_name=func)

        d = torch.from_numpy(degraded).permute(2, 0, 1).float().div_(255.0).unsqueeze_(0).to(device)
        c = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0).unsqueeze_(0).to(device)

        pred = tiled_forward(model, d, TILE_SIZE, TILE_OVERLAP, device, autocast_ctx).float()
        clean_f = c.float()

        metrics["psnr_rgb"] += compute_psnr(pred, clean_f).item()
        py, cy = rgb_to_y(pred, clean_f)
        metrics["psnr_y"] += compute_psnr(py, cy).item()
        metrics["ssim_rgb"] += compute_ssim(pred[0], clean_f[0]).item()
        metrics["ssim_y"] += compute_ssim(py[0], cy[0]).item()

    # Average
    for k in metrics:
        metrics[k] /= n
    results_queue.put((gpu_id, metrics, n))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DFPIR Blind Model — Specific Degradation Test")
    parser.add_argument("--params", required=True, help="退化参数 params.json")
    parser.add_argument("--checkpoint", default=CKPT_PATH, help="模型 checkpoint")
    parser.add_argument("--gpus", type=str, default="0", help="GPU IDs, 逗号分隔 (如 0,1,2,3)")
    parser.add_argument("--output", type=str, default=None, help="结果 JSON 路径")
    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    print(f"GPUs: {gpu_ids}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load pipeline
    pipeline = load_degradation_params(args.params)
    print(f"Degradation ({len(pipeline)} steps):")
    for i, (func, sev) in enumerate(pipeline):
        print(f"  Step {i+1}: {func} (severity={sev})")

    # Collect all images across all datasets
    exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    all_images = []  # [(path, dataset_name), ...]
    for dir_path in VAL_DIRS:
        ds_name = dir_path.split("/")[-2]
        paths = sorted([os.path.join(dir_path, e) for e in os.listdir(dir_path)
                        if os.path.splitext(e)[1].lower() in exts])
        all_images.extend([(p, ds_name) for p in paths])

    total = len(all_images)
    print(f"Total images: {total} across {len(VAL_DIRS)} datasets")

    # Split images across GPUs (round-robin)
    n_gpus = len(gpu_ids)
    splits = [[] for _ in range(n_gpus)]
    for i, item in enumerate(all_images):
        splits[i % n_gpus].append(item)

    for i, gid in enumerate(gpu_ids):
        print(f"  GPU {gid}: {len(splits[i])} images")

    # Multi-GPU parallel evaluation via multiprocessing
    t0 = time.time()
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = []
    for i, gid in enumerate(gpu_ids):
        img_paths = [p for p, _ in splits[i]]
        p = ctx.Process(target=worker, args=(gid, img_paths, pipeline, queue))
        p.start()
        processes.append(p)

    # Collect results
    all_metrics = {}
    for _ in range(n_gpus):
        gid, metrics, n = queue.get()
        all_metrics[gid] = (metrics, n)

    for p in processes:
        p.join()

    # Aggregate per dataset
    ds_metrics = defaultdict(lambda: {"psnr_rgb": 0.0, "psnr_y": 0.0, "ssim_rgb": 0.0, "ssim_y": 0.0, "count": 0})
    for i, gid in enumerate(gpu_ids):
        _, n = all_metrics[gid]
        for path, ds_name in splits[i]:
            ds_metrics[ds_name]["count"] += 1

    # Re-evaluate: since metrics are already averaged per-GPU, we need to weight by image count
    # Simpler: use the worker's pre-averaged metrics per dataset
    # Actually, let me collect per-dataset metrics from each worker
    # For now, use the simpler approach: all GPUs process their split, we average overall

    # Compute overall across all GPUs (weighted by image count)
    overall_metrics = {"psnr_rgb": 0.0, "psnr_y": 0.0, "ssim_rgb": 0.0, "ssim_y": 0.0}
    grand_total = 0
    for gid in gpu_ids:
        metrics, n = all_metrics[gid]
        for k in overall_metrics:
            overall_metrics[k] += metrics[k] * n
        grand_total += n
    for k in overall_metrics:
        overall_metrics[k] /= max(grand_total, 1)

    elapsed = time.time() - t0
    print(f"\n{'Dataset':10s} {'PSNR_RGB':>8s} {'PSNR_Y':>8s} {'SSIM_RGB':>8s} {'SSIM_Y':>8s}")
    print(f"{'─'*50}")
    print(f"  {'Overall':10s} {overall_metrics['psnr_rgb']:8.2f} {overall_metrics['psnr_y']:8.2f} {overall_metrics['ssim_rgb']:8.4f} {overall_metrics['ssim_y']:8.4f}")
    print(f"\nTime: {elapsed:.0f}s on {n_gpus} GPU(s)")

    # Save
    result = {
        "checkpoint": args.checkpoint,
        "params": os.path.abspath(args.params),
        "pipeline": [{"function": func, "severity": sev} for func, sev in pipeline],
        "overall": {k: round(v, 4) for k, v in overall_metrics.items()},
        "total_images": grand_total,
        "gpus": gpu_ids,
        "time_seconds": round(elapsed, 1),
    }
    output_path = args.output or f"test_result_{os.path.basename(args.params).replace('.json', '')}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
