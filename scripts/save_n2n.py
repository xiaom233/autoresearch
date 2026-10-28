"""Test Neighbor2Neighbor pretrained models on 24 blind challenges.
Uses gauss5-50 for gaussian/speckle noise, poisson5-50 for poisson.
Output: benchmark_results/n2n/{DIV2K|LSDIR}_gpu{N}.tar
"""
import json, os, sys, io, tarfile, glob
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from pathlib import Path

GPU_OFFSET = int(os.environ.get('AR_GPU_OFFSET', 0))
NUM_GPUS = int(os.environ.get('AR_NUM_GPUS', 8))

sys.path.insert(0, '/data/zyli/projects/autoresearch')
sys.path.insert(0, '/data/zyli/projects/autoresearch/benchmark/Neighbor2Neighbor')
from x_distortion import add_distortion
from arch_unet import UNet

device = torch.device('cuda:0')
DEG_DIR = Path('/data/zyli/projects/autoresearch/exp40/degradation_gt')
OUT_DIR = Path('/data/zyli/projects/autoresearch/benchmark_results/n2n')
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path('/data/zyli/projects/autoresearch/benchmark/Neighbor2Neighbor/pretrained_model')

# Load models
models = {}
for name, path in [('gauss', 'model_gauss5-50_b4e100r02.pth'),
                    ('poisson', 'model_poisson5-50_b4e100r02.pth')]:
    m = UNet(in_nc=3, out_nc=3, n_feature=48).to(device)
    ckpt = torch.load(str(MODEL_DIR / path), map_location=device, weights_only=True)
    m.load_state_dict(ckpt)
    m.eval()
    models[name] = m
print(f"Loaded {len(models)} N2N models")

def infer_n2n(model, img_tensor):
    """N2N inference with 32× pad + reflect."""
    _, _, H, W = img_tensor.shape
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    if pad_h or pad_w:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    with torch.no_grad():
        out = model(img_tensor)
    return out[:, :, :H, :W]

def to_jpg(t):
    if t.dim() == 4: t = t.squeeze(0)
    arr = t.clamp(0,1).mul(255).byte().permute(1,2,0).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(arr, 'RGB').save(buf, format='JPEG', quality=100, subsampling=0)
    return buf.getvalue()

def pick_model(pipeline):
    """Pick best N2N model based on noise type in pipeline."""
    for step in pipeline:
        f = step['function']
        if 'poisson' in f:
            return models['poisson']
        if 'gauss' in f or 'speckle' in f or 'impulse' in f:
            return models['gauss']
    return models['gauss']  # default

all_bids = sorted([f'blind_{i:04d}' for i in range(1, 25)])
my_bids = [b for i, b in enumerate(all_bids) if i % NUM_GPUS == GPU_OFFSET]
print(f"GPU{GPU_OFFSET}: {len(my_bids)} challenges")

for ds_name, ds_path in [('DIV2K', '/data/zyli/projects/autoresearch/datasets/DIV2K/DIV2K_valid_HR'),
                          ('LSDIR', '/data/zyli/projects/autoresearch/datasets/LSDIR/val1/HR/val')]:
    if not os.path.isdir(ds_path): continue
    img_paths = sorted(glob.glob(f'{ds_path}/*.png') + glob.glob(f'{ds_path}/*.jpg'))
    print(f"[{ds_name}] {len(img_paths)} images")
    out_tar = tarfile.open(OUT_DIR / f'{ds_name}_gpu{GPU_OFFSET}.tar', 'w')
    clean_cache = {p: np.array(Image.open(p).convert('RGB'), dtype=np.uint8) for p in img_paths}

    for bid in my_bids:
        pipeline = json.load(open(DEG_DIR / f'{bid}_params.json'))['pipeline']
        model = pick_model(pipeline)
        count = 0; print(f"  {bid} ...")
        for img_path in img_paths:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            clean = clean_cache[img_path]
            degraded = clean.copy()
            for step in pipeline: degraded = add_distortion(degraded, severity=step['severity'], distortion_name=step['function'])
            d_t = torch.from_numpy(degraded).permute(2,0,1).float().div_(255).unsqueeze(0).to(device)
            restored = infer_n2n(model, d_t)
            for tag, t in [('degraded', d_t.float()), ('restored', restored),
                           ('clean', torch.from_numpy(clean).permute(2,0,1).float().div_(255).unsqueeze(0))]:
                jpg = to_jpg(t)
                info = tarfile.TarInfo(name=f'{bid}/{img_name}.{tag}.jpg')
                info.size = len(jpg)
                out_tar.addfile(info, io.BytesIO(jpg))
            count += 1
            if count % 50 == 0: print(f"    {count}/{len(img_paths)}")
        print(f"    done ({count})")
    out_tar.close()
    print(f"[{ds_name}] Done")
print(f"GPU{GPU_OFFSET} Done")
