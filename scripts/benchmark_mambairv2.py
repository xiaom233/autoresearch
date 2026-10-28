"""MambaIRv2 ColorDN benchmark. Same protocol as exp40.
Global image preload + DataLoader for parallel degradation. Tile if OOM.
Usage: AR_GPU_OFFSET=0 AR_NUM_GPUS=8 python benchmark_mambairv2.py
"""
import json, os, sys, io, tarfile, glob, time
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

GPU_OFFSET = int(os.environ.get('AR_GPU_OFFSET', 0))
NUM_GPUS = int(os.environ.get('AR_NUM_GPUS', 8))

sys.path.insert(0, '/data/zyli/projects/autoresearch')
sys.path.insert(0, '/data/zyli/projects/autoresearch/benchmark/MambaIR-main')
from x_distortion import add_distortion
from basicsr.archs.mambairv2_arch import MambaIRv2

device = torch.device('cuda:0')
DEG_DIR = Path('/data/zyli/projects/autoresearch/exp40/degradation_gt')
OUT_DIR = Path('/data/zyli/projects/autoresearch/benchmark_results/mambairv2')
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIG = dict(
    upscale=1, in_chans=3, img_size=128, img_range=1.,
    embed_dim=174, d_state=16, depths=[6,6,6,6,6,6], num_heads=[6,6,6,6,6,6],
    window_size=16, inner_rank=64, num_tokens=128,
    convffn_kernel_size=5, mlp_ratio=2,
)
MEAN = (0.4488, 0.4371, 0.4040)
CKPT_PATH = 'benchmark/MambaIR-main/experiments/pretrained_models/mambairv2_ColorDN_15.pth'
TILE = 1024
TILE_OVERLAP = 64

print("Loading MambaIRv2 ColorDN...")
model = MambaIRv2(**MODEL_CONFIG).to(device)
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
model.load_state_dict(ckpt['params'], strict=True)
model.eval()
print(f"  {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")


def to_jpg(t):
    if t.dim() == 4: t = t.squeeze(0)
    arr = t.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(arr, 'RGB').save(buf, format='JPEG', quality=100, subsampling=0)
    return buf.getvalue()


@torch.no_grad()
def mambair_tiled(x):
    """Tiled inference for large images."""
    _, _, H, W = x.shape
    ws = 16
    ph = (ws - H % ws) % ws
    pw = (ws - W % ws) % ws
    x_pad = F.pad(x, (0, pw, 0, ph), mode='reflect') if ph or pw else x

    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    xn = x_pad - mean
    pH, pW = xn.shape[2], xn.shape[3]

    # If small enough, single pass
    if pH <= TILE and pW <= TILE:
        out = model(xn) + mean
        return out[:, :, :H, :W]

    # Tiled inference
    out = torch.zeros_like(xn)
    weight = torch.zeros(1, 1, pH, pW, device=device)

    for y in range(0, pH, TILE - TILE_OVERLAP):
        for x0 in range(0, pW, TILE - TILE_OVERLAP):
            ye = min(y + TILE, pH)
            xe = min(x0 + TILE, pW)
            tile = xn[:, :, y:ye, x0:xe]
            tile_out = model(tile)
            out[:, :, y:ye, x0:xe] += tile_out
            weight[:, :, y:ye, x0:xe] += 1.0

    out = out / weight + mean
    return out[:, :, :H, :W]


all_bids = sorted([f'blind_{i:04d}' for i in range(1, 25)])
my_bids = [b for i, b in enumerate(all_bids) if i % NUM_GPUS == GPU_OFFSET]
print(f"GPU{GPU_OFFSET}: {len(my_bids)} challenges")

for ds_name, ds_path in [('DIV2K', '/data/zyli/projects/autoresearch/datasets/DIV2K/DIV2K_valid_HR'),
                          ('LSDIR', '/data/zyli/projects/autoresearch/datasets/LSDIR/val1/HR/val')]:
    if not os.path.isdir(ds_path): continue
    img_paths = sorted(glob.glob(f'{ds_path}/*.png') + glob.glob(f'{ds_path}/*.jpg'))
    print(f"[{ds_name}] {len(img_paths)} images")

    # Preload clean images once per dataset
    clean_cache = {p: np.array(Image.open(p).convert('RGB'), dtype=np.uint8) for p in img_paths}
    out_tar = tarfile.open(OUT_DIR / f'{ds_name}_gpu{GPU_OFFSET}.tar', 'w')

    for bid in my_bids:
        pipeline = json.load(open(DEG_DIR / f'{bid}_params.json'))['pipeline']
        print(f"  {bid} ...")

        class DegradeDataset(Dataset):
            def __init__(self, paths, pipeline, cache):
                self.items = [(p, os.path.splitext(os.path.basename(p))[0]) for p in paths]
                self.pipeline = pipeline
                self.cache = cache
            def __len__(self): return len(self.items)
            def __getitem__(self, idx):
                p, name = self.items[idx]
                clean = self.cache[p].copy()
                degraded = clean.copy()
                for step in self.pipeline:
                    degraded = add_distortion(degraded, severity=step['severity'],
                                               distortion_name=step['function'])
                return (name,
                        torch.from_numpy(degraded).permute(2, 0, 1).float().div_(255),
                        torch.from_numpy(clean).permute(2, 0, 1).float().div_(255))

        ds = DegradeDataset(img_paths, pipeline, clean_cache)
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
        count = 0
        for img_name, degraded, clean in loader:
            img_name = img_name[0] if isinstance(img_name, list) else img_name
            degraded = degraded.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            restored = mambair_tiled(degraded)
            for tag, t in [('degraded', degraded[:, :, :restored.shape[2], :restored.shape[3]]),
                           ('restored', restored), ('clean', clean)]:
                jpg = to_jpg(t)
                info = tarfile.TarInfo(name=f'{bid}/{img_name}.{tag}.jpg')
                info.size = len(jpg)
                out_tar.addfile(info, io.BytesIO(jpg))
            count += 1
            if count % 50 == 0:
                print(f"    {count}/{len(img_paths)}")
        print(f"    done ({count})")
    out_tar.close()
    print(f"[{ds_name}] Done")

print(f"\nGPU{GPU_OFFSET} Done")
