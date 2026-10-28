"""Evaluate DFPIR-ft checkpoints from exp37 on DIV2K/LSDIR separately.
Output: benchmark_results/dfpir_ft/{DIV2K|LSDIR}_gpu{N}.tar
Usage: AR_GPU_OFFSET=0 AR_NUM_GPUS=8 python -u save_dfpir_ft.py
"""
import json, os, sys, io, tarfile, glob
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from pathlib import Path

GPU_OFFSET = int(os.environ.get('AR_GPU_OFFSET', 0))
NUM_GPUS = int(os.environ.get('AR_NUM_GPUS', 8))

sys.path.insert(0, '/data/zyli/projects/autoresearch')
from x_distortion import add_distortion
sys.path.insert(0, '/data/zyli/projects/autoresearch/benchmark/DFPIR')
from net.model import ChannelShuffle_skip_textguaid

device = torch.device('cuda:0')
CKPT_DIR = Path('/data/zyli/projects/autoresearch/exp37/experiments')
DEG_DIR = Path('/data/zyli/projects/autoresearch/exp37/degradation_gt')
OUT_DIR = Path('/data/zyli/projects/autoresearch/benchmark_results/dfpir_ft')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Find DFPIR-ft checkpoints
ckpt_map = {}
for exp_dir in sorted(CKPT_DIR.iterdir()):
    name = exp_dir.name
    if not name.startswith('dfpir_ft_blind_'):
        continue
    bid = name.replace('dfpir_ft_', '')
    ckpt_dir = exp_dir
    pts = sorted(ckpt_dir.glob('dfpir_ft_iter*.pt'))
    if not pts:
        continue
    latest = pts[-1]
    ckpt_map[bid] = str(latest)

print(f"Found {len(ckpt_map)} DFPIR-ft checkpoints")
for bid in sorted(ckpt_map.keys()):
    print(f"  {bid}: {ckpt_map[bid]}")

def to_jpg(t):
    if t.dim() == 4: t = t.squeeze(0)
    arr = t.clamp(0,1).mul(255).byte().permute(1,2,0).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(arr, 'RGB').save(buf, format='JPEG', quality=100, subsampling=0)
    return buf.getvalue()

all_bids = sorted(ckpt_map.keys())
my_bids = [b for i, b in enumerate(all_bids) if i % NUM_GPUS == GPU_OFFSET]
print(f"GPU{GPU_OFFSET}: {len(my_bids)} challenges: {my_bids}")

for ds_name, ds_path in [('DIV2K', '/data/zyli/projects/autoresearch/datasets/DIV2K/DIV2K_valid_HR'),
                          ('LSDIR', '/data/zyli/projects/autoresearch/datasets/LSDIR/val1/HR/val')]:
    if not os.path.isdir(ds_path): continue

    img_paths = sorted(glob.glob(f'{ds_path}/*.png') + glob.glob(f'{ds_path}/*.jpg'))
    print(f"[{ds_name}] {len(img_paths)} images")

    out_path = OUT_DIR / f'{ds_name}_gpu{GPU_OFFSET}.tar'
    saved_set = set()
    if out_path.exists():
        try:
            with tarfile.open(out_path, 'r') as t:
                for info in t.getmembers():
                    parts = info.name.split('/')
                    if len(parts) >= 2:
                        saved_set.add((parts[0], parts[1].rsplit('.', 2)[0]))
            print(f"  RESUME ({len(saved_set)} saved)")
        except:
            out_path.unlink()
            saved_set = set()

    out_tar = tarfile.open(out_path, 'a' if saved_set else 'w')
    clean_cache = {}
    for p in img_paths:
        clean_cache[p] = np.array(Image.open(p).convert('RGB'), dtype=np.uint8)

    for bid in my_bids:
        ckpt_path = ckpt_map[bid]
        params_file = DEG_DIR / f'{bid}_params.json'
        if not params_file.exists(): continue
        pipeline = json.load(open(params_file))['pipeline']

        try:
            model = ChannelShuffle_skip_textguaid(device=device).to(device)
            ck = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ck['state_dict'] if 'state_dict' in ck else ck, strict=False)
            model.eval()
        except Exception as e:
            print(f"  {bid}: SKIP model error: {e}")
            continue

        text_code = torch.randn(1, 512).to(device)
        count = 0
        print(f"  {bid} ...")

        with torch.no_grad():
            for img_path in img_paths:
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                if (bid, img_name) in saved_set:
                    count += 1
                    continue

                clean = clean_cache[img_path]
                degraded = clean.copy()
                for step in pipeline:
                    degraded = add_distortion(degraded, severity=step['severity'],
                                               distortion_name=step['function'])

                d_t = torch.from_numpy(degraded).permute(2,0,1).float().div_(255).unsqueeze(0).to(device)
                H, W = d_t.shape[2], d_t.shape[3]
                pad = 16
                ph, pw = (pad - H%pad)%pad, (pad - W%pad)%pad
                dpad = F.pad(d_t, (0, pw, 0, ph), mode='reflect') if ph or pw else d_t

                # DFPIR: tiled inference for large images
                if max(H, W) > 1024:
                    T = 1024
                    pred = torch.zeros(1, 3, H, W, device=device, dtype=torch.float32)
                    for y in range(0, H, T):
                        for x in range(0, W, T):
                            y2, x2 = min(y+T, H), min(x+T, W)
                            tile = dpad[:, :, y:y2, x:x2]
                            th, tw = y2-y, x2-x
                            if th < T or tw < T:
                                tile = F.pad(tile, (0, T-tw, 0, T-th))
                            ph2 = (16 - tile.shape[2]%16)%16
                            pw2 = (16 - tile.shape[3]%16)%16
                            if ph2 or pw2:
                                tile = F.pad(tile, (0, pw2, 0, ph2), mode='reflect')
                            out = model(tile, text_code.expand(tile.shape[0], -1))
                            if isinstance(out, (list,tuple)): out = out[0]
                            pred[:, :, y:y2, x:x2] = out[:, :, :th, :tw]
                    restored = pred
                else:
                    out = model(dpad, text_code.expand(dpad.shape[0], -1))
                    if isinstance(out, (list,tuple)): out = out[0]
                    restored = out[:, :, :H, :W]

                for tag, t in [('degraded', d_t[:,:,:H,:W]),
                               ('restored', restored),
                               ('clean', torch.from_numpy(clean).permute(2,0,1).float().div_(255).unsqueeze(0).to(device))]:
                    jpg = to_jpg(t)
                    info = tarfile.TarInfo(name=f'{bid}/{img_name}.{tag}.jpg')
                    info.size = len(jpg)
                    out_tar.addfile(info, io.BytesIO(jpg))

                count += 1
                if count % 50 == 0:
                    print(f"    {count}/{len(img_paths)}")

        print(f"    done ({count} imgs)")

    out_tar.close()
    print(f"[{ds_name}] GPU{GPU_OFFSET}: Done")

print(f"GPU{GPU_OFFSET} Done")
