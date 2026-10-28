"""DIP (Deep Image Prior) benchmark: per-challenge step calibration on 5 images,
then fixed-step inference on DIV2K+LSDIR. Fully GPU-bound, no CPU resize.
Usage: AR_GPU_OFFSET=0 AR_NUM_GPUS=8 ./doubledip/bin/python scripts/benchmark_dip.py
"""
import json, os, sys, io, tarfile, glob, time, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from pathlib import Path

GPU_OFFSET = int(os.environ.get('AR_GPU_OFFSET', 0))
NUM_GPUS = int(os.environ.get('AR_NUM_GPUS', 8))

sys.path.insert(0, '/tmp/DoubleDIP')
sys.path.insert(0, '/data/zyli/projects/autoresearch')
from net import skip
from utils.image_io import np_to_torch, torch_to_np
from net.noise import get_noise
from x_distortion import add_distortion

device = torch.device('cuda:0')
torch.backends.cudnn.benchmark = True
DTYPE = torch.cuda.FloatTensor
INPUT_DEPTH = 8
LEARNING_RATE = 0.001
PATIENCE = 50
MAX_ITER = 4000

CHALLENGE_DIR = Path('/data/zyli/projects/autoresearch/exp40/challenges/phase4')
OUT_DIR = Path('/data/zyli/projects/autoresearch/benchmark_results/dip')
OUT_DIR.mkdir(parents=True, exist_ok=True)
CALIB_DIR = Path('/data/zyli/projects/autoresearch/benchmark_results/dip_calib')
CALIB_DIR.mkdir(parents=True, exist_ok=True)


def to_jpg(t):
    if t.dim() == 4: t = t.squeeze(0)
    arr = t.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(arr, 'RGB').save(buf, format='JPEG', quality=100, subsampling=0)
    return buf.getvalue()


def dip_restore(model, noise, img_tensor):
    with torch.no_grad():
        out = model(noise)
    return out


def build_net(h, w):
    return skip(INPUT_DEPTH, 3,
        num_channels_down=[8, 16, 32, 64, 128],
        num_channels_up=[8, 16, 32, 64, 128],
        num_channels_skip=[0, 0, 0, 4, 4],
        upsample_mode='bilinear',
        need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').type(DTYPE)


def calibrate_step(degraded_paths, clean_paths):
    """Find optimal DIP steps using 5 training pairs (GT available)."""
    best_steps = []
    for d_path, c_path in zip(degraded_paths, clean_paths):
        clean = np.array(Image.open(c_path).convert('RGB'), dtype=np.float32) / 255.0
        degraded = np.array(Image.open(d_path).convert('RGB'), dtype=np.float32) / 255.0
        original_chw = degraded.transpose(2, 0, 1).copy()
        H, W = original_chw.shape[1], original_chw.shape[2]
        ph = (32 - H % 32) % 32; pw = (32 - W % 32) % 32
        padded = np.pad(original_chw, ((0, 0), (0, ph), (0, pw)), mode='reflect')
        clean_padded = np.pad(clean.transpose(2, 0, 1), ((0, 0), (0, ph), (0, pw)), mode='reflect')

        image_t = np_to_torch(padded).type(DTYPE)
        clean_t = np_to_torch(clean_padded).type(DTYPE)
        noise = get_noise(INPUT_DEPTH, 'noise', (padded.shape[1], padded.shape[2]),
                          var=1 / 10.).type(DTYPE).detach()

        model = build_net(padded.shape[1], padded.shape[2])
        mse = nn.MSELoss().cuda()
        opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_psnr = 0; best_step = 0; no_improve = 0
        for step in range(MAX_ITER):
            opt.zero_grad()
            out = model(noise)
            loss = mse(out, image_t)
            loss.backward()
            opt.step()
            p = 10 * torch.log10(1.0 / (out - clean_t).pow(2).mean()).item()
            if p > best_psnr:
                best_psnr = p; best_step = step; no_improve = 0
            else:
                no_improve += 1
            if no_improve >= PATIENCE:
                break
        best_steps.append(best_step)
        del model, opt, image_t, clean_t, noise

    return int(np.median(best_steps)), best_steps


all_bids = sorted([f'blind_{i:04d}' for i in range(1, 25)])
my_bids = [b for i, b in enumerate(all_bids) if i % NUM_GPUS == GPU_OFFSET]
print(f"GPU{GPU_OFFSET}: {len(my_bids)} challenges")

# ---- Phase 1: Calibration ----
print("Phase 1: Calibrating optimal steps...")
step_map = {}
for bid in my_bids:
    ch_dir = CHALLENGE_DIR / bid
    degraded_paths = sorted([str(p) for p in ch_dir.glob('degraded_?.png')])
    clean_paths = sorted([str(p) for p in ch_dir.glob('clean_?.png')])
    opt_step, all_steps = calibrate_step(degraded_paths, clean_paths)
    step_map[bid] = opt_step
    print(f"  {bid}: steps={all_steps} → median={opt_step}")

# Save calibration
json.dump(step_map, open(CALIB_DIR / f'steps_gpu{GPU_OFFSET}.json', 'w'), indent=2)

# ---- Phase 2: Inference on DIV2K+LSDIR ----
print("\nPhase 2: Inference...")
for ds_name, ds_path in [('DIV2K', '/data/zyli/projects/autoresearch/datasets/DIV2K/DIV2K_valid_HR'),
                          ('LSDIR', '/data/zyli/projects/autoresearch/datasets/LSDIR/val1/HR/val')]:
    if not os.path.isdir(ds_path): continue
    img_paths = sorted(glob.glob(f'{ds_path}/*.png') + glob.glob(f'{ds_path}/*.jpg'))
    clean_cache = {p: np.array(Image.open(p).convert('RGB'), dtype=np.uint8) for p in img_paths}

    for bid in my_bids:
        pipeline = json.load(open(f'/data/zyli/projects/autoresearch/exp40/degradation_gt/{bid}_params.json'))['pipeline']
        n_steps = step_map[bid]
        print(f"  {bid} ({n_steps} steps)...")

        out_tar = tarfile.open(OUT_DIR / f'{ds_name}_gpu{GPU_OFFSET}.tar', 'a')
        count = 0
        for img_path in img_paths:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            clean = clean_cache[img_path]
            degraded = clean.copy()
            for step in pipeline:
                degraded = add_distortion(degraded, severity=step['severity'],
                                          distortion_name=step['function'])

            # DIP restore
            original_chw = (degraded.astype(np.float32) / 255.0).transpose(2, 0, 1)
            H, W = original_chw.shape[1], original_chw.shape[2]
            ph = (32 - H % 32) % 32; pw = (32 - W % 32) % 32
            padded = np.pad(original_chw, ((0, 0), (0, ph), (0, pw)), mode='reflect')
            image_t = np_to_torch(padded).type(DTYPE)
            noise = get_noise(INPUT_DEPTH, 'noise', (padded.shape[1], padded.shape[2]),
                              var=1 / 10.).type(DTYPE).detach()

            model = build_net(padded.shape[1], padded.shape[2])
            mse = nn.MSELoss().cuda()
            opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

            for _ in range(n_steps):
                opt.zero_grad()
                out = model(noise)
                loss = mse(out, image_t)
                loss.backward()
                opt.step()

            with torch.no_grad():
                restored = model(noise)[:, :, :H, :W]

            d_t = image_t[:, :, :H, :W]
            for tag, t in [('degraded', d_t.float()), ('restored', restored.float()),
                           ('clean', torch.from_numpy(clean).permute(2, 0, 1).float().div_(255).unsqueeze(0))]:
                jpg = to_jpg(t)
                info = tarfile.TarInfo(name=f'{bid}/{img_name}.{tag}.jpg')
                info.size = len(jpg)
                out_tar.addfile(info, io.BytesIO(jpg))
            count += 1
            del model, opt, image_t, noise
            if count % 10 == 0: print(f"    {count}/{len(img_paths)}")

        out_tar.close()
        print(f"    done ({count})")

print(f"\nGPU{GPU_OFFSET} Done")
