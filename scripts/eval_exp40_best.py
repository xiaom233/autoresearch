"""Evaluate exp40 Best(R0,R1) checkpoints on DIV2K/LSDIR with PSNR/SSIM/LPIPS.
Output: benchmark_results/exp40_best/{DIV2K|LSDIR}_gpu{N}.json
Usage: AR_GPU_OFFSET=0 AR_NUM_GPUS=8 python -u eval_exp40_best.py
"""
import json, os, sys, io, glob, csv
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from pathlib import Path

GPU_OFFSET = int(os.environ.get('AR_GPU_OFFSET', 0))
NUM_GPUS = int(os.environ.get('AR_NUM_GPUS', 8))

sys.path.insert(0, '/data/zyli/projects/autoresearch')
from x_distortion import add_distortion
from train import RestoreNet

device = torch.device('cuda:0')
EXP_DIR = Path('/data/zyli/projects/autoresearch/exp40/experiments')
DEG_DIR = Path('/data/zyli/projects/autoresearch/exp40/degradation_gt')
OUT_DIR = Path('/data/zyli/projects/autoresearch/benchmark_results/exp40_best')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Best(R0,R1) from exp40/summarize/final_performance.md
BEST_CHOICES = {
    'blind_0001': 'r1','blind_0002': 'r1','blind_0003': 'r1',
    'blind_0004': 'r0','blind_0005': 'r1','blind_0006': 'r0',
    'blind_0007': 'r0','blind_0009': 'r1','blind_0010': 'r0',
    'blind_0011': 'r0','blind_0012': 'r0','blind_0013': 'r1',
    'blind_0014': 'r1','blind_0015': 'r0','blind_0016': 'r1',
    'blind_0017': 'r1','blind_0018': 'r1','blind_0019': 'r1',
    'blind_0020': 'r1','blind_0021': 'r1','blind_0022': 'r1',
    'blind_0023': 'r0','blind_0024': 'r1',
}

import lpips
lpips_fn = lpips.LPIPS(net='vgg').to(device)

def ssim_val(a, b):
    C1, C2 = 0.01**2, 0.03**2
    mu1, mu2 = a.mean(), b.mean()
    s1 = ((a-mu1)**2).mean(); s2 = ((b-mu2)**2).mean()
    s12 = ((a-mu1)*(b-mu2)).mean()
    return float(((2*mu1*mu2+C1)*(2*s12+C2))/((mu1**2+mu2**2+C1)*(s1+s2+C2)))

# Find checkpoints (sorted by step number)
ckpt_map = {}
for exp_dir in sorted(EXP_DIR.iterdir()):
    name = exp_dir.name
    ckpt_dir = exp_dir / 'checkpoints'
    if not ckpt_dir.is_dir(): continue
    pts = sorted(ckpt_dir.glob('*.pt'), key=lambda p: int(p.stem.split('step')[-1]))
    if not pts: continue
    latest = pts[-1]
    for b in BEST_CHOICES:
        if b in name:
            if name.startswith('r1_') or '_r1' in name:
                ckpt_map.setdefault(b, {})['r1'] = str(latest)
            elif 'r1' not in name and 'qwen' not in name and 'strategy' not in name and 'perstep' not in name and 'v2' not in name:
                ckpt_map.setdefault(b, {})['r0'] = str(latest)
            break

# Build model from checkpoint
def build_model_from_ckpt(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    state = ck['model']
    keys = list(state.keys())

    arch = {
        'embed_dim': 64, 'attention_type': 'swin', 'window_size': 8,
        'use_simple_gate': True, 'use_gcm': False, 'use_color_pre': False,
        'use_dual_branch': False,
    }

    if any('ocab' in k for k in keys):
        arch['attention_type'] = 'ocab'
        arch['window_size'] = 16
        arch['embed_dim'] = 60
        arch['depths'] = (2, 2, 2, 2)
    if any('gcm' in k for k in keys):
        arch['use_gcm'] = True
    if any('color_pre' in k for k in keys):
        arch['use_color_pre'] = True
    if any('dual' in k for k in keys):
        arch['use_dual_branch'] = True

    kwargs = {k: v for k, v in arch.items()}
    model = RestoreNet(in_ch=3, **kwargs).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"    Warning: {len(missing)} missing keys")
    model.eval()
    return model

# Process
all_bids = sorted(ckpt_map.keys())
my_bids = [b for i, b in enumerate(all_bids) if i % NUM_GPUS == GPU_OFFSET]
print(f"GPU{GPU_OFFSET}: {len(my_bids)} challenges: {my_bids}")

results = {}

for ds_name, ds_path in [('DIV2K', '/data/zyli/projects/autoresearch/datasets/DIV2K/DIV2K_valid_HR'),
                          ('LSDIR', '/data/zyli/projects/autoresearch/datasets/LSDIR/val1/HR/val')]:
    if not os.path.isdir(ds_path): continue

    clean_images = sorted(glob.glob(f'{ds_path}/*.png') + glob.glob(f'{ds_path}/*.jpg'))
    print(f"[{ds_name}] {len(clean_images)} clean images")

    clean_cache = {}
    for p in clean_images:
        clean_cache[p] = np.array(Image.open(p).convert('RGB'), dtype=np.uint8)

    ds_results = {}

    for bid in my_bids:
        best_var = BEST_CHOICES.get(bid, 'r0')
        avail = ckpt_map.get(bid, {})
        if best_var not in avail:
            avail_keys = list(avail.keys())
            if not avail_keys: continue
            best_var = avail_keys[0]

        ckpt_path = ckpt_map[bid][best_var]
        params_file = DEG_DIR / f'{bid}_params.json'
        if not params_file.exists(): continue
        pipeline = json.load(open(params_file))['pipeline']

        try:
            model = build_model_from_ckpt(ckpt_path)
        except Exception as e:
            print(f"  {bid}: SKIP (model error: {e})")
            continue
        print(f"  {bid} ({best_var})")

        psnr_sum = ssim_sum = lpips_sum = count = 0

        with torch.no_grad():
            for img_path in clean_images:
                clean = clean_cache[img_path]
                degraded = clean.copy()
                for step in pipeline:
                    degraded = add_distortion(degraded, severity=step['severity'],
                                               distortion_name=step['function'])

                d_t = torch.from_numpy(degraded).permute(2,0,1).float().div_(255).unsqueeze(0).to(device)
                H, W = d_t.shape[2], d_t.shape[3]
                pad = 16
                ph, pw = (pad - H%pad)%pad, (pad - W%pad)%pad
                if ph or pw:
                    d_t = F.pad(d_t, (0, pw, 0, ph), mode='reflect')

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = model(d_t)
                if isinstance(out, (list,tuple)): out = out[0]
                restored = out[:,:,:H,:W].float().squeeze(0)

                c_t = torch.from_numpy(clean).permute(2,0,1).float().div_(255).to(device)
                if restored.shape != c_t.shape:
                    restored = F.interpolate(restored.unsqueeze(0),
                                             size=(c_t.shape[1], c_t.shape[2]),
                                             mode='bilinear', align_corners=False).squeeze(0)

                mse = (restored - c_t).pow(2).mean()
                psnr = 10*torch.log10(1.0/mse).item() if mse > 0 else 100.0
                ss = ssim_val(restored, c_t)
                with torch.no_grad():
                    lp = lpips_fn(restored.unsqueeze(0)*2-1, c_t.unsqueeze(0)*2-1).item()

                psnr_sum += psnr; ssim_sum += ss; lpips_sum += lp; count += 1

        avg_p = psnr_sum/count if count else 0
        avg_s = ssim_sum/count if count else 0
        avg_l = lpips_sum/count if count else 0
        ds_results[bid] = {'psnr': round(avg_p,2), 'ssim': round(avg_s,4),
                           'lpips': round(avg_l,4), 'count': count}
        print(f"    {ds_name}: PSNR={avg_p:.2f} SSIM={avg_s:.4f} LPIPS={avg_l:.4f}")

    results[ds_name] = ds_results

# Save per-GPU JSON
out_file = OUT_DIR / f'results_gpu{GPU_OFFSET}.json'
json.dump(results, open(out_file, 'w'), indent=2)
print(f"GPU{GPU_OFFSET}: Saved to {out_file}")
