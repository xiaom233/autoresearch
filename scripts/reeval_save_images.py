"""GT re-evaluation with image saving as tar. Fixed seed for deterministic degradation.
Usage: AR_GPU_ID=N .venv/bin/python3 scripts/reeval_save_images.py [EXP]
Output: expN/results/images/{group}_{bid}.tar (webdataset format)
"""
import json, os, sys, glob, math, io, tarfile
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, '.')
from train import RestoreNet, ValDataset, evaluate_all, _pad_for_window, compute_psnr

# Fixed seed — critical for deterministic motion blur angle + noise
SEED = 42
np.random.seed(SEED)

GPU_ID = int(os.environ.get('AR_GPU_ID', 0))
EXP = sys.argv[1] if len(sys.argv) > 1 else 'exp40'
device = torch.device('cuda:0')
amp = torch.amp.autocast('cuda', dtype=torch.bfloat16)

VAL_DIRS = ['datasets/DIV2K/DIV2K_valid_HR', 'datasets/LSDIR/LSDIR/val1']
VAL_DIRS = [(d, d.split('/')[-2]) for d in VAL_DIRS if os.path.isdir(d)]

# Collect models: r1_* and direct_swin_*
models = []
for d in os.listdir(f'{EXP}/experiments'):
    if not (d.startswith('qwen_direct_swin_blind_')):
        continue
    ckpt_dir = f'{EXP}/experiments/{d}/checkpoints'
    if not os.path.isdir(ckpt_dir): continue
    pts = glob.glob(f'{ckpt_dir}/*.pt')
    if not pts: continue
    parts = d.split('_')
    for i, p in enumerate(parts):
        if p == 'blind':
            bid = '_'.join(parts[i:])
            break
    else: continue
    if not os.path.exists(f'{EXP}/degradation_gt/{bid}_params.json'): continue
    models.append((d, bid))

models.sort(key=lambda x: x[0])
my_models = [models[i] for i in range(len(models)) if i % 8 == GPU_ID]
print(f'[{GPU_ID}] {len(my_models)} models (seed={SEED})')

def tensor_to_jpg(tensor):
    arr = tensor[0].clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(arr, 'RGB').save(buf, format='JPEG', quality=100, subsampling=0)
    return buf.getvalue()

for name, bid in my_models:
    ckpt_dir = f'{EXP}/experiments/{name}/checkpoints'
    log_file = f'{EXP}/logs/{name}.log'
    gt_params = f'{EXP}/degradation_gt/{bid}_params.json'
    pts = sorted(glob.glob(f'{ckpt_dir}/*.pt'))
    if not pts: continue

    embed_dim = 64; attn = 'swin'; ws = 8; use_sg = use_gcm = 0
    if os.path.exists(log_file):
        with open(log_file, errors='ignore') as f:
            for line in f:
                if 'EXP_META:' in line:
                    try:
                        mo = json.loads(line.split('EXP_META: ')[1].split(' ===')[0])['model']
                        attn = mo.get('attention_type', 'swin')
                        ws = mo.get('window_size', 8)
                        embed_dim = mo.get('embed_dim', 64)
                        use_sg = int(bool(mo.get('simple_gate', 0)))
                        use_gcm = int(bool(mo.get('gcm', 0)))
                    except: pass; break

    kwargs = dict(in_ch=3, embed_dim=embed_dim, attention_type=attn, window_size=ws,
                  use_simple_gate=use_sg, use_gcm=use_gcm)
    if attn == 'ocab' and embed_dim == 30:
        kwargs['num_heads'] = [6, 6, 6, 6]

    model = RestoreNet(**kwargs).to(device)
    ck = torch.load(pts[-1], map_location=device, weights_only=True)
    model.load_state_dict(ck['model'])
    model.eval()

    out_dir = f'{EXP}/results/images'
    os.makedirs(out_dir, exist_ok=True)
    tar_path = f'{out_dir}/{name}.tar'
    tar = tarfile.open(tar_path, 'w')

    psnr_sum = 0.0; n_valid = 0; global_idx = 0
    with torch.no_grad():
        for ds_path, ds_name in VAL_DIRS:
            np.random.seed(SEED)  # Re-seed before each dataset for determinism
            try:
                ds = ValDataset([ds_path], count=0, params_path=gt_params)
            except Exception as e:
                print(f'[{GPU_ID}] {name}/{ds_name}: SKIP {e}')
                continue
            if len(ds) == 0: continue

            loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                                  num_workers=4, pin_memory=True)
            for degraded, clean in loader:
                degraded = degraded.to(device, non_blocking=True)
                clean_t = clean.to(device, non_blocking=True)

                degraded_pad, orig_H, orig_W = _pad_for_window(degraded, ws)
                with amp:
                    out = model(degraded_pad)
                pred_f = out[:, :, :orig_H, :orig_W].float()
                clean_f = clean_t[:, :, :orig_H, :orig_W].float()
                degraded_save = degraded[:, :, :orig_H, :orig_W].float()
                psnr_sum += compute_psnr(pred_f, clean_f)
                n_valid += 1

                prefix = f'{global_idx:06d}'; global_idx += 1
                for tag, tensor in [('degraded', degraded_save), ('restored', pred_f), ('clean', clean_f)]:
                    jpg = tensor_to_jpg(tensor)
                    info = tarfile.TarInfo(name=f'{prefix}.{tag}.jpg')
                    info.size = len(jpg)
                    tar.addfile(info, io.BytesIO(jpg))

            torch.cuda.empty_cache()

    tar.close()
    avg = psnr_sum / n_valid if n_valid > 0 else 0.0
    print(f'[{GPU_ID}] {name}: PSNR={avg:.2f} ({n_valid} imgs) -> tar')

print(f'[{GPU_ID}] Done')
