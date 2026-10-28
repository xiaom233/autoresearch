"""Phase 1: Infer exp40 Best(R0,R1) checkpoints on DIV2K/LSDIR, save tar.
Output: benchmark_results/exp40_best/{DIV2K|LSDIR}_gpu{N}.tar
Format: {bid}/{img_name}.{degraded|restored|clean}.jpg  (matches benchmark_inference.py)
Usage: AR_GPU_OFFSET=0 AR_NUM_GPUS=8 python -u save_exp40_best.py
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
from train import RestoreNet

device = torch.device('cuda:0')
EXP_DIR = Path('/data/zyli/projects/autoresearch/exp40/experiments')
DEG_DIR = Path('/data/zyli/projects/autoresearch/exp40/degradation_gt')
OUT_DIR = Path('/data/zyli/projects/autoresearch/benchmark_results/exp40_best')
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHOICES = {
    'blind_0001': 'r1','blind_0002': 'r1','blind_0003': 'r1',
    'blind_0004': 'r0','blind_0005': 'r1','blind_0006': 'r0',
    'blind_0007': 'r0','blind_0008': 'r0','blind_0009': 'r1','blind_0010': 'r0',
    'blind_0011': 'r0','blind_0012': 'r0','blind_0013': 'r1',
    'blind_0014': 'r1','blind_0015': 'r0','blind_0016': 'r1',
    'blind_0017': 'r1','blind_0018': 'r1','blind_0019': 'r1',
    'blind_0020': 'r1','blind_0021': 'r1','blind_0022': 'r1',
    'blind_0023': 'r0','blind_0024': 'r1',
}

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

def build_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    state = ck['model']
    keys = list(state.keys())
    # Detect embed_dim from conv_first weight
    embed_dim = state['conv_first.weight'].shape[0]
    arch = {'embed_dim': embed_dim, 'attention_type': 'swin', 'window_size': 8,
            'use_simple_gate': True, 'use_gcm': False, 'use_color_pre': False,
            'use_dual_branch': False}
    # OCAB detection: conv-based QKV (4D weight) vs linear (2D)
    has_ocab = any('ocab' in k for k in keys)
    if not has_ocab:
        # Check qkv weight dim: 4D = conv-based (OCAB), 2D = linear (swin/mdta)
        for k in keys:
            if 'qkv' in k and 'weight' in k:
                if len(state[k].shape) == 4:
                    has_ocab = True
                break
    if has_ocab:
        arch.update({'attention_type': 'ocab', 'window_size': 8})
        # Compute num_heads: pick largest divisor of embed_dim >= 3
        for nh in [6, 5, 3, 2]:
            if embed_dim % nh == 0:
                arch['num_heads'] = (nh, nh, nh, nh)
                break
    if any('gcm' in k for k in keys): arch['use_gcm'] = True
    if any('color_pre' in k for k in keys): arch['use_color_pre'] = True
    if any('dual' in k for k in keys): arch['use_dual_branch'] = True
    model = RestoreNet(in_ch=3, **arch).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

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

    # Pre-load clean images (fast path for non-DataLoader mode)
    USE_DATALOADER = os.environ.get('AR_DATALOADER', '1') == '1'
    clean_cache = {}
    for p in img_paths:
        clean_cache[p] = np.array(Image.open(p).convert('RGB'), dtype=np.uint8)

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
            model = build_model(ckpt_path)
        except Exception as e:
            print(f"  {bid}: SKIP model error: {e}")
            continue

        count = 0
        print(f"  {bid} ({best_var}) ...")

        if USE_DATALOADER:
            # DataLoader for parallel degradation
            class DegradeDataset(torch.utils.data.Dataset):
                def __init__(self, img_paths, pipeline, clean_cache):
                    self.items = [(p, os.path.splitext(os.path.basename(p))[0])
                                  for p in img_paths]
                    self.pipeline = pipeline
                    self.clean_cache = clean_cache
                def __len__(self): return len(self.items)
                def __getitem__(self, idx):
                    img_path, img_name = self.items[idx]
                    clean = self.clean_cache[img_path].copy()
                    degraded = clean.copy()
                    for step in self.pipeline:
                        degraded = add_distortion(degraded, severity=step['severity'],
                                                   distortion_name=step['function'])
                    return (img_name,
                            torch.from_numpy(degraded).permute(2,0,1).float().div_(255),
                            torch.from_numpy(clean).permute(2,0,1).float().div_(255))

            ds = DegradeDataset(img_paths, pipeline, clean_cache)
            loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                                  num_workers=4, pin_memory=True,
                                                  persistent_workers=True)

            with torch.no_grad():
                for img_name, degraded, clean in loader:
                    img_name = img_name[0] if isinstance(img_name, list) else img_name
                    if (bid, img_name) in saved_set:
                        count += 1
                        continue

                    degraded = degraded.to(device, non_blocking=True)
                    H, W = degraded.shape[2], degraded.shape[3]
                    pad = 16
                    ph, pw = (pad - H%pad)%pad, (pad - W%pad)%pad
                    dpad = F.pad(degraded, (0, pw, 0, ph), mode='reflect') if ph or pw else degraded

                    out = model(dpad)
                    if isinstance(out, (list,tuple)): out = out[0]
                    restored = out[:,:,:H,:W].float()

                    for tag, t in [('degraded', degraded.float()),
                                   ('restored', restored), ('clean', clean.float())]:
                        jpg = to_jpg(t)
                        info = tarfile.TarInfo(name=f'{bid}/{img_name}.{tag}.jpg')
                        info.size = len(jpg)
                        out_tar.addfile(info, io.BytesIO(jpg))

                    count += 1
                    if count % 50 == 0:
                        print(f"    {count}/{len(img_paths)}")
        else:
            # Sequential fallback
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
                    if ph or pw:
                        d_t = F.pad(d_t, (0, pw, 0, ph), mode='reflect')

                    out = model(d_t)
                    if isinstance(out, (list,tuple)): out = out[0]
                    restored = out[:,:,:H,:W].float()

                    for tag, t in [('degraded', d_t[:,:,:H,:W].float()),
                                   ('restored', restored), ('clean', torch.from_numpy(clean).permute(2,0,1).float().div_(255).unsqueeze(0))]:
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
