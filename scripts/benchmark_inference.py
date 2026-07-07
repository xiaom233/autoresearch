"""Benchmark inference: per-method, per-dataset, multi-GPU sharded.
Usage: AR_METHOD=<n> AR_GPU=0 AR_GPU_OFFSET=0 AR_NUM_GPUS=1 <python> scripts/benchmark_inference.py [EXP]
Output: benchmark_results/{method}/{dataset}_gpu{N}.tar
  Each entry: {bid}/{img_name}.{degraded|restored|clean}.jpg
"""
import json, os, sys, io, tarfile, argparse, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Ensure project root is in sys.path regardless of CWD
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

np.random.seed(42)

METHOD = os.environ['AR_METHOD']
GPU_ID = int(os.environ.get('AR_GPU_ID', 0))
NUM_GPUS = int(os.environ.get('AR_NUM_GPUS', 1))
GPU_OFFSET = int(os.environ.get('AR_GPU_OFFSET', 0))
EXP = sys.argv[1] if len(sys.argv) > 1 else 'exp40'

from x_distortion import add_distortion

device = torch.device(f'cuda:{GPU_ID}')
OUT_BASE = 'benchmark_results'

# ============================================================
# Model loading
# ============================================================
model = None
is_airnet = False
is_dfpir_official = False
text_code_dfpir = None

if METHOD == 'xrestormer':
    sys.path.insert(0, os.path.join(_project_root, 'benchmark/X-Restormer'))
    import xrestormer.archs
    from xrestormer.archs.xrestormer_arch import XRestormer
    model = XRestormer(dim=48, num_blocks=[2,4,4,4], window_size=8,
                       spatial_heads=[1,2,4,8]).to(device)
    ckpt = torch.load('benchmark/X-Restormer/weights/denoise_300k.pth', map_location=device)
    model.load_state_dict(ckpt['params'])
    model_type = 'xrestormer'

elif METHOD == 'datprl-ir':
    sys.path.insert(0, os.path.join(_project_root, 'benchmark/DATPRL-IR/All_in_One'))
    from net.promptrestormerv3_arch import DualPromptRestormerv3_only_task_prompt
    model = DualPromptRestormerv3_only_task_prompt().to(device)
    ckpt = torch.load('benchmark/DATPRL-IR/DATPRL-IR_net_g_cdd11.pth', map_location=device)
    model.load_state_dict(ckpt['params'])

elif METHOD == 'r2r':
    sys.path.insert(0, os.path.join(_project_root, 'benchmark/R2R'))
    from net.model_5D import R2RLocal
    model = R2RLocal(
        ckpt_path='benchmark/R2R/ckpt/train_ckpt_5D_f/',
        prompts_path='benchmark/R2R/prompt-20260622T130433Z-3-001/prompt/save_prompts_5D',
        prompts_name='last',
        train_mode='finetune'
    ).to(device)

elif METHOD == 'dcpt':
    sys.path.insert(0, os.path.join(_project_root, 'benchmark/PromptIR'))
    from net.model import PromptIR
    net = PromptIR(decoder=True, dim=48, num_blocks=[4,6,6,8], heads=[1,2,4,8],
                   ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias',
                   num_refinement_blocks=4).to(device)
    ckpt = torch.load('benchmark/dcpt/weights/dcpt_promptir_5d.pth', map_location=device)
    net.load_state_dict(ckpt['params'], strict=False)
    model = net

elif METHOD == 'promptir':
    sys.path.insert(0, os.path.join(_project_root, 'benchmark/PromptIR'))
    from net.model import PromptIR
    # Manually load state_dict (Lightning load_from_checkpoint hangs)
    net = PromptIR(decoder=True, dim=48, num_blocks=[4,6,6,8], heads=[1,2,4,8],
                   ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias',
                   num_refinement_blocks=4).to(device)
    ckpt = torch.load('benchmark/PromptIR/ckpt/model.ckpt', map_location=device, weights_only=True)
    state = ckpt['state_dict']
    state = {k.replace('net.', ''): v for k, v in state.items()}
    net.load_state_dict(state, strict=False)
    model = net

elif METHOD == 'airnet':
    sys.path.insert(0, os.path.join(_project_root, 'benchmark/AirNet'))
    from net.model import AirNet
    opt = argparse.Namespace(); opt.batch_size = 5
    model = AirNet(opt).to(device)
    ckpt = torch.load('benchmark/AirNet/ckpt/All.pth', map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    is_airnet = True

elif METHOD == 'dfpir':
    sys.path.insert(0, os.path.join(_project_root, 'benchmark/DFPIR'))
    from net.model import ChannelShuffle_skip_textguaid
    model = ChannelShuffle_skip_textguaid(device=device).to(device)
    ckpt = torch.load('benchmark/DFPIR/DFPIR-5D-pn31.29-0.8889_pr37.62-0.9779_ph31.64-0.9794_pb28.82-0.8734_pl23.82-0.8428_avr30.64-0.9125.pth.tar',
                      map_location=device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model_type = 'dfpir_official'
    is_dfpir_official = True
    text_code_dfpir = torch.randn(1, 512).to(device)

if model is None:
    print(f"[{METHOD}] FAILED"); sys.exit(1)
model.eval()
print(f"[{METHOD}] Model loaded")

# ============================================================
# Helpers
# ============================================================
def to_jpg(t):
    """tensor [1,3,H,W] or [3,H,W] -> JPEG bytes"""
    if t.dim() == 4: t = t.squeeze(0)
    arr = t.clamp(0,1).mul(255).byte().permute(1,2,0).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(arr, 'RGB').save(buf, format='JPEG', quality=100, subsampling=0)
    return buf.getvalue()

# ============================================================
# DataLoader Dataset
# ============================================================
class BenchDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, pipeline, bid):
        self.img_paths = img_paths
        self.pipeline = pipeline
        self.bid = bid
    def __len__(self): return len(self.img_paths)
    def __getitem__(self, idx):
        clean = np.array(Image.open(self.img_paths[idx]).convert('RGB'), dtype=np.uint8)
        degraded = clean.copy()
        for step in self.pipeline:
            degraded = add_distortion(degraded, severity=step['severity'],
                                       distortion_name=step['function'])
        img_name = os.path.splitext(os.path.basename(self.img_paths[idx]))[0]
        return (self.bid, img_name,
                torch.from_numpy(degraded).permute(2,0,1).float().div_(255),
                torch.from_numpy(clean).permute(2,0,1).float().div_(255))

# ============================================================
# Process datasets
# ============================================================
for ds_name, ds_path in [('DIV2K', 'datasets/DIV2K/DIV2K_valid_HR'),
                          ('LSDIR', 'datasets/LSDIR/val1/HR/val')]:
    if not os.path.isdir(ds_path): continue

    img_paths = sorted(os.path.join(ds_path, f) for f in os.listdir(ds_path)
                       if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.webp')))

    os.makedirs(f'{OUT_BASE}/{METHOD}', exist_ok=True)

    # Auto-detect GPU count from existing valid tars (scan ALL offsets, not just this GPU)
    existing_tars = []
    for candidate in sorted(glob.glob(f'{OUT_BASE}/{METHOD}/{ds_name}_gpu*.tar')):
        try:
            tarfile.open(candidate, 'r').close()
            existing_tars.append(candidate)
        except: pass

    if existing_tars:
        ds_num_gpus = len(existing_tars)
        if GPU_OFFSET >= ds_num_gpus:
            print(f"[{METHOD}] {ds_name}_gpu{GPU_OFFSET}: not needed ({ds_num_gpus} GPUs, tar count)")
            continue
    else:
        ds_num_gpus = NUM_GPUS
    ds_gpu_offset = GPU_OFFSET

    out_path = f'{OUT_BASE}/{METHOD}/{ds_name}_gpu{ds_gpu_offset}.tar'

    # Resume: check existing tar for already-saved images
    saved_set = set()
    if os.path.exists(out_path):
        try:
            with tarfile.open(out_path, 'r') as t:
                for info in t.getmembers():
                    parts = info.name.split('/')
                    if len(parts) >= 2:
                        bid = parts[0]
                        img_name = parts[1].rsplit('.', 2)[0]
                        saved_set.add((bid, img_name))
            print(f"[{METHOD}] {ds_name}_gpu{ds_gpu_offset}: RESUME ({len(saved_set)} already saved)")
        except:
            print(f"[{METHOD}] {ds_name}_gpu{ds_gpu_offset}: CORRUPT, restarting")
            os.remove(out_path)
            saved_set = set()

    out_tar = tarfile.open(out_path, 'a' if saved_set else 'w')
    total = 0; saved = len(saved_set)

    # Pre-build (img_idx, pipeline) list → single Dataset
    pipelines = {}
    for i in range(1, 25):
        bid = f'blind_{i:04d}'
        pf = f'{EXP}/degradation_gt/{bid}_params.json'
        if os.path.exists(pf):
            pipelines[bid] = json.load(open(pf))['pipeline']

    all_items = []
    for bid, pipeline in pipelines.items():
        for idx, img_path in enumerate(img_paths):
            all_items.append((bid, pipeline, img_path, idx))
    print(f"[{METHOD}] {ds_name}: {len(all_items)} total items, filtering to GPU{ds_gpu_offset}/{ds_num_gpus}")

    # Pre-load all clean images into memory (eliminates disk I/O bottleneck)
    print(f"[{METHOD}] {ds_name}: pre-loading {len(img_paths)} clean images into memory...")
    clean_cache = {}
    for img_path in img_paths:
        clean_cache[img_path] = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)

    # Single Dataset for all challenges
    class AllBenchDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, idx):
            bid, pipeline, img_path, img_idx = self.items[idx]
            clean = clean_cache[img_path].copy()
            degraded = clean.copy()
            for step in pipeline:
                degraded = add_distortion(degraded, severity=step['severity'],
                                           distortion_name=step['function'])
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            return (bid, img_name,
                    torch.from_numpy(degraded).permute(2,0,1).float().div_(255),
                    torch.from_numpy(clean_cache[img_path]).permute(2,0,1).float().div_(255))

    # Filter to this GPU shard
    my_items = [all_items[i] for i in range(len(all_items)) if i % ds_num_gpus == ds_gpu_offset]
    if saved_set:
        img_path_to_name = {img_path: os.path.splitext(os.path.basename(img_path))[0] for img_path in img_paths}
        my_items = [item for item in my_items
                    if (item[0], img_path_to_name.get(item[2], '')) not in saved_set]
        print(f"[{METHOD}] {ds_name}_gpu{GPU_OFFSET}: {len(my_items)} new items to process (skipped {len(saved_set)} saved)")
    ds = AllBenchDataset(my_items)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                          num_workers=4, pin_memory=True, persistent_workers=True)
    print(f"[{METHOD}] {ds_name}: {len(my_items)} items for this GPU")

    with torch.no_grad():
        for bid_i, img_name, degraded, clean in loader:
                saved += 1

                degraded = degraded.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
                H, W = degraded.shape[2], degraded.shape[3]

                init_pad = 64 if METHOD == 'xrestormer' else 16
                ph, pw = (init_pad - H % init_pad) % init_pad, (init_pad - W % init_pad) % init_pad
                dpad = F.pad(degraded, (0, pw, 0, ph), mode='reflect') if ph or pw else degraded

                # Tiled forward for large images (skip for R2R which handles full image)
                if max(H, W) > 1024 and METHOD != 'r2r':
                    T = 1024
                    pred = torch.zeros(1, 3, H, W, device=device, dtype=torch.bfloat16)
                    for y in range(0, H, T):
                        for x in range(0, W, T):
                            y2, x2 = min(y+T, H), min(x+T, W)
                            tile = dpad[:, :, y:y2, x:x2]
                            th, tw = y2-y, x2-x
                            if th < T or tw < T:
                                tile = F.pad(tile, (0, T-tw, 0, T-th))
                            pad_m = 64 if METHOD == 'xrestormer' else 16
                            ph2 = (pad_m - tile.shape[2] % pad_m) % pad_m
                            pw2 = (pad_m - tile.shape[3] % pad_m) % pad_m
                            if ph2 or pw2:
                                tile = F.pad(tile, (0, pw2, 0, ph2), mode='reflect')
                            if is_airnet:
                                out = model(tile, tile)
                            elif is_dfpir_official:
                                out = model(tile, text_code_dfpir.expand(tile.shape[0], -1))
                            else:
                                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                                    out = model(tile)
                            if isinstance(out, (list,tuple)): out = out[0]
                            pred[:, :, y:y2, x:x2] = out[:, :, :th, :tw]
                    restored = pred.float()
                else:
                    if is_airnet:
                        out = model(dpad, dpad)
                    elif is_dfpir_official:
                        out = model(dpad, text_code_dfpir.expand(dpad.shape[0], -1))
                    else:
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            out = model(dpad)
                    if isinstance(out, (list,tuple)): out = out[0]
                    restored = out[:, :, :H, :W].float()

                # Save
                bid_s = bid_i[0] if isinstance(bid_i, list) else bid_i
                img_s = img_name[0] if isinstance(img_name, list) else img_name
                for tag, t in [('degraded', degraded.float()), ('restored', restored), ('clean', clean.float())]:
                    jpg = to_jpg(t)
                    info = tarfile.TarInfo(name=f'{bid_s}/{img_s}.{tag}.jpg')
                    info.size = len(jpg)
                    out_tar.addfile(info, io.BytesIO(jpg))

                if saved % 50 == 0:
                    print(f"  [{METHOD}] {ds_name}: {saved} imgs")

    out_tar.close()
    print(f"[{METHOD}] {ds_name}: Done ({saved} imgs)")

print(f"[{METHOD}] GPU{GPU_OFFSET} Done")
