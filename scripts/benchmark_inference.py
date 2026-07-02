"""Benchmark inference: run IR methods on exp40 challenge images, save tar.
Usage: AR_METHOD=<name> AR_GPU=<N> <python> scripts/benchmark_inference.py
Each method uses its own conda environment.
"""
import json, os, sys, glob, io, tarfile, argparse
import numpy as np
import torch
from PIL import Image

np.random.seed(42)

METHOD = os.environ.get('AR_METHOD', 'airnet')
GPU_ID = int(os.environ.get('AR_GPU_ID', 0))
BENCH = 'benchmark'
OUT_DIR = 'benchmark_results/images'
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')

# Collect challenge images
challenges = sorted(d for d in os.listdir('exp40/challenges/phase4')
                    if os.path.isdir(f'exp40/challenges/phase4/{d}'))
images = []
for ch in challenges:
    d = f'exp40/challenges/phase4/{ch}'
    for fname in sorted(os.listdir(d)):
        if fname.startswith('degraded') and fname.endswith('.png'):
            degraded_path = os.path.join(d, fname)
            clean_path = os.path.join(d, fname.replace('degraded', 'clean'))
            if os.path.exists(clean_path):
                images.append((ch, degraded_path, clean_path))

print(f"[{METHOD}] {len(images)} images from {len(challenges)} challenges")

def load_image(path):
    img = Image.open(path).convert('RGB')
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)

def tensor_to_jpg(tensor):
    t = tensor.squeeze(0) if tensor.dim() == 4 else tensor
    arr = t.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(arr, 'RGB').save(buf, format='JPEG', quality=100, subsampling=0)
    return buf.getvalue()

# ============================================================
# Model loading per method
# ============================================================
model = None
model_type = 'standard'  # 'airnet' needs special forward

if METHOD == 'airnet':
    sys.path.insert(0, f'{BENCH}/AirNet')
    from net.model import AirNet
    opt = argparse.Namespace()
    opt.batch_size = 5  # Match checkpoint: K = batch_size * 256 = 1280
    model = AirNet(opt).to(device)
    ckpt = torch.load(f'{BENCH}/AirNet/ckpt/All.pth', map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model_type = 'airnet'

elif METHOD == 'dcpt':
    sys.path.insert(0, f'{BENCH}/dcpt')
    from net.model import DCPT_Model
    model = DCPT_Model(in_ch=3, out_ch=3).to(device)
    ckpt = torch.load(f'{BENCH}/dcpt/weights/dcpt_promptir_5d.pth', map_location=device, weights_only=True)
    state = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state, strict=False)

elif METHOD == 'dfpir':
    DFPIR_DIR = 'resource/Degradation-Aware-Feature-Perturbation-for-All-in-One-Image-Restoration-Modified-Version-'
    sys.path.insert(0, DFPIR_DIR)
    from net.model import ChannelShuffle_skip_textguaid
    import torch.nn as nn
    import torch.nn.functional as F
    # DFPIRBlind wrapper (same as in test_degradation.py)
    class DFPIRBlind(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model
            self.text_embed = nn.Parameter(torch.randn(1, 512) * 0.02)
        def forward(self, x):
            _, _, H, W = x.shape
            ph, pw = (16 - H % 16) % 16, (16 - W % 16) % 16
            if ph or pw: x = F.pad(x, (0, pw, 0, ph), mode='reflect')
            b = x.shape[0]
            out = self.base(x, self.text_embed.expand(b, -1))
            return out[:, :, :H, :W]
    base = ChannelShuffle_skip_textguaid(dim=48, num_blocks=[4,6,6,8], heads=[1,2,4,8],
                                          ffn_expansion_factor=2.66, bias=False,
                                          LayerNorm_type='WithBias')
    model = DFPIRBlind(base).to(device)
    ckpt = torch.load(f'{DFPIR_DIR}/dfpir_blind/checkpoints/dfpir_blind_step301920.pt',
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])

elif METHOD == 'promptir':
    sys.path.insert(0, f'{BENCH}/PromptIR')
    from net.model import PromptIR
    model = PromptIR().to(device)
    ckpt = torch.load(f'{BENCH}/PromptIR/ckpt/model.ckpt', map_location=device, weights_only=True)
    state = ckpt.get('state_dict', ckpt)
    state = {k.replace('model.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

elif METHOD == 'r2r':
    sys.path.insert(0, f'{BENCH}/R2R')
    from model.r2r import R2R_Model
    model = R2R_Model().to(device)
    ckpt = torch.load(f'{BENCH}/R2R/ckpt/train_ckpt_5D_f/last.ckpt', map_location=device, weights_only=True)
    state = ckpt.get('state_dict', ckpt)
    state = {k.replace('model.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

elif METHOD == 'datprl-ir':
    sys.path.insert(0, f'{BENCH}/DATPRL-IR')
    from net.model import DATPRL_IR_Model
    model = DATPRL_IR_Model().to(device)
    ckpts = glob.glob(f'{BENCH}/DATPRL-IR/**/*.pth', recursive=True)
    if ckpts:
        ckpt = torch.load(ckpts[0], map_location=device, weights_only=True)
        state = ckpt.get('model', ckpt)
        model.load_state_dict(state, strict=False)

if model is None:
    print(f"[{METHOD}] FAILED to load model")
    sys.exit(1)

model.eval()
print(f"[{METHOD}] Model loaded. Running inference...")

# ============================================================
# Inference
# ============================================================
tar_path = f'{OUT_DIR}/{METHOD}_exp40.tar'
tar = tarfile.open(tar_path, 'w')
global_idx = 0

with torch.no_grad():
    for ch, degraded_path, clean_path in images:
        degraded = load_image(degraded_path).unsqueeze(0).to(device)
        clean = load_image(clean_path).unsqueeze(0)
        _, _, H, W = degraded.shape

        # Pad to multiple of 16 (required by most IR models)
        pad_h = ((H + 15) // 16) * 16 - H
        pad_w = ((W + 15) // 16) * 16 - W
        if pad_h > 0 or pad_w > 0:
            degraded_pad = torch.nn.functional.pad(degraded, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            degraded_pad = degraded

        # Forward
        if model_type == 'airnet':
            out = model(degraded_pad, degraded_pad)
        else:
            out = model(degraded_pad)

        if isinstance(out, (list, tuple)):
            out = out[0]

        restored = out[:, :, :H, :W].float()

        prefix = f'{ch}/{global_idx:04d}'
        global_idx += 1
        for tag, tensor in [('degraded', degraded.float()), ('restored', restored), ('clean', clean.float())]:
            jpg = tensor_to_jpg(tensor)
            info = tarfile.TarInfo(name=f'{prefix}.{tag}.jpg')
            info.size = len(jpg)
            tar.addfile(info, io.BytesIO(jpg))

tar.close()
print(f"[{METHOD}] Saved {global_idx} images to {tar_path}")
