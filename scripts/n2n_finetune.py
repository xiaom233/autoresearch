"""N2N finetune per challenge: train on 5 degraded images, infer on DIV2K+LSDIR.
Usage: AR_GPU_OFFSET=0 AR_NUM_GPUS=8 python n2n_finetune.py
"""
import json, os, sys, io, tarfile, glob, time, math
import numpy as np
import torch, torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

GPU_OFFSET = int(os.environ.get('AR_GPU_OFFSET', 0))
NUM_GPUS = int(os.environ.get('AR_NUM_GPUS', 8))

sys.path.insert(0, '/data/zyli/projects/autoresearch')
sys.path.insert(0, '/data/zyli/projects/autoresearch/benchmark/Neighbor2Neighbor')
from x_distortion import add_distortion
from arch_unet import UNet

device = torch.device('cuda:0')
DEG_DIR = Path('/data/zyli/projects/autoresearch/exp40/degradation_gt')
CHALLENGE_DIR = Path('/data/zyli/projects/autoresearch/exp40/challenges/phase4')
OUT_DIR = Path('/data/zyli/projects/autoresearch/benchmark_results/n2n_ft')
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = Path('/data/zyli/projects/autoresearch/benchmark_results/n2n_ft_ckpts')
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Align with M_blind: 15094 steps x 16 batch = 241,504 samples
M_BLIND_SAMPLES = 241504
PATCH_SIZE = 256
BATCH_SIZE = 4
LR = 3e-4
GAMMA = 0.5
MILESTONES = [20, 40, 60, 80]
NUM_WORKERS = 4


class N2NDataset(Dataset):
    """Random patch crops from preloaded images. Each __getitem__ returns a
    different random (x1, x2) N2N neighbor pair, so every epoch sees fresh crops."""

    def __init__(self, img_paths, patch_size=256, length=1000):
        self.patch_size = patch_size
        self.length = length
        # uint8 storage (~41 MB for 5 x 2K images) to keep multiprocessing overhead low
        self._imgs = []
        self._shapes = []
        for p in img_paths:
            img = np.array(Image.open(p).convert('RGB'), dtype=np.uint8)
            self._imgs.append(img)
            self._shapes.append(img.shape[:2])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        i = np.random.randint(len(self._imgs))
        img = self._imgs[i]
        h, w = self._shapes[i]
        y = np.random.randint(0, h - self.patch_size + 1)
        x = np.random.randint(0, w - self.patch_size + 1)
        patch = img[y:y + self.patch_size, x:x + self.patch_size]
        patch_t = torch.from_numpy(patch).permute(2, 0, 1).float().div_(255.0)
        # N2N sub-sampling: neighbor pairs
        x1 = patch_t[:, 0::2, 0::2]
        x2 = patch_t[:, 0::2, 1::2]
        return x1, x2


def to_jpg(t):
    if t.dim() == 4: t = t.squeeze(0)
    arr = t.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(arr, 'RGB').save(buf, format='JPEG', quality=100, subsampling=0)
    return buf.getvalue()


all_bids = sorted([f'blind_{i:04d}' for i in range(1, 25)])
my_bids = [b for i, b in enumerate(all_bids) if i % NUM_GPUS == GPU_OFFSET]
print(f"GPU{GPU_OFFSET}: {len(my_bids)} challenges")


def n2n_infer(model, img_tensor):
    _, _, H, W = img_tensor.shape
    ph = (32 - H % 32) % 32
    pw = (32 - W % 32) % 32
    if ph or pw:
        img_tensor = F.pad(img_tensor, (0, pw, 0, ph), mode='reflect')
    with torch.no_grad():
        out = model(img_tensor)
    return out[:, :, :H, :W]


# Preload clean images once (reused across all challenges to avoid IO bottleneck)
print("Preloading clean images...")
CLEAN_CACHE = {}
for ds_name, ds_path in [('DIV2K', '/data/zyli/projects/autoresearch/datasets/DIV2K/DIV2K_valid_HR'),
                          ('LSDIR', '/data/zyli/projects/autoresearch/datasets/LSDIR/val1/HR/val')]:
    if not os.path.isdir(ds_path): continue
    img_paths = sorted(glob.glob(f'{ds_path}/*.png') + glob.glob(f'{ds_path}/*.jpg'))
    CLEAN_CACHE[ds_name] = {p: np.array(Image.open(p).convert('RGB'), dtype=np.uint8) for p in img_paths}
    print(f"  {ds_name}: {len(CLEAN_CACHE[ds_name])} images")
print("Preloading done.")


for bid in my_bids:
    # -- Training phase --
    print(f"\n{'='*50}\n{bid}: Training N2N on 5 images\n{'='*50}")

    # Load pretrained N2N
    model = UNet(in_nc=3, out_nc=3, n_feature=48).to(device)
    ckpt_path = '/data/zyli/projects/autoresearch/benchmark/Neighbor2Neighbor/pretrained_model/model_gauss5-50_b4e100r02.pth'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)

    # Dataset: 5 degraded images -> random patches
    ch_dir = CHALLENGE_DIR / bid
    img_paths = sorted(ch_dir.glob('degraded_*.png'))
    print(f"  Loaded {len(img_paths)} images")

    PATCHES_PER_EPOCH = 1000
    epochs = max(10, math.ceil(M_BLIND_SAMPLES / PATCHES_PER_EPOCH))
    steps_per_epoch = PATCHES_PER_EPOCH // BATCH_SIZE

    dataset = N2NDataset([str(p) for p in img_paths], patch_size=PATCH_SIZE, length=PATCHES_PER_EPOCH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2,
                        persistent_workers=True)
    print(f"  {PATCHES_PER_EPOCH} patches/epoch, {epochs} epochs, {NUM_WORKERS} workers")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
        milestones=[int(epochs * m / 100) for m in MILESTONES], gamma=GAMMA)
    save_every = max(1, math.ceil(epochs / 10))

    model.train()
    t0 = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        for x1, x2 in loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            optimizer.zero_grad()
            out = model(x1)
            loss = F.mse_loss(out, x2)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        elapsed = time.time() - t0
        eta = elapsed / (epoch + 1) * (epochs - epoch - 1)
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            ckpt_out = CKPT_DIR / f'{bid}_epoch{epoch+1}.pth'
            torch.save(model.state_dict(), ckpt_out)
            print(f"    epoch {epoch+1}/{epochs}: loss={epoch_loss/steps_per_epoch:.6f} lr={scheduler.get_last_lr()[0]:.2e} [{elapsed:.0f}s/{eta:.0f}s] [saved {ckpt_out.name}]")
        elif epoch % max(1, epochs // 10) == 0:
            print(f"    epoch {epoch+1}/{epochs}: loss={epoch_loss/steps_per_epoch:.6f} lr={scheduler.get_last_lr()[0]:.2e} [{elapsed:.0f}s/{eta:.0f}s]")

    # Save final model for inference
    final_ckpt = CKPT_DIR / f'{bid}.pth'
    torch.save(model.state_dict(), final_ckpt)
    print(f"  Final model: {final_ckpt}")

    # -- Inference phase --
    print(f"  Inference...")
    model.eval()
    pipeline = json.load(open(DEG_DIR / f'{bid}_params.json'))['pipeline']

    for ds_name, ds_path in [('DIV2K', '/data/zyli/projects/autoresearch/datasets/DIV2K/DIV2K_valid_HR'),
                              ('LSDIR', '/data/zyli/projects/autoresearch/datasets/LSDIR/val1/HR/val')]:
        if ds_name not in CLEAN_CACHE: continue
        clean_cache = CLEAN_CACHE[ds_name]
        img_paths = list(clean_cache.keys())
        out_tar = tarfile.open(OUT_DIR / f'{ds_name}_gpu{GPU_OFFSET}.tar', 'a')
        count = 0

        for img_path in img_paths:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            clean = clean_cache[img_path]
            degraded = clean.copy()
            for step in pipeline: degraded = add_distortion(degraded, severity=step['severity'], distortion_name=step['function'])
            d_t = torch.from_numpy(degraded).permute(2, 0, 1).float().div_(255).unsqueeze(0).to(device)
            restored = n2n_infer(model, d_t)
            for tag, t in [('degraded', d_t.float()), ('restored', restored),
                           ('clean', torch.from_numpy(clean).permute(2, 0, 1).float().div_(255).unsqueeze(0))]:
                jpg = to_jpg(t)
                info = tarfile.TarInfo(name=f'{bid}/{img_name}.{tag}.jpg')
                info.size = len(jpg)
                out_tar.addfile(info, io.BytesIO(jpg))
            count += 1
        out_tar.close()
        print(f"    {ds_name}: {count} imgs")

    del model

print(f"\nGPU{GPU_OFFSET} Done")
