"""SSIM re-eval: bytes pre-load (outside DataLoader) + JPEG decode (inside DataLoader workers).
DIV2K and LSDIR processed separately.

Usage:
  .venv/bin/python3 scripts/benchmark_metrics.py --ssim method1 method2 ...
  .venv/bin/python3 scripts/benchmark_metrics.py --merge [--ssim] method1 method2 ...
"""
import json, os, tarfile, io, csv, glob, sys
import torch, torch.nn.functional as F
import torch.utils.data
from PIL import Image
import numpy as np

ALL_METRICS = {'psnr', 'ssim', 'lpips'}
RES_DIR = 'benchmark_results'
DATASETS = ['DIV2K', 'LSDIR']
NUM_GPUS = 8

def parse_metrics(args):
    metrics = set(ALL_METRICS)
    for i, a in enumerate(args):
        if a == '--metrics' and i+1 < len(args):
            metrics = set(args[i+1].split(',')) & ALL_METRICS
        elif a in ('--psnr', '--ssim', '--lpips') and not (i>0 and args[i-1]=='--metrics'):
            metrics = {a[2:]}
    return metrics

def parse_methods(args):
    skip = False; methods = []
    for a in args:
        if skip: skip = False; continue
        if a in ('--merge', '--psnr', '--ssim', '--lpips', '--metrics'): continue
        if a in ALL_METRICS or ',' in a: continue
        methods.append(a)
    return methods if methods else ['airnet','promptir','dcpt','dfpir']

raw_args = sys.argv[1:]
SELECTED_METRICS = parse_metrics(raw_args)
METHODS = parse_methods(raw_args)

# ============================================================
# Merge
# ============================================================
if '--merge' in sys.argv:
    for ds_name in DATASETS:
        merged = {}
        for method in METHODS:
            mdir = f'{RES_DIR}/{method}'
            for f in sorted(glob.glob(f'{mdir}/metrics_{ds_name}_gpu*.json')):
                data = json.load(open(f))
                for m_name in data.get(ds_name, {}):
                    merged.setdefault(m_name, {})
                    for bid, m in data[ds_name][m_name].items():
                        merged[m_name].setdefault(bid, {f'{k}_sum':0 for k in ALL_METRICS})
                        for k in ALL_METRICS: merged[m_name][bid].setdefault(f'{k}_count', 0)
                        for k in ALL_METRICS:
                            if f'{k}_sum' in m:
                                merged[m_name][bid][f'{k}_sum'] += m[f'{k}_sum']
                                merged[m_name][bid][f'{k}_count'] += m.get(f'{k}_count', 0)
        with open(f'{RES_DIR}/{ds_name}_benchmark.csv','w',newline='') as fh:
            fields = ['method','dataset','challenge']+list(ALL_METRICS)+['count']
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
            for method in sorted(merged):
                for bid in sorted(merged[method]):
                    m = merged[method][bid]
                    row = {'method':method,'dataset':ds_name,'challenge':bid}
                    for k in ALL_METRICS:
                        c = m[f'{k}_count']; row[k] = round(m[f'{k}_sum']/c, 4) if c>0 else None
                    row['count'] = max(m[f'{k}_count'] for k in ALL_METRICS)
                    w.writerow(row)
        for method in sorted(merged):
            mdir = f'{RES_DIR}/{method}'; os.makedirs(mdir, exist_ok=True)
            with open(f'{mdir}/{ds_name}_benchmark.csv','w',newline='') as fh:
                fields = ['method','dataset','challenge']+list(ALL_METRICS)+['count']
                w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
                for bid in sorted(merged[method]):
                    m = merged[method][bid]
                    row = {'method':method,'dataset':ds_name,'challenge':bid}
                    for k in ALL_METRICS:
                        c = m[f'{k}_count']; row[k] = round(m[f'{k}_sum']/c, 4) if c>0 else None
                    row['count'] = max(m[f'{k}_count'] for k in ALL_METRICS)
                    w.writerow(row)
            vals = [m['psnr_sum']/m['psnr_count'] for m in merged[method].values() if m['psnr_count']>0]
            if vals: print(f"  {method} {ds_name}: PSNR={sum(vals)/len(vals):.2f}")
    sys.exit(0)

# ============================================================
# JPEG decode (multi-worker via DataLoader)
# ============================================================
def decode_jpeg(data):
    """bytes → [C,H,W] float32 [0,1] tensor"""
    img = np.array(Image.open(io.BytesIO(data))).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)

class PairDataset(torch.utils.data.Dataset):
    """Dataset of (r_bytes, c_bytes) pairs. JPEG decode happens in __getitem__ (inside worker)."""
    def __init__(self, pairs):
        self.pairs = pairs  # [(method, bid, r_bytes, c_bytes), ...]
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        method, bid, r_bytes, c_bytes = self.pairs[idx]
        r = decode_jpeg(r_bytes); c = decode_jpeg(c_bytes)
        if r.shape != c.shape:
            r = F.interpolate(r.unsqueeze(0), size=(c.shape[1],c.shape[2]),
                              mode='bilinear',align_corners=False).squeeze(0)
        return method, bid, r, c

# Gaussian SSIM kernel
_SSIM_KERNEL = None
def _gk(size=11, sigma=1.5, channels=1):
    coords = torch.arange(size, dtype=torch.float32) - (size-1)/2.
    g = torch.exp(-(coords**2)/(2*sigma**2)); g = g/g.sum()
    return (g[:,None]*g[None,:]).view(1,1,size,size).repeat(channels,1,1,1)

# ============================================================
# Main: one dataset at a time, one GPU
# ============================================================
gpu_id = int(os.environ.get('AR_GPU_ID', 0))
device = torch.device(f'cuda:0')
print(f"[GPU{gpu_id}] Metrics: {sorted(SELECTED_METRICS)}")
if 'ssim' in SELECTED_METRICS:
    print(f"[GPU{gpu_id}] 🔴 SSIM: 标准 GPU 高斯核 (11x11 σ=1.5), 等价 skimage. 旧版简化标量 SSIM 已修正.")

for ds_name in DATASETS:
    print(f"\n[GPU{gpu_id}] === {ds_name} ===")

    # Phase 1: read bytes from tars assigned to this GPU
    clean_dirs = {'DIV2K':'datasets/DIV2K/DIV2K_valid_HR',
                   'LSDIR':'datasets/LSDIR/val1/HR/val'}
    clean_cache = {}
    has_clean = ds_name in clean_dirs and os.path.isdir(clean_dirs[ds_name])
    if has_clean:
        for f in os.listdir(clean_dirs[ds_name]):
            if f.lower().endswith(('.png','.jpg','.jpeg')):
                clean_cache[os.path.splitext(f)[0]] = os.path.join(clean_dirs[ds_name],f)

    my_pairs = []
    for method in METHODS:
        tars = sorted(glob.glob(f'{RES_DIR}/{method}/{ds_name}_gpu*.tar'))
        valid_tars = [tp for tp in tars if os.path.exists(tp)]
        my_tars = [t for i,t in enumerate(valid_tars) if i%NUM_GPUS==gpu_id]
        for tar_path in my_tars:
            with tarfile.open(tar_path,'r') as tar:
                entries = {}
                for info in tar.getmembers():
                    parts=info.name.split('/')
                    if len(parts)<2: continue
                    bid,fname=parts[0],parts[1]; key=(bid,fname.rsplit('.',2)[0])
                    if '.restored.jpg' in fname: entries.setdefault(key,{})['restored']=info
                    elif '.clean.jpg' in fname: entries.setdefault(key,{})['clean']=info
                for key,e in entries.items():
                    if 'restored' not in e: continue
                    if 'clean' in e:
                        r=tar.extractfile(e['restored']).read(); c=tar.extractfile(e['clean']).read()
                    elif has_clean and key[1] in clean_cache:
                        r=tar.extractfile(e['restored']).read()
                        with open(clean_cache[key[1]],'rb') as cf: c=cf.read()
                    else: continue
                    my_pairs.append((method, key[0], r, c))
            print(f"  [GPU{gpu_id}] read {method}/{ds_name}/{os.path.basename(tar_path)}: {len(my_pairs)} total")

    mb = sum(len(r)+len(c) for _,_,r,c in my_pairs)/1024/1024
    print(f"  [GPU{gpu_id}] {ds_name}: {len(my_pairs)} pairs, {mb:.1f} MB")

    # Phase 2: DataLoader parallel decode + GPU compute
    ds = PairDataset(my_pairs)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                          num_workers=8, pin_memory=True,
                                          persistent_workers=False)
    kernel = None
    results = {}
    for method, bid, r, c in loader:
        r, c = r.squeeze(0).to(device), c.squeeze(0).to(device)
        method, bid = method[0], bid[0]
        if kernel is None or kernel.device != device:
            kernel = _gk(11, 1.5, r.shape[0]).to(device)
        k=kernel; C1,C2=0.01**2,0.03**2
        p=r.unsqueeze(0); t=c.unsqueeze(0)
        C=r.shape[0]
        mx=F.conv2d(p,k,groups=C,padding=5); my=F.conv2d(t,k,groups=C,padding=5)
        ssim=((2*mx*my+C1)*(2*(F.conv2d(p*t,k,groups=C,padding=5)-mx*my)+C2))/((mx**2+my**2+C1)*((F.conv2d(p**2,k,groups=C,padding=5)-mx**2)+(F.conv2d(t**2,k,groups=C,padding=5)-my**2)+C2))
        results.setdefault(method,{}).setdefault(bid,{'ssim_sum':0.0,'ssim_count':0})
        results[method][bid]['ssim_sum'] += float(ssim.mean().item())
        results[method][bid]['ssim_count'] += 1

    # Save
    for method in results:
        out_dir=f'{RES_DIR}/{method}'; os.makedirs(out_dir,exist_ok=True)
        out=f'{out_dir}/metrics_{ds_name}_gpu{gpu_id}.json'
        json.dump({ds_name:{method:results[method]}},open(out,'w'),indent=2)
        n=sum(v['ssim_count'] for v in results[method].values())
        print(f"  [GPU{gpu_id}] {method}/{ds_name}: {n} SSIM → {out}")
    del my_pairs, ds, loader  # free memory

print(f"[GPU{gpu_id}] Done.")
