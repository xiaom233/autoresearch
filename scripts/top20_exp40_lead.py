"""Find top 20 images where exp40 leads most in non-global challenges."""
import json, os, tarfile, io, glob, torch
import numpy as np
from PIL import Image
from collections import defaultdict

RES_DIR = 'benchmark_results'
DEVICE = torch.device('cuda:0')

# Non-global challenges from exp42 GT
non_global = ['blind_0001','blind_0003','blind_0004','blind_0005','blind_0006',
              'blind_0007','blind_0011','blind_0012','blind_0013','blind_0016',
              'blind_0017','blind_0022']

# All methods to compare
all_methods = ['xrestormer','airnet','promptir','dfpir','r2r','dcpt',
               'datprl-ir','n2n','n2n_ft','dip','mambairv2','agenticir',
               'dfpir_ft','exp40_best']
exp40_method = 'exp40_best'
other_methods = [m for m in all_methods if m != exp40_method]

def load_jpg(data):
    return torch.from_numpy(np.array(Image.open(io.BytesIO(data))).astype(np.float32)/255.0).permute(2,0,1).to(DEVICE)

# Pre-load clean images
clean_cache = {}
for ds_name, ds_path in [('DIV2K','datasets/DIV2K/DIV2K_valid_HR')]:
    for f in os.listdir(ds_path):
        if f.lower().endswith(('.png','.jpg','.jpeg')):
            name = os.path.splitext(f)[0]
            img = Image.open(os.path.join(ds_path, f))
            clean_cache[name] = torch.from_numpy(np.array(img).astype(np.float32)/255.0).permute(2,0,1).to(DEVICE)

# Read all methods' restored images for non-global challenges
# Structure: {challenge: {method: {image_name: restored_tensor}}}
data = defaultdict(lambda: defaultdict(dict))

for method in all_methods:
    for ds_name in ['DIV2K']:  # DIV2K only
        tars = sorted(glob.glob(f'{RES_DIR}/{method}/{ds_name}_gpu*.tar'))
        for tar_path in tars:
            try:
                with tarfile.open(tar_path, 'r') as tar:
                    entries = {}
                    for info in tar.getmembers():
                        parts = info.name.split('/')
                        if len(parts) < 2: continue
                        bid, fname = parts[0], parts[1]
                        if bid not in non_global: continue
                        key = (bid, fname.rsplit('.', 2)[0])
                        if '.restored.jpg' in fname:
                            entries.setdefault(key, {})['restored'] = info
                        elif '.clean.jpg' in fname:
                            entries.setdefault(key, {})['clean'] = info

                    for key, e in entries.items():
                        if 'restored' not in e: continue
                        bid, img_name = key
                        restored = load_jpg(tar.extractfile(e['restored']).read())
                        if img_name in clean_cache:
                            # Compute PSNR
                            clean = clean_cache[img_name]
                            r, c = restored, clean
                            if r.shape != c.shape:
                                r = torch.nn.functional.interpolate(
                                    r.unsqueeze(0), size=(c.shape[1],c.shape[2]),
                                    mode='bilinear',align_corners=False).squeeze(0)
                            mse = (r - c).pow(2).mean()
                            psnr = 10 * torch.log10(1.0/mse).item() if mse > 0 else 100.0
                            data[bid][method][img_name] = psnr
                print(f"  {method}/{ds_name}/{os.path.basename(tar_path)}: {sum(len(v) for v in data[bid].values()) if bid in data else 0} images")
            except Exception as ex:
                print(f"  {method}/{ds_name}/{os.path.basename(tar_path)}: ERROR {ex}")

# Compute exp40 lead per image
leads = []  # [(challenge, image_name, exp40_psnr, best_other_psnr, lead)]
for bid in non_global:
    if bid not in data or exp40_method not in data[bid]:
        continue
    for img_name, exp40_psnr in data[bid][exp40_method].items():
        # Best PSNR among other methods for this image
        best_other = 0
        best_method = ''
        for method in other_methods:
            if method in data[bid] and img_name in data[bid][method]:
                p = data[bid][method][img_name]
                if p > best_other:
                    best_other = p
                    best_method = method
        if best_other > 0:
            lead = exp40_psnr - best_other
            leads.append((bid, img_name, exp40_psnr, best_other, best_method, lead))

# Sort by lead descending
leads.sort(key=lambda x: -x[5])

# Top 20
print(f"\n=== Top 20 images where exp40 leads most (non-global, DIV2K) ===")
print(f"{'Rank':<6} {'Challenge':<14} {'Image':<10} {'exp40 PSNR':>10} {'Best Other':>12} {'Method':<15} {'Lead':>8}")
print("-" * 80)
for i, (bid, img, e40, best, bm, lead) in enumerate(leads[:20]):
    print(f"{i+1:<6} {bid:<14} {img:<10} {e40:>10.2f} {best:>12.2f} {bm:<15} {lead:>+8.2f}")

# Save
out = 'benchmark_results/top20_exp40_lead_non_global.json'
json.dump([{'rank':i+1,'challenge':b,'image':img,'exp40_psnr':e40,'best_other_psnr':best,'best_method':bm,'lead':lead}
           for i,(b,img,e40,best,bm,lead) in enumerate(leads[:20])], open(out,'w'), indent=2)
print(f"\nSaved: {out}")

# Also save all data
all_out = 'benchmark_results/top20_exp40_lead_non_global_all.json'
json.dump([{'challenge':b,'image':img,'exp40_psnr':e40,'best_other_psnr':best,'best_method':bm,'lead':lead}
           for b,img,e40,best,bm,lead in leads], open(all_out,'w'), indent=2)
print(f"All {len(leads)} images saved: {all_out}")
