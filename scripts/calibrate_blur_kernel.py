#!/usr/bin/env python3
"""抗噪模糊核校准实验 — 128组合 × 5 seeds = 640 cases, 16线程并行"""
import json, os, sys, subprocess, numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

sys.path.insert(0, '.')
from x_distortion import add_distortion

BLUR_TYPES = ['blur_gaussian', 'blur_lens', 'blur_motion', 'blur_glass']
BLUR_SEVS = [2, 4]
NOISE_TYPES = ['noise_gaussian_RGB', 'noise_speckle', 'noise_poisson', 'noise_impulse']
NOISE_SEVS = [1, 3]
ORDERS = ['blur_first', 'noise_first']
SEEDS = [42, 123, 456, 789, 1024]

OUT_DIR = '/tmp/blur_calibration'
os.makedirs(OUT_DIR, exist_ok=True)

# Fixed 256x256 clean source
CLEAN_SRC = 'datasets/DIV2K/DIV2K_valid_HR/0801.png'
clean = np.array(Image.open(CLEAN_SRC).convert('RGB'), dtype=np.uint8)
h, w = clean.shape[:2]
clean = clean[h//2-128:h//2+128, w//2-128:w//2+128].copy()
Image.fromarray(clean).save(os.path.join(OUT_DIR, "clean.png"))

# Generate cases
cases = []
for bt in BLUR_TYPES:
    for bs in BLUR_SEVS:
        for nt in NOISE_TYPES:
            for ns in NOISE_SEVS:
                for order in ORDERS:
                    for seed in SEEDS:
                        cases.append({'blur_type':bt,'blur_sev':bs,'noise_type':nt,'noise_sev':ns,'order':order,'seed':seed})
print(f"Cases: {len(cases)}")

print("Generating images...")
for i, c in enumerate(cases):
    np.random.seed(c['seed'])
    img = clean.copy()
    if c['order'] == 'blur_first':
        img = add_distortion(img, severity=c['blur_sev'], distortion_name=c['blur_type'])
        img = add_distortion(img, severity=c['noise_sev'], distortion_name=c['noise_type'])
    else:
        img = add_distortion(img, severity=c['noise_sev'], distortion_name=c['noise_type'])
        img = add_distortion(img, severity=c['blur_sev'], distortion_name=c['blur_type'])
    cid = f"{c['blur_type']}_{c['blur_sev']}__{c['noise_type']}_{c['noise_sev']}__{c['order']}__{c['seed']}"
    Image.fromarray(img).save(os.path.join(OUT_DIR, f"{cid}_degraded.png"))
    if (i+1) % 100 == 0: print(f"  {i+1}/{len(cases)}")
print("Images ready")

# Run calibration
SCRIPT = ".claude/skills/image-degradation-simulator/scripts/blur_kernel_recovery.py"
CLEAN_PATH = os.path.join(OUT_DIR, "clean.png")

def run_case(case):
    cid = f"{case['blur_type']}_{case['blur_sev']}__{case['noise_type']}_{case['noise_sev']}__{case['order']}__{case['seed']}"
    dp = os.path.join(OUT_DIR, f"{cid}_degraded.png")
    try:
        r = subprocess.run(['.venv/bin/python3', SCRIPT, '--target', dp, '--clean', CLEAN_PATH],
                          capture_output=True, text=True, timeout=20)
        bt, conf, pca, flat, dens = "ERROR", "?", 0, 0, 0
        for line in r.stdout.split('\n'):
            if 'Type:' in line: bt = line.split(':')[1].strip()
            if 'Confidence:' in line: conf = line.split(':')[1].strip()
            if '"pca_ratio"' in line:
                try: pca = float(line.split(':')[1].strip().rstrip(','))
                except: pass
            if '"flatness"' in line:
                try: flat = float(line.split(':')[1].strip().rstrip(','))
                except: pass
            if '"density"' in line:
                try: dens = float(line.split(':')[1].strip().rstrip(','))
                except: pass
        return {'case':case,'predicted':bt,'confidence':conf,'pca_ratio':pca,'flatness':flat,'density':dens,'error':None}
    except Exception as e:
        return {'case':case,'predicted':'ERROR','confidence':'?','pca_ratio':0,'flatness':0,'density':0,'error':str(e)}

print("\nRunning (16 threads)...")
results = []
with ThreadPoolExecutor(max_workers=16) as ex:
    futures = {ex.submit(run_case, c): c for c in cases}
    done = 0
    for f in as_completed(futures):
        results.append(f.result()); done += 1
        if done % 50 == 0: print(f"  {done}/{len(cases)}")

# Analysis
def summarize(filter_fn, label):
    subset = [r for r in results if filter_fn(r['case'])]
    n = len(subset)
    if n == 0: return
    is_motion = lambda r: 'blur_motion' in r['case']['blur_type']
    mc = sum(1 for r in subset if is_motion(r) and 'motion' in r['predicted'])
    mt = sum(1 for r in subset if is_motion(r))
    correct = sum(1 for r in subset if r['case']['blur_type'] in r['predicted'])
    pca_by_type = defaultdict(list)
    for r in subset: pca_by_type[r['case']['blur_type']].append(r['pca_ratio'])
    print(f"\n{label} (n={n}):")
    print(f"  Overall: {correct}/{n} = {correct/n*100:.1f}%")
    print(f"  Motion:  {mc}/{mt} = {mc/max(mt,1)*100:.1f}%")
    for bt in BLUR_TYPES:
        pcas = pca_by_type.get(bt, [])
        if pcas: print(f"  {bt}: pca={np.mean(pcas):.2f}±{np.std(pcas):.2f}, kernel_ok={sum(1 for p in pcas if p>1.1)}/{len(pcas)}")

summarize(lambda c: True, "ALL")
summarize(lambda c: c['noise_sev']==1, "Noise sev=1")
summarize(lambda c: c['noise_sev']==3, "Noise sev=3")
summarize(lambda c: c['order']=='blur_first', "Blur→Noise")
summarize(lambda c: c['order']=='noise_first', "Noise→Blur")
summarize(lambda c: c['blur_sev']==2, "Blur sev=2")
summarize(lambda c: c['blur_sev']==4, "Blur sev=4")

# Lens vs Gaussian
lg = [r for r in results if r['case']['blur_type'] in ['blur_lens','blur_gaussian'] and r['case']['blur_sev']==4]
lf = [r['flatness'] for r in lg if 'blur_lens' in r['case']['blur_type']]
gf = [r['flatness'] for r in lg if 'blur_gaussian' in r['case']['blur_type']]
if lf and gf:
    print(f"\nLens vs Gaussian (sev=4): lens_flat={np.mean(lf):.3f}±{np.std(lf):.3f}, gauss_flat={np.mean(gf):.3f}±{np.std(gf):.3f}")

json.dump([{**r['case'],'predicted':r['predicted'],'pca_ratio':r['pca_ratio'],'flatness':r['flatness'],'density':r['density']} for r in results],
          open(os.path.join(OUT_DIR,'results.json'),'w'), indent=2)
print(f"\nSaved: {OUT_DIR}/results.json")
