#!/usr/bin/env python3
"""统一 GT 重评估脚本 — 按数据集分别报告 (DIV2K + LSDIR)

用法:
  .venv/bin/python3 scripts/reeval.py exp36 v2 0    # GPU0 评估 exp36 v2 模型
  .venv/bin/python3 scripts/reeval.py exp36 R1 0    # GPU0 评估 exp36 R1 模型

输出: expN/results/reeval_{EXP}_{GPU}.json
      {bid: {DIV2K: {psnr_rgb}, LSDIR: {psnr_rgb}}}
"""
import json, os, sys, glob, torch
sys.path.insert(0, '.')

EXP = sys.argv[1]
GPU = int(sys.argv[3])
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)

from train import evaluate_all, ValDataset, RestoreNet

MAPPING = f'.gt_mappings/{EXP}_mapping.json'
gt_map = json.load(open(MAPPING))
device = torch.device('cuda:0')
amp = torch.amp.autocast('cuda', dtype=torch.bfloat16)

VAL_SETS = {
    'DIV2K': ['datasets/DIV2K/DIV2K_valid_HR'],
    'LSDIR': ['datasets/LSDIR/val1/HR/val'],
}

all_bids = sorted(gt_map.keys())
my_bids = [all_bids[i] for i in range(len(all_bids)) if i % 8 == GPU]

# Checkpoint search: scan all prefixes, prefer exact match with MODEL_TYPE
# MODEL_TYPE is v2/Ft/R1/v1 — determines priority order
MODEL_TYPE = sys.argv[2] if len(sys.argv) > 2 else 'v2'
PREFIX_PRIORITY = {
    'R1': ['exp36_R1', 'exp36_v2', 'exp36_Ft', 'exp36_v1'],
    'v2': ['exp36_v2', 'exp36_R1', 'exp36_Ft', 'exp36_v1'],
    'Ft': ['exp36_Ft', 'exp36_v2', 'exp36_R1', 'exp36_v1'],
    'v1': ['exp36_v1', 'exp36_v2', 'exp36_R1', 'exp36_Ft'],
}.get(MODEL_TYPE, [f'{EXP}_v2', f'{EXP}_R1', f'{EXP}_Ft', f'{EXP}_v1'])

results = {}
for bid in my_bids:
    ckpt_dir = None; logf = None
    for prefix in PREFIX_PRIORITY:
        d = f'{EXP}/experiments/{prefix}_{bid}/checkpoints'
        if os.path.isdir(d) and glob.glob(f'{d}/*15092.pt'):
            ckpt_dir = d; logf = f'{EXP}/logs/{prefix}_{bid}.log'; break

    if not ckpt_dir:
        print(f'[{GPU}] {bid}: NO CHECKPOINT')
        continue

    pts = sorted(glob.glob(f'{ckpt_dir}/*15092.pt'))

    # Extract arch from log
    arch = {'attention_type': 'swin', 'window_size': 8, 'dual_branch': 0, 'color_pre': 0, 'gcm': 0}
    if logf and os.path.exists(logf):
        with open(logf, errors='ignore') as f:
            for line in f:
                if 'EXP_META:' in line:
                    try:
                        mo = json.loads(line.split('EXP_META: ')[1].split(' ===')[0])['model']
                        for k in arch:
                            if k in ['dual_branch', 'color_pre', 'gcm']:
                                arch[k] = int(bool(mo.get(k, 0)))
                            else:
                                arch[k] = mo.get(k, arch[k])
                    except: pass; break

    model = RestoreNet(in_ch=3, embed_dim=64, attention_type=arch['attention_type'],
                       window_size=arch['window_size'], use_dual_branch=arch['dual_branch'],
                       use_color_pre=arch['color_pre'], use_gcm=arch['gcm']).to(device)
    ck = torch.load(pts[-1], map_location=device, weights_only=True)
    model.load_state_dict(ck['model']); model.eval()

    per_ds = {}
    for ds_name, ds_dirs in VAL_SETS.items():
        if not all(os.path.isdir(d) for d in ds_dirs): continue
        tmp = f'/tmp/reeval_{EXP}_{bid}_{ds_name}.json'
        json.dump({'pipeline': gt_map[bid]['pipeline']}, open(tmp, 'w'))
        vs = [(ds_name, ValDataset(ds_dirs, count=0, params_path=tmp))]
        _, ov = evaluate_all(model, vs, device, amp)
        os.remove(tmp)
        per_ds[ds_name] = {'psnr_rgb': round(ov['psnr_rgb'], 2), 'psnr_y': round(ov.get('psnr_y', 0), 2)}

    results[bid] = per_ds
    d2 = per_ds.get('DIV2K', {}).get('psnr_rgb', 0)
    ls = per_ds.get('LSDIR', {}).get('psnr_rgb', 0)
    print(f'[{GPU}] {bid} ({MODEL_TYPE}): DIV2K={d2:.2f} LSDIR={ls:.2f}')

json.dump(results, open(f'{EXP}/results/reeval_{EXP}_{GPU}.json', 'w'), indent=2)
