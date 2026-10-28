"""Compute model efficiency: params, inference time for 3×1024×1024 input.
10 runs, remove max/min, average. Save to benchmark_results/efficiency.csv
"""
import sys, os, time, csv
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, '/data/zyli/projects/autoresearch')
sys.path.insert(0, '/data/zyli/projects/autoresearch/benchmark')

device = torch.device('cuda:0')
input_shape = (1, 3, 1024, 1024)
WARMUP = 3
RUNS = 10

def measure(model, name):
    """Warmup + 10 runs, remove max/min, return avg time and params."""
    x = torch.randn(*input_shape).to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(x)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    times.sort()
    avg_time = sum(times[1:-1]) / (RUNS - 2)  # remove min and max

    params = sum(p.numel() for p in model.parameters())
    return params, avg_time * 1000  # ms

results = {}

# ── X-Restormer ──────────────────────────────────────
sys.path.insert(0, 'benchmark/X-Restormer')
import xrestormer.archs
from xrestormer.archs.xrestormer_arch import XRestormer
m = XRestormer(dim=48, num_blocks=[2,4,4,4], window_size=8, spatial_heads=[1,2,4,8]).to(device)
p, t = measure(m, 'X-Restormer')
results['X-Restormer'] = (p, t)
print(f'X-Restormer: {p/1e6:.2f}M params, {t:.1f}ms')
del m

# ── AirNet ───────────────────────────────────────────
sys.path.insert(0, 'benchmark/AirNet')
from net.model import AirNet
import argparse
opt = argparse.Namespace(); opt.batch_size = 5
m = AirNet(opt).to(device)
p, t = measure(m, 'AirNet')
results['AirNet'] = (p, t)
print(f'AirNet: {p/1e6:.2f}M params, {t:.1f}ms')
del m

# ── PromptIR ─────────────────────────────────────────
sys.path.insert(0, 'benchmark/PromptIR')
from net.model import PromptIR
m = PromptIR(decoder=True, dim=48, num_blocks=[4,6,6,8], heads=[1,2,4,8],
             ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias',
             num_refinement_blocks=4).to(device)
p, t = measure(m, 'PromptIR')
results['PromptIR'] = (p, t)
print(f'PromptIR: {p/1e6:.2f}M params, {t:.1f}ms')
del m

# ── DFPIR ────────────────────────────────────────────
sys.path.insert(0, 'benchmark/DFPIR')
from net.model import ChannelShuffle_skip_textguaid
m = ChannelShuffle_skip_textguaid(device=device).to(device)
# DFPIR needs text_code
text_code = torch.randn(1, 512).to(device)
def dfpir_forward(x):
    return m(x, text_code.expand(x.shape[0], -1))
m.forward_orig = m.forward
m.forward = dfpir_forward
p, t = measure(m, 'DFPIR')
m.forward = m.forward_orig
results['DFPIR'] = (p, t)
print(f'DFPIR: {p/1e6:.2f}M params, {t:.1f}ms')
del m

# ── R2R ──────────────────────────────────────────────
sys.path.insert(0, 'benchmark/R2R')
from net.model_5D import R2RLocal
m = R2RLocal(
    ckpt_path='benchmark/R2R/ckpt/train_ckpt_5D_f/',
    prompts_path='benchmark/R2R/prompt-20260622T130433Z-3-001/prompt/save_prompts_5D',
    prompts_name='last', train_mode='finetune'
).to(device)
p, t = measure(m, 'R2R')
results['R2R'] = (p, t)
print(f'R2R: {p/1e6:.2f}M params, {t:.1f}ms')
del m

# ── DCPT ─────────────────────────────────────────────
sys.path.insert(0, 'benchmark/dcpt')
# DCPT uses same architecture as PromptIR
from net.model import PromptIR as DCPTModel
m = DCPTModel(decoder=True, dim=48, num_blocks=[4,6,6,8], heads=[1,2,4,8],
              ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias',
              num_refinement_blocks=4).to(device)
p, t = measure(m, 'DCPT')
results['DCPT'] = (p, t)
print(f'DCPT: {p/1e6:.2f}M params, {t:.1f}ms')
del m

# ── DATPRL-IR ────────────────────────────────────────
sys.path.insert(0, 'benchmark/DATPRL-IR/All_in_One')
from net.promptrestormerv3_arch import DualPromptRestormerv3_only_task_prompt
m = DualPromptRestormerv3_only_task_prompt().to(device)
p, t = measure(m, 'DATPRL-IR')
results['DATPRL-IR'] = (p, t)
print(f'DATPRL-IR: {p/1e6:.2f}M params, {t:.1f}ms')
del m

# ── exp40-Best (RestoreNet) ─────────────────────────
# Use the most common architecture: Swin+SG embed_dim=64
from train import RestoreNet
m = RestoreNet(in_ch=3, embed_dim=64, use_simple_gate=True).to(device)
p, t = measure(m, 'exp40-Best')
results['exp40-Best'] = (p, t)
print(f'exp40-Best: {p/1e6:.2f}M params, {t:.1f}ms')
del m

# ── DFPIR-ft (same as DFPIR) ────────────────────────
results['DFPIR-ft'] = results['DFPIR']

# AgenticIR - skip (LLM agent, not measurable)

# ── Save CSV ─────────────────────────────────────────
with open('benchmark_results/efficiency.csv', 'w') as f:
    w = csv.writer(f)
    w.writerow(['Method', 'Params(M)', 'MACs(G)', 'Time(ms)'])
    for name in ['X-Restormer','AirNet','PromptIR','DFPIR','R2R','DCPT',
                 'DATPRL-IR','exp40-Best','DFPIR-ft']:
        p, t = results.get(name, (0, 0))
        # Estimate MACs roughly: 2 * params * pixels for conv-like architectures
        # Skip MACs for now (no thop)
        macs = ''
        w.writerow([name, f'{p/1e6:.2f}', macs, f'{t:.1f}'])

print("\nSaved to benchmark_results/efficiency.csv")
