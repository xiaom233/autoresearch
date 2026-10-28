"""Measure single model efficiency. Usage: python measure_one.py METHOD"""
import sys, os, time, csv, argparse
import torch, torch.nn.functional as F
import numpy as np
from thop import profile

parser = argparse.ArgumentParser()
parser.add_argument('method', type=str)
args = parser.parse_args()
METHOD = args.method

sys.path.insert(0, '/data/zyli/projects/autoresearch')

device = torch.device('cuda:0')
x = torch.randn(1, 3, 1024, 1024).to(device)
WARMUP, RUNS = 3, 10

def measure(model):
    model.eval()
    # MACs
    macs, params = profile(model, inputs=(x,), verbose=False)

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(x)
        torch.cuda.synchronize()
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
    avg_t = sum(times[1:-1]) / (RUNS - 2) * 1000  # ms
    return params, macs / 1e9, avg_t  # MACs in G

if METHOD == 'X-Restormer':
    sys.path.insert(0, 'benchmark/X-Restormer')
    from xrestormer.archs.xrestormer_arch import XRestormer
    m = XRestormer(dim=48, num_blocks=[2,4,4,4], window_size=8, spatial_heads=[1,2,4,8]).to(device)

elif METHOD == 'AirNet':
    sys.path.insert(0, 'benchmark/AirNet')
    from net.model import AirNet
    import argparse as ap; opt = ap.Namespace(); opt.batch_size = 5
    m = AirNet(opt).to(device)
    m._orig_forward = m.forward
    m.forward = lambda x: m._orig_forward(x, x)

elif METHOD == 'PromptIR':
    sys.path.insert(0, 'benchmark/PromptIR')
    from net.model import PromptIR
    m = PromptIR(decoder=True, dim=48, num_blocks=[4,6,6,8], heads=[1,2,4,8],
                 ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias',
                 num_refinement_blocks=4).to(device)

elif METHOD == 'DFPIR':
    sys.path.insert(0, 'benchmark/DFPIR')
    from net.model import ChannelShuffle_skip_textguaid
    m = ChannelShuffle_skip_textguaid(device=device).to(device)
    tc = torch.randn(1, 512).to(device)
    # Wrap to accept single input
    class DFPIRWrapper(torch.nn.Module):
        def __init__(self, model, text_code):
            super().__init__()
            self.model = model
            self.text_code = text_code
        def forward(self, x):
            return self.model(x, self.text_code.expand(x.shape[0], -1))
    m = DFPIRWrapper(m, tc)

elif METHOD == 'R2R':
    sys.path.insert(0, 'benchmark/R2R')
    from net.model_5D import R2RLocal
    m = R2RLocal(ckpt_path='benchmark/R2R/ckpt/train_ckpt_5D_f/',
                 prompts_path='benchmark/R2R/prompt-20260622T130433Z-3-001/prompt/save_prompts_5D',
                 prompts_name='last', train_mode='finetune').to(device)

elif METHOD == 'DCPT':
    sys.path.insert(0, 'benchmark/PromptIR')
    from net.model import PromptIR as DCPTModel
    m = DCPTModel(decoder=True, dim=48, num_blocks=[4,6,6,8], heads=[1,2,4,8],
                  ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias',
                  num_refinement_blocks=4).to(device)

elif METHOD == 'DATPRL-IR':
    sys.path.insert(0, 'benchmark/DATPRL-IR/All_in_One')
    from net.promptrestormerv3_arch import DualPromptRestormerv3_only_task_prompt
    m = DualPromptRestormerv3_only_task_prompt().to(device)

elif METHOD == 'exp40-Best':
    from train import RestoreNet
    m = RestoreNet(in_ch=3, embed_dim=64, use_simple_gate=True).to(device)

else:
    print(f"Unknown: {METHOD}"); sys.exit(1)

p, macs, t = measure(m)
print(f'{METHOD},{p/1e6:.2f},{macs:.2f},{t:.1f}')
