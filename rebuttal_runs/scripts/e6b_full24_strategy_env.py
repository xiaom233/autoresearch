#!/usr/bin/env python3
"""Print shell env assignments for an E6-B training_strategy.json."""
import json
import sys

d = json.load(open(sys.argv[1]))
a = d.get("architecture", d)


def b(key, default=False):
    v = a.get(key, default)
    return 1 if v else 0


pairs = {
    "AR_ATTENTION_TYPE": a.get("attention_type", "swin"),
    "AR_EMBED_DIM": int(a.get("embed_dim", 64)),
    "AR_WINDOW_SIZE": int(a.get("window_size", 8)),
    "AR_USE_SIMPLE_GATE": b("use_simple_gate", True),
    "AR_USE_GCM": b("use_gcm"),
    "AR_USE_COLOR_PRE": b("use_color_pre"),
    "AR_USE_COLOR_MLP": b("use_color_mlp"),
    "AR_USE_DUAL_BRANCH": b("use_dual_branch"),
    "AR_USE_CSN": b("use_csn"),
    "AR_USE_CHANNEL_CURVE": b("use_channel_curve"),
    "AR_USE_PCP": b("use_pcp"),
    "AR_USE_SCA": b("use_sca"),
    "AR_USE_GDFN": b("use_gdfn"),
    "AR_USE_FPRO": b("use_fpro"),
    "AR_USE_FREQMOD": b("use_freqmod"),
    "AR_LEARNING_RATE": d.get("lr", 0.0005),
    "AR_EPOCH_BUDGET": int(d.get("epoch_budget", 2)),
    "AR_BATCH_SIZE": int(d.get("batch_size", 16)),
}
print(" ".join(f"{k}={v}" for k, v in pairs.items()))
