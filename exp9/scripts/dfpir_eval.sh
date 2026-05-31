#!/bin/bash
# DFPIR 大模型基准评估
# 用法: bash exp9/scripts/dfpir_eval.sh <params.json> [gpus] [output.json]
# 示例: bash exp9/scripts/dfpir_eval.sh exp9/degradation/D1_dual.json
#       bash exp9/scripts/dfpir_eval.sh params.json 0,1,2,3 results.json

PARAMS="${1:?Usage: $0 <params.json> [gpus] [output.json]}"
GPUS="${2:-0}"
OUTPUT="${3:-dfpir_result.json}"

CKPT="resource/Degradation-Aware-Feature-Perturbation-for-All-in-One-Image-Restoration-Modified-Version-/dfpir_blind/checkpoints/dfpir_blind_step301920.pt"
SCRIPT="resource/Degradation-Aware-Feature-Perturbation-for-All-in-One-Image-Restoration-Modified-Version-/test_degradation.py"

echo "=== DFPIR Evaluation ==="
echo "Params: $PARAMS"
echo "GPUs: $GPUS"
echo "Output: $OUTPUT"

conda run -n dfpir python "$SCRIPT" --params "$PARAMS" --gpus "$GPUS" --output "$OUTPUT" 2>&1 | tail -5

if [ -f "$OUTPUT" ]; then
    PSNR=$(python3 -c "import json; d=json.load(open('$OUTPUT')); print(f\"{d['overall']['psnr_rgb']:.2f}\")")
    echo "DFPIR PSNR: $PSNR"
fi
