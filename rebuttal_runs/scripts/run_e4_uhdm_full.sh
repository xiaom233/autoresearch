#!/bin/bash
cd /data/zyli/projects/autoresearch
PY=/home/zyli/anaconda3/envs/dfpir/bin/python
META=rebuttal_runs/E4/meta_swin_sg.json
OUT=rebuttal_runs/E4/uhdm/full_eval
PAIRS=datasets/E4/uhdm/test_full
mkdir -p "$OUT"

run_restore(){ # name ckpt out gpu meta(optional 1/0)
  local name=$1 ckpt=$2 out=$3 gpu=$4 usemeta=$5
  local extra=""
  if [ "$usemeta" = "1" ]; then extra="--meta $META"; fi
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PY rebuttal_runs/scripts/eval_paired.py \
    --pairs-dir "$PAIRS" --model-kind restorenet --checkpoint "$ckpt" \
    $extra --controller "$name" --scenario uhdm-full --out "$out" \
    --tile 512 --overlap 32 --gpu-id "$gpu" --shard-id "$gpu" --num-gpus 8 \
    > "$OUT/${name}_gpu${gpu}.log" 2>&1
}

run_blinded(){ # gpu
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PY rebuttal_runs/scripts/eval_paired.py \
    --pairs-dir "$PAIRS" --model-kind dfpir \
    --checkpoint resource/Degradation-Aware-Feature-Perturbation-for-All-in-One-Image-Restoration-Modified-Version-/dfpir_blind/checkpoints/dfpir_blind_step301920.pt \
    --controller DFPIR-blinded --scenario uhdm-full --out "$OUT/DFPIR-blinded_gpu$1.jsonl" \
    --tile 512 --overlap 64 --gpu-id "$1" --shard-id "$1" --num-gpus 8 \
    > "$OUT/DFPIR-blinded_gpu$1.log" 2>&1
}

run_official(){ # gpu
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PY rebuttal_runs/scripts/eval_dfpir_official.py \
    --pairs-dir "$PAIRS" \
    --checkpoint benchmark/DFPIR/DFPIR-5D-pn31.29-0.8889_pr37.62-0.9779_ph31.64-0.9794_pb28.82-0.8734_pl23.82-0.8428_avr30.64-0.9125.pth.tar \
    --controller DFPIR-official --scenario uhdm-full --out "$OUT/DFPIR-official_gpu$1.jsonl" \
    --tile 1024 --gpu-id "$1" --shard-id "$1" --num-gpus 8 \
    > "$OUT/DFPIR-official_gpu$1.log" 2>&1
}

echo "M_blind start $(date)"
for g in 0 1 2 3 4 5 6 7; do
  run_restore M_blind resource/blind_pretrain/checkpoints/blind_pretrain_step15092.pt "$OUT/M_blind_gpu$g.jsonl" "$g" 0 &
done
wait

echo "E4-UHDM-rule start $(date)"
for g in 0 1 2 3 4 5 6 7; do
  run_restore E4-UHDM-rule rebuttal_runs/E4/uhdm/rule_R0/checkpoints/rule_R0_step11319.pt "$OUT/E4-UHDM-rule_gpu$g.jsonl" "$g" 1 &
done
wait

echo "DFPIR-blinded start $(date)"
for g in 0 1 2 3 4 5 6 7; do run_blinded "$g" & done
wait

echo "DFPIR-official start $(date)"
for g in 0 1 2 3 4 5 6 7; do run_official "$g" & done
wait

echo "ALL DONE $(date)"
