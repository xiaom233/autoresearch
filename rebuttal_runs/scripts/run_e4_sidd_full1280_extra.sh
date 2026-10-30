#!/bin/bash
# Evaluate the 160 SIDD validation blocks missing from the 1,120 test set.
# Existing 1,120 rows + these 160 = full 1,280 (40 scenes x 32).
set -u
cd /data/zyli/projects/autoresearch || exit 1
PAIRS=datasets/E4/sidd/test_support_scenes
OUT=rebuttal_runs/E4/sidd/full_eval_1280_extra
SCEN=sidd-full1280
mkdir -p "$OUT"
DFPIR=/home/zyli/anaconda3/envs/dfpir/bin/python
XREST=/home/zyli/anaconda3/envs/xrestormer/bin/python
R2R=/home/zyli/anaconda3/envs/r2r/bin/python
AIRNET=/home/zyli/anaconda3/envs/airnet/bin/python
DATPRL=/home/zyli/anaconda3/envs/datprl-ir/bin/python
GPUS=0,1,2,3,4,5,6,7
META=rebuttal_runs/E4/meta_swin_sg.json

run_restore() { # name ckpt use_meta
  local name=$1 ckpt=$2 metaflag=$3
  local extra=""
  [ "$metaflag" = "1" ] && extra="--meta $META"
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$GPUS $DFPIR rebuttal_runs/scripts/eval_paired.py \
      --pairs-dir "$PAIRS" --model-kind restorenet --checkpoint "$ckpt" $extra \
      --controller "$name" --scenario "$SCEN" --out "$OUT/${name}_gpu$g.jsonl" \
      --tile 0 --gpu-id "$g" --shard-id "$g" --num-gpus 8 \
      > "$OUT/${name}_gpu$g.log" 2>&1 &
  done
  wait
  echo "[$(date -u +%FT%TZ)] $name done"
}

run_simple() { # name env script ckpt [extra args...]
  local name=$1 py=$2 script=$3 ckpt=$4; shift 4
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$GPUS "$py" "rebuttal_runs/scripts/$script" \
      --pairs-dir "$PAIRS" --checkpoint "$ckpt" --controller "$name" \
      --scenario "$SCEN" --out "$OUT/${name}_gpu$g.jsonl" \
      --gpu-id "$g" --shard-id "$g" --num-gpus 8 "$@" \
      > "$OUT/${name}_gpu$g.log" 2>&1 &
  done
  wait
  echo "[$(date -u +%FT%TZ)] $name done"
}

run_restore M_blind resource/blind_pretrain/checkpoints/blind_pretrain_step15092.pt 0
run_restore E4-SIDD-skill rebuttal_runs/E4/sidd/skill_R0/checkpoints/skill_R0_step7546.pt 1
run_restore E4-SIDD-rule rebuttal_runs/E4/sidd/rule_R0/checkpoints/rule_R0_step3773.pt 1

# blinded DFPIR: tile512 / overlap64
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$GPUS $DFPIR rebuttal_runs/scripts/eval_paired.py \
    --pairs-dir "$PAIRS" --model-kind dfpir \
    --checkpoint resource/Degradation-Aware-Feature-Perturbation-for-All-in-One-Image-Restoration-Modified-Version-/dfpir_blind/checkpoints/dfpir_blind_step301920.pt \
    --controller DFPIR-blinded --scenario "$SCEN" --out "$OUT/DFPIR-blinded_gpu$g.jsonl" \
    --tile 512 --overlap 64 --gpu-id "$g" --shard-id "$g" --num-gpus 8 \
    > "$OUT/DFPIR-blinded_gpu$g.log" 2>&1 &
done
wait
echo "[$(date -u +%FT%TZ)] DFPIR-blinded done"

# official DFPIR, primary CLIP prompt sigma=25
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$GPUS $DFPIR rebuttal_runs/scripts/eval_dfpir_official.py \
    --pairs-dir "$PAIRS" \
    --checkpoint benchmark/DFPIR/DFPIR-5D-pn31.29-0.8889_pr37.62-0.9779_ph31.64-0.9794_pb28.82-0.8734_pl23.82-0.8428_avr30.64-0.9125.pth.tar \
    --controller DFPIR-official-clip25 --scenario "$SCEN" \
    --out "$OUT/DFPIR-official-clip25_gpu$g.jsonl" --tile 0 \
    --text-prompt "Gaussian noise with a standard deviation of 25" \
    --gpu-id "$g" --shard-id "$g" --num-gpus 8 \
    > "$OUT/DFPIR-official-clip25_gpu$g.log" 2>&1 &
done
wait
echo "[$(date -u +%FT%TZ)] DFPIR-official-clip25 done"

run_simple X-Restormer "$XREST" eval_xrestormer_paired.py benchmark/X-Restormer/weights/denoise_300k.pth
run_simple PromptIR "$DFPIR" eval_promptir_paired.py benchmark/PromptIR/ckpt/model.ckpt --variant promptir
run_simple DCPT-5D "$DFPIR" eval_promptir_paired.py benchmark/dcpt/weights/dcpt_promptir_5d.pth --variant dcpt
run_simple R2R-5D "$R2R" eval_r2r_paired.py benchmark/R2R/ckpt/train_ckpt_5D_f/last.ckpt
run_simple AirNet "$AIRNET" eval_airnet_paired.py benchmark/AirNet/ckpt/All.pth
run_simple DATPRL-IR-5Task "$DATPRL" eval_datprl_paired.py benchmark/DATPRL-IR/DATPRL-IR_net_g_5task_epoch=131-step=2695780.ckpt
run_simple DATPRL-IR-CDD11 "$DATPRL" eval_datprl_paired.py benchmark/DATPRL-IR/DATPRL-IR_net_g_cdd11.pth

echo "[$(date -u +%FT%TZ)] ALL_EXTRA_DONE"
