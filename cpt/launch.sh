#!/usr/bin/env bash
# Continued pretraining on one 8-GPU node. Override any variable from the environment, e.g.
#   MODEL=/models/Qwen-32B-base DATA="/data/packed/domain:0.8 /data/packed/general:0.2" \
#   OUT=/nfs/ckpt/cpt-32b TRAIN_TOKENS=1e9 bash cpt/launch.sh
# Smoke test first: MAX_STEPS=5 SAVE_EVERY=5 bash cpt/launch.sh
set -euo pipefail

: "${MODEL:?set MODEL to the HF base model directory}"
: "${DATA:?set DATA to packed data prefixes, e.g. /data/packed/domain:0.8 /data/packed/general:0.2}"
: "${OUT:?set OUT to a checkpoint directory on NFS}"
TRAIN_TOKENS=${TRAIN_TOKENS:-1e9}
NGPU=${NGPU:-8}
SEQ_LEN=${SEQ_LEN:-4096}
MICRO_BATCH=${MICRO_BATCH:-4}
GLOBAL_BATCH_TOKENS=${GLOBAL_BATCH_TOKENS:-2097152}
LR=${LR:-2e-5}
SAVE_EVERY=${SAVE_EVERY:-200}
MAX_STEPS=${MAX_STEPS:-0}
EXTRA_ARGS=${EXTRA_ARGS:---cpu-offload}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}

mkdir -p "$OUT"
# shellcheck disable=SC2086
torchrun --standalone --nproc_per_node "$NGPU" "$(dirname "$0")/train_fsdp.py" \
    --model "$MODEL" --data $DATA --out "$OUT" \
    --train-tokens "$TRAIN_TOKENS" --seq-len "$SEQ_LEN" --micro-batch "$MICRO_BATCH" \
    --global-batch-tokens "$GLOBAL_BATCH_TOKENS" --lr "$LR" \
    --save-every "$SAVE_EVERY" --max-steps "$MAX_STEPS" $EXTRA_ARGS \
    2>&1 | tee -a "$OUT/train.log"
