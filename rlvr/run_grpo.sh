#!/usr/bin/env bash
# GRPO/DAPO-style RLVR baseline on one 8-GPU node (verl 0.9 + vLLM). Override from the env, e.g.
#   MODEL=/nfs/models/Qwen3-1.7B-Base DATA=/nfs/data/rlvr OUT=/nfs/ckpt/rlvr/grpo-1.7b-s1 bash rlvr/run_grpo.sh
# Smoke test: SMOKE=1 MODEL=... DATA=... OUT=... bash rlvr/run_grpo.sh
# Extra Hydra overrides can be appended as arguments.
set -euo pipefail

: "${MODEL:?set MODEL to the HF base model directory, e.g. Qwen3-1.7B-Base}"
: "${DATA:?set DATA to the output dir of rlvr/prepare_data.py}"
: "${OUT:?set OUT to a checkpoint/log directory on NFS}"
HERE=$(cd "$(dirname "$0")" && pwd)
NGPU=${NGPU:-8}
SEED=${SEED:-1}
N=${N:-16}                          # rollouts per prompt (group size)
BATCH=${BATCH:-128}                 # prompts per step -> BATCH*N rollouts
MINI=${MINI:-64}                    # prompts per PPO mini-batch -> BATCH/MINI updates per step
LR=${LR:-1e-6}
MAX_RESP=${MAX_RESP:-4096}
STEPS=${STEPS:-500}
TP=${TP:-1}                         # vLLM tensor parallel; 1 for <=8B on 85 GB cards
GPU_MEM=${GPU_MEM:-0.6}
TEST_FREQ=${TEST_FREQ:-25}
SAVE_FREQ=${SAVE_FREQ:-100}
VAL_N=${VAL_N:-8}                   # samples per val problem (reported as mean@VAL_N)
NAME=${NAME:-$(basename "$OUT")}
VAL_BEFORE=True

if [[ "${SMOKE:-0}" == "1" ]]; then
    BATCH=16; MINI=16; N=8; MAX_RESP=1024; STEPS=3; TEST_FREQ=3; SAVE_FREQ=3; VAL_N=1; VAL_BEFORE=False
fi

export VLLM_USE_V1=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA/train.parquet" \
    data.val_files="$DATA/val.parquet" \
    data.prompt_key=prompt \
    data.train_batch_size="$BATCH" \
    data.max_prompt_length=1024 \
    data.max_response_length="$MAX_RESP" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.seed="$SEED" \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="$LR" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$MINI" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(( (1024 + MAX_RESP) * 4 )) \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n="$N" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM" \
    actor_rollout_ref.rollout.max_num_batched_tokens=$(( 1024 + MAX_RESP )) \
    actor_rollout_ref.rollout.seed="$SEED" \
    actor_rollout_ref.rollout.val_kwargs.n="$VAL_N" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    reward.custom_reward_function.path="$HERE/reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.n_gpus_per_node="$NGPU" \
    trainer.nnodes=1 \
    trainer.total_training_steps="$STEPS" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.val_before_train="$VAL_BEFORE" \
    trainer.project_name=rlvr-diversity \
    trainer.experiment_name="$NAME" \
    trainer.logger='["console","tensorboard"]' \
    trainer.default_local_dir="$OUT" \
    "$@" 2>&1 | tee -a "$OUT/train.log"
