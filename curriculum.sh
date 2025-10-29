#!/bin/bash
#SBATCH -A a135
#SBATCH --job-name=grpo-staged-grounding
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --time=48:00:00  # Increased time for more epochs
#SBATCH --output=job_outputs/%x_%j.out
#SBATCH --exclusive
#SBATCH --exclude=nid[006569,006601,006609,006622-006623,006628-006629,006632,006638,006651,006653-006655,006658-06662,006664-006665,006669-006671,006674-006677]

srun --environment=pytorch-venv --gpus-per-task=4 bash -c "
export SLURM_CPUS_PER_TASK=288
export SLURM_GPUS=4
echo \${SLURM_PROCID}

unset ROCR_VISIBLE_DEVICES
set -x
ENGINE=\${1:-vllm}

export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_MULTIPROC=1

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_NCCL_SO_PATH=/usr/lib/aarch64-linux-gnu/

export WORK_DIR=/iopsstor/scratch/cscs/rfahrni/code/verl
export DATA_DIR=/iopsstor/scratch/cscs/rfahrni/data/grpo_medical_grounding_curriculum
export VALID_FILE=/iopsstor/scratch/cscs/rfahrni/data/grpo_medical_grounding_single_staged/valid.parquet
cd \$WORK_DIR

# Use a single base path but keep stage subdirectories for organization
export BASE_SAVE_PATH=\$SCRATCH/checkpoints/grpo_staged_curriculum_continuous
export WANDB_DIR=\${BASE_SAVE_PATH}

if [ ! -d \"\$BASE_SAVE_PATH\" ]; then
    mkdir -p \$BASE_SAVE_PATH
fi

NODES=\$(scontrol show hostnames \"\$SLURM_JOB_NODELIST\" | tr \"\n\" \" \" | xargs)
NODES_ARR=(\$NODES)
MASTER_NODE=\${NODES_ARR[0]}
REWARD_NODE=\${NODES_ARR[1]}

export MASTER_NODE_IP=\$(srun --overlap --nodes=1 --ntasks=1 -w \"\$MASTER_NODE\" hostname --ip-address)
export REWARD_NODE_IP=\$(srun --overlap --nodes=1 --ntasks=1 -w \"\$REWARD_NODE\" hostname --ip-address)

echo \"master node ip \${MASTER_NODE_IP}\"
echo \"reward node ip \${REWARD_NODE_IP}\"

export PORT=\$((6542 + \$SLURM_JOB_ID % 1000))
export RAY_ADDRESS=\"\${MASTER_NODE_IP}:\${PORT}\"
echo \"ray address \${RAY_ADDRESS}\"

export WANDB_RESUME=allow
export WANDB_ENTITY=\"krauthammerlab\"
export WANDB_API_KEY=15b5344c70fad59908246ded2a98fdef6a4e9eda

# Install required packages
pip install evaluate
pip install \"transformers==4.52.4\" \"tokenizers==0.20.1\" \"accelerate==0.34.2\" -U
pip install scipy

if [[ \$SLURM_PROCID -eq 0 ]]; then
  echo \"starting head\"
  echo \"master node ip \${MASTER_NODE_IP}\"
  ray start --head --node-ip-address=\$MASTER_NODE_IP --port=\$PORT --num-cpus=\$SLURM_CPUS_PER_TASK --num-gpus=\$SLURM_GPUS --block &
else
  echo \"starting non head\"
  echo \"master node ip \${MASTER_NODE_IP}\"
  ray start --address=\$RAY_ADDRESS --num-cpus=\$SLURM_CPUS_PER_TASK --num-gpus=\$SLURM_GPUS --block &
fi

sleep 20

ray status

if [[ \$SLURM_PROCID -eq 0 ]]; then

# ============================================================================
# CURRICULUM TRAINING: 3 STAGES WITH CONTINUOUS CHECKPOINTING
# ============================================================================

echo \"\"
echo \"================================================================================\"
echo \"  STARTING CURRICULUM GRPO TRAINING (CONTINUOUS)\"
echo \"  Stage 1: Easy samples (5 epochs warmup)\"
echo \"  Stage 2: Easy + Medium samples (5 epochs) - RESUMES from Stage 1\"
echo \"  Stage 3: All difficulties (10 epochs) - RESUMES from Stage 2\"
echo \"================================================================================\"
echo \"\"

# Define base model and common hyperparameters
BASE_MODEL=\"/capstor/store/cscs/swissai/a135/RadVLM_project/models/qwen2.5VL_full\"

# ============================================================================
# STAGE 1: EASY SAMPLES ONLY (Warmup Phase - 5 epochs)
# ============================================================================

echo \"\"
echo \"================================================================================\"
echo \"  STAGE 1: TRAINING ON EASY SAMPLES (5 EPOCHS)\"
echo \"  Data: \${DATA_DIR}/epoch1_data.parquet\"
echo \"  Starting fresh from base model\"
echo \"================================================================================\"
echo \"\"

export STAGE1_SAVE_PATH=\"\${BASE_SAVE_PATH}/stage1\"
mkdir -p \$STAGE1_SAVE_PATH

python3 -m verl.trainer.main_ppo \\
    algorithm.adv_estimator=grpo \\
    algorithm.norm_adv_by_std_in_grpo=True \\
    algorithm.kl_penalty=kl \\
    algorithm.kl_ctrl.type=fixed \\
    algorithm.kl_ctrl.kl_coef=0.01 \\
    algorithm.kl_ctrl.target_kl=0.10 \\
    algorithm.kl_ctrl.horizon=10000 \\
    algorithm.use_kl_in_reward=False \\
    \\
    data.train_files=\${DATA_DIR}/epoch1_data.parquet \\
    data.val_files=\${VALID_FILE} \\
    data.train_batch_size=256 \\
    data.max_prompt_length=4096 \\
    data.max_response_length=2048 \\
    data.filter_overlong_prompts=True \\
    data.truncation=error \\
    data.image_key=images \\
    data.dataloader_num_workers=8 \\
    data.validation_shuffle=False \\
    data.return_multi_modal_inputs=True \\
    data.val_batch_size=4 \\
    \\
    actor_rollout_ref.model.path=\${BASE_MODEL} \\
    actor_rollout_ref.actor.optim.lr=1e-6 \\
    actor_rollout_ref.actor.ppo_epochs=4 \\
    actor_rollout_ref.actor.grad_clip=1.0 \\
    actor_rollout_ref.actor.use_kl_loss=True \\
    actor_rollout_ref.actor.kl_loss_coef=0.01 \\
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \\
    actor_rollout_ref.actor.entropy_coeff=0.01 \\
    actor_rollout_ref.actor.clip_ratio=0.2 \\
    actor_rollout_ref.actor.loss_agg_mode=token-mean \\
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \\
    actor_rollout_ref.actor.use_dynamic_bsz=False \\
    actor_rollout_ref.actor.shuffle=False \\
    \\
    actor_rollout_ref.model.use_remove_padding=True \\
    actor_rollout_ref.model.enable_gradient_checkpointing=True \\
    actor_rollout_ref.actor.fsdp_config.param_offload=False \\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \\
    \\
    actor_rollout_ref.rollout.temperature=1.0 \\
    actor_rollout_ref.rollout.top_p=1.0 \\
    actor_rollout_ref.rollout.response_length=2048 \\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\
    actor_rollout_ref.rollout.name=\$ENGINE \\
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \\
    actor_rollout_ref.rollout.enable_chunked_prefill=False \\
    actor_rollout_ref.rollout.enforce_eager=True \\
    actor_rollout_ref.rollout.free_cache_engine=True \\
    actor_rollout_ref.rollout.n=8 \\
    actor_rollout_ref.rollout.do_sample=True \\
    actor_rollout_ref.rollout.val_kwargs.n=1 \\
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \\
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \\
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \\
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \\
    \\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \\
    \\
    custom_reward_function.path=\${WORK_DIR}/custom_rewards/grounding_reward_notag_singleturn2.py \\
    custom_reward_function.name=compute_score \\
    reward_model.reward_manager=naive \\
    \\
    trainer.critic_warmup=0 \\
    trainer.logger='[\"console\",\"wandb\"]' \\
    trainer.project_name=grpo_medical_grounding_continuous \\
    trainer.experiment_name=Qwen_RL_staged_continuous \\
    trainer.n_gpus_per_node=4 \\
    trainer.nnodes=\$SLURM_NNODES \\
    trainer.save_freq=50 \\
    trainer.test_freq=10 \\
    trainer.val_before_train=True \\
    trainer.log_val_generations=5 \\
    trainer.default_local_dir=\${STAGE1_SAVE_PATH} \\
    trainer.total_epochs=5

# Find the latest checkpoint from stage 1
STAGE1_CHECKPOINT=\$(find \${STAGE1_SAVE_PATH} -name \"global_step*\" -type d | sort -V | tail -1)

if [ -z \"\$STAGE1_CHECKPOINT\" ]; then
    echo \"ERROR: No checkpoint found for Stage 1 at \${STAGE1_SAVE_PATH}\"
    exit 1
fi

echo \"\"
echo \"✓ Stage 1 complete. Checkpoint: \${STAGE1_CHECKPOINT}\"
echo \"\"

# ============================================================================
# STAGE 2: EASY + MEDIUM SAMPLES (5 epochs - RESUME from Stage 1)
# ============================================================================

echo \"\"
echo \"================================================================================\"
echo \"  STAGE 2: TRAINING ON EASY + MEDIUM SAMPLES (5 EPOCHS)\"
echo \"  Data: \${DATA_DIR}/epoch2_data.parquet\"
echo \"  RESUMING from Stage 1 checkpoint: \${STAGE1_CHECKPOINT}\"
echo \"================================================================================\"
echo \"\"

export STAGE2_SAVE_PATH=\"\${BASE_SAVE_PATH}/stage2\"
mkdir -p \$STAGE2_SAVE_PATH

python3 -m verl.trainer.main_ppo \\
    algorithm.adv_estimator=grpo \\
    algorithm.norm_adv_by_std_in_grpo=True \\
    algorithm.kl_penalty=kl \\
    algorithm.kl_ctrl.type=fixed \\
    algorithm.kl_ctrl.kl_coef=0.01 \\
    algorithm.kl_ctrl.target_kl=0.10 \\
    algorithm.kl_ctrl.horizon=10000 \\
    algorithm.use_kl_in_reward=False \\
    \\
    data.train_files=\${DATA_DIR}/epoch2_data.parquet \\
    data.val_files=\${VALID_FILE} \\
    data.train_batch_size=256 \\
    data.max_prompt_length=4096 \\
    data.max_response_length=2048 \\
    data.filter_overlong_prompts=True \\
    data.truncation=error \\
    data.image_key=images \\
    data.dataloader_num_workers=8 \\
    data.validation_shuffle=False \\
    data.return_multi_modal_inputs=True \\
    data.val_batch_size=4 \\
    \\
    actor_rollout_ref.model.path=\${BASE_MODEL} \\
    actor_rollout_ref.actor.optim.lr=8e-7 \\
    actor_rollout_ref.actor.ppo_epochs=4 \\
    actor_rollout_ref.actor.grad_clip=1.0 \\
    actor_rollout_ref.actor.use_kl_loss=True \\
    actor_rollout_ref.actor.kl_loss_coef=0.01 \\
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \\
    actor_rollout_ref.actor.entropy_coeff=0.01 \\
    actor_rollout_ref.actor.clip_ratio=0.2 \\
    actor_rollout_ref.actor.loss_agg_mode=token-mean \\
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \\
    actor_rollout_ref.actor.use_dynamic_bsz=False \\
    actor_rollout_ref.actor.shuffle=False \\
    \\
    actor_rollout_ref.model.use_remove_padding=True \\
    actor_rollout_ref.model.enable_gradient_checkpointing=True \\
    actor_rollout_ref.actor.fsdp_config.param_offload=False \\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \\
    \\
    actor_rollout_ref.rollout.temperature=1.0 \\
    actor_rollout_ref.rollout.top_p=1.0 \\
    actor_rollout_ref.rollout.response_length=2048 \\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\
    actor_rollout_ref.rollout.name=\$ENGINE \\
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \\
    actor_rollout_ref.rollout.enable_chunked_prefill=False \\
    actor_rollout_ref.rollout.enforce_eager=True \\
    actor_rollout_ref.rollout.free_cache_engine=True \\
    actor_rollout_ref.rollout.n=8 \\
    actor_rollout_ref.rollout.do_sample=True \\
    actor_rollout_ref.rollout.val_kwargs.n=1 \\
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \\
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \\
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \\
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \\
    \\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \\
    \\
    custom_reward_function.path=\${WORK_DIR}/custom_rewards/grounding_reward_notag_singleturn2.py \\
    custom_reward_function.name=compute_score \\
    reward_model.reward_manager=naive \\
    \\
    trainer.critic_warmup=0 \\
    trainer.logger='[\"console\",\"wandb\"]' \\
    trainer.project_name=grpo_medical_grounding_continuous \\
    trainer.experiment_name=Qwen_RL_staged_continuous \\
    trainer.n_gpus_per_node=4 \\
    trainer.nnodes=\$SLURM_NNODES \\
    trainer.save_freq=50 \\
    trainer.test_freq=10 \\
    trainer.val_before_train=True \\
    trainer.log_val_generations=5 \\
    trainer.default_local_dir=\${STAGE2_SAVE_PATH} \\
    trainer.total_epochs=5 \\
    trainer.resume_mode=resume_path \\
    trainer.resume_from_path=\${STAGE1_CHECKPOINT}

# Find the latest checkpoint from stage 2
STAGE2_CHECKPOINT=\$(find \${STAGE2_SAVE_PATH} -name \"global_step*\" -type d | sort -V | tail -1)

if [ -z \"\$STAGE2_CHECKPOINT\" ]; then
    echo \"ERROR: No checkpoint found for Stage 2 at \${STAGE2_SAVE_PATH}\"
    exit 1
fi

echo \"\"
echo \"✓ Stage 2 complete. Checkpoint: \${STAGE2_CHECKPOINT}\"
echo \"\"

# ============================================================================
# STAGE 3: FULL CURRICULUM (10 epochs - RESUME from Stage 2)
# ============================================================================

echo \"\"
echo \"================================================================================\"
echo \"  STAGE 3: TRAINING ON FULL DIFFICULTY SPECTRUM (10 EPOCHS)\"
echo \"  Data: \${DATA_DIR}/epoch3_data.parquet\"
echo \"  RESUMING from Stage 2 checkpoint: \${STAGE2_CHECKPOINT}\"
echo \"================================================================================\"
echo \"\"

export STAGE3_SAVE_PATH=\"\${BASE_SAVE_PATH}/stage3_final\"
mkdir -p \$STAGE3_SAVE_PATH

python3 -m verl.trainer.main_ppo \\
    algorithm.adv_estimator=grpo \\
    algorithm.norm_adv_by_std_in_grpo=True \\
    algorithm.kl_penalty=kl \\
    algorithm.kl_ctrl.type=fixed \\
    algorithm.kl_ctrl.kl_coef=0.01 \\
    algorithm.kl_ctrl.target_kl=0.10 \\
    algorithm.kl_ctrl.horizon=10000 \\
    algorithm.use_kl_in_reward=False \\
    \\
    data.train_files=\${DATA_DIR}/epoch3_data.parquet \\
    data.val_files=\${VALID_FILE} \\
    data.train_batch_size=256 \\
    data.max_prompt_length=4096 \\
    data.max_response_length=2048 \\
    data.filter_overlong_prompts=True \\
    data.truncation=error \\
    data.image_key=images \\
    data.dataloader_num_workers=8 \\
    data.validation_shuffle=False \\
    data.return_multi_modal_inputs=True \\
    data.val_batch_size=4 \\
    \\
    actor_rollout_ref.model.path=\${BASE_MODEL} \\
    actor_rollout_ref.actor.optim.lr=5e-7 \\
    actor_rollout_ref.actor.ppo_epochs=4 \\
    actor_rollout_ref.actor.grad_clip=1.0 \\
    actor_rollout_ref.actor.use_kl_loss=True \\
    actor_rollout_ref.actor.kl_loss_coef=0.01 \\
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \\
    actor_rollout_ref.actor.entropy_coeff=0.01 \\
    actor_rollout_ref.actor.clip_ratio=0.2 \\
    actor_rollout_ref.actor.loss_agg_mode=token-mean \\
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \\
    actor_rollout_ref.actor.use_dynamic_bsz=False \\
    actor_rollout_ref.actor.shuffle=False \\
    \\
    actor_rollout_ref.model.use_remove_padding=True \\
    actor_rollout_ref.model.enable_gradient_checkpointing=True \\
    actor_rollout_ref.actor.fsdp_config.param_offload=False \\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \\
    \\
    actor_rollout_ref.rollout.temperature=1.0 \\
    actor_rollout_ref.rollout.top_p=1.0 \\
    actor_rollout_ref.rollout.response_length=2048 \\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\
    actor_rollout_ref.rollout.name=\$ENGINE \\
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \\
    actor_rollout_ref.rollout.enable_chunked_prefill=False \\
    actor_rollout_ref.rollout.enforce_eager=True \\
    actor_rollout_ref.rollout.free_cache_engine=True \\
    actor_rollout_ref.rollout.n=8 \\
    actor_rollout_ref.rollout.do_sample=True \\
    actor_rollout_ref.rollout.val_kwargs.n=1 \\
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \\
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \\
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \\
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \\
    \\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \\
    \\
    custom_reward_function.path=\${WORK_DIR}/custom_rewards/grounding_reward_notag_singleturn2.py \\
    custom_reward_function.name=compute_score \\
    reward_model.reward_manager=naive \\
    \\
    trainer.critic_warmup=0 \\
    trainer.logger='[\"console\",\"wandb\"]' \\
    trainer.project_name=grpo_medical_grounding_continuous \\
    trainer.experiment_name=Qwen_RL_staged_continuous \\
    trainer.n_gpus_per_node=4 \\
    trainer.nnodes=\$SLURM_NNODES \\
    trainer.save_freq=25 \\
    trainer.test_freq=5 \\
    trainer.val_before_train=True \\
    trainer.log_val_generations=5 \\
    trainer.default_local_dir=\${STAGE3_SAVE_PATH} \\
    trainer.total_epochs=10 \\
    trainer.resume_mode=resume_path \\
    trainer.resume_from_path=\${STAGE2_CHECKPOINT}

# Find the final checkpoint
FINAL_CHECKPOINT=\$(find \${STAGE3_SAVE_PATH} -name \"global_step*\" -type d | sort -V | tail -1)

echo \"\"
echo \"================================================================================\"
echo \"  ✅ CONTINUOUS CURRICULUM TRAINING COMPLETE\"
echo \"  Final checkpoint: \${FINAL_CHECKPOINT}\"
echo \"  This checkpoint contains all 20 epochs of continuous training:\"
echo \"  - Epochs 1-5: Easy samples (Stage 1)\"
echo \"  - Epochs 6-10: Easy + Medium samples (Stage 2)\" 
echo \"  - Epochs 11-20: Full difficulty spectrum (Stage 3)\"
echo \"================================================================================\"
echo \"\"

# Create a symlink to the final checkpoint for easy access
if [ ! -z \"\$FINAL_CHECKPOINT\" ]; then
    ln -sf \${FINAL_CHECKPOINT} \${BASE_SAVE_PATH}/FINAL_CHECKPOINT
    echo \"Created symlink: \${BASE_SAVE_PATH}/FINAL_CHECKPOINT -> \${FINAL_CHECKPOINT}\"
fi

srun --overlap --ntasks=\$SLURM_NNODES ray stop --force

exit

else
wait
fi
"