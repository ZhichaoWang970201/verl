#!/bin/bash
#SBATCH --job-name=rlvr_rfpp_8n
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
# SBATCH --mem=200G
#SBATCH --mem-per-cpu=0
#SBATCH --time=00:00:00
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --output=/mnt/home/zhichao/logs/slurm-%j.out
#SBATCH --error=/mnt/home/zhichao/logs/slurm-%j.err

# REINFORCE++: reinforce_plus_plus advantage + PPO-clip (loss_mode=vanilla).
# Global advantage normalization + discounted returns across whole response.
algo=reinforce_plus_plus
loss_mode=vanilla
beta=1
lr=3e-6
rollout_n=16
sp_size=2
train_batch_size=1024
gen_batch_size=$((train_batch_size * 3))
max_response_length=8192
clip_ratio_low=0.2
clip_ratio_high=0.2
ENV_DIR=/mnt/home/zhichao/verl-qwen3-32b/.venv
ENVLOCK=
VERL_DIR=/mnt/home/zhichao/verl-qwen3-32b
YOUR_RUN_NAME="REINFORCE_PP_LR_${lr}_8N"
YOUR_PROJECT_NAME="Qwen3-chat"

math_train_path="$HOME/data/DAPO_data/dapo-math-17k.parquet"
# Eval suite: AIME 2024 (DAPO format, 30-row dedup), AIME 2025, MATH-500,
# GSM8K, AMC23, MinervaMath. DAPO reward manager dispatches by data_source so
# each parquet's native prompt/scorer are preserved. val_kwargs.n=32 ->
# mean@32 / best@32 (~pass@32) / maj@32 (cons@32).
aime24_test_path="$HOME/data/DAPO_data/aime-2024-dedup30.parquet"
aime25_test_path="$HOME/data/DAPO_data/aime-2025.parquet"
math500_test_path="$HOME/data/DAPO_data/math-500.parquet"
gsm8k_test_path="$HOME/data/DAPO_data/gsm8k-test.parquet"
amc23_test_path="$HOME/data/DAPO_data/amc23.parquet"
minervamath_test_path="$HOME/data/DAPO_data/minervamath.parquet"

train_files="[$math_train_path]"
test_files="[$aime24_test_path,$aime25_test_path,$math500_test_path,$gsm8k_test_path,$amc23_test_path,$minervamath_test_path]"

nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' '))

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6380
dashboard_port=8267
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torch_cache_${SLURM_JOB_ID}
# B200 (sm_100) workaround: vLLM auto-selects flashinfer TRTLLM-Gen, which JIT-
# downloads cubins under ~/.cache/flashinfer (NFS) and crashes on POSIX flock
# (ENOLCK). Force FlashAttention backend; redirect any other flashinfer cache
# to local /tmp so flock succeeds.
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_${SLURM_JOB_ID}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

printenv


echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "source $ENV_DIR/bin/activate && \
        export LD_PRELOAD=$ENVLOCK && \
        export HF_HUB_OFFLINE=1 && \
        export TRANSFORMERS_OFFLINE=1 && \
        ray start --head \
        --node-ip-address=\"$head_node_ip\" \
        --port=\"$port\" \
        --dashboard-port=\"$dashboard_port\" \
        --num-cpus \"${SLURM_CPUS_PER_TASK}\" \
        --num-gpus \"${SLURM_GPUS_PER_NODE}\" \
        --block" &
sleep 10

worker_num=$((SLURM_JOB_NUM_NODES - 1))
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "worker_num: $worker_num"

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        bash -c "source $ENV_DIR/bin/activate && \
            export LD_PRELOAD=$ENVLOCK && \
            export HF_HUB_OFFLINE=1 && \
            export TRANSFORMERS_OFFLINE=1 && \
            ray start --address \"$ip_head\" \
            --num-cpus \"${SLURM_CPUS_PER_TASK}\" \
            --num-gpus \"${SLURM_GPUS_PER_NODE}\" \
            --block" &
    sleep 5
done
sleep 10


echo "Testing Ray initialization in the slurm nodes..."
bash -c "source $ENV_DIR/bin/activate && python3 -c '
import ray
try:
    ray.init(address=\"$ip_head\")
    print(\"\n=== Ray Cluster Status ===\")
    print(f\"Number of nodes: {len(ray.nodes())}\")
    for node in ray.nodes():
        print(\"Node: {}, Status: {}\".format(node[\"NodeManagerHostname\"], node[\"Alive\"]))
    ray.shutdown()
    print(\"Ray initialization successful!\")
except Exception as e:
    print(f\"Ray initialization failed: {str(e)}\")
'"
echo "=== Ray test completed ==="

echo "Initializing training..."

PYTHONUNBUFFERED=1 \
srun --overlap --nodes=1 --ntasks=1 \
bash -c "source '"$ENV_DIR"'/bin/activate && \
    export LD_PRELOAD=$ENVLOCK && \
    export PYTHONPATH=$VERL_DIR:\${PYTHONPATH:-} && \
    cd $VERL_DIR && \
    python3 -m recipe.dapo.main_dapo \
    algorithm.adv_estimator=$algo \
    actor_rollout_ref.actor.policy_loss.loss_mode=$loss_mode \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.prompt_key=prompt \
    data.train_batch_size=$train_batch_size \
    data.gen_batch_size=$gen_batch_size \
    data.max_prompt_length=2048 \
    data.max_response_length=$max_response_length \
    data.truncation='left' \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=acc \
    algorithm.filter_groups.max_num_gen_batches=10 \
    actor_rollout_ref.model.path=Qwen/Qwen3-30B-A3B-Base \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=$beta \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.fsdp_size=8 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    reward.reward_manager.name=dapo \
    reward.reward_kwargs.overlong_buffer_cfg.enable=False \
    reward.reward_kwargs.max_resp_len=$max_response_length \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$YOUR_PROJECT_NAME \
    trainer.experiment_name=$YOUR_RUN_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$SLURM_NNODES \
    trainer.val_before_train=True \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size}"
