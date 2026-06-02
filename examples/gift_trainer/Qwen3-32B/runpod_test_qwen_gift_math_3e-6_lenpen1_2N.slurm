#!/bin/bash
#SBATCH --job-name=rlvr_gift_lenpen1_2N
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
# SBATCH --mem=200G
#SBATCH --mem-per-cpu=0
#SBATCH --time=00:00:00
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --output=/workspace/zhichao/logs/slurm-%j.out
#SBATCH --error=/workspace/zhichao/logs/slurm-%j.err
# Pin to known-good nodes (leaves node-[1,2,4,5,7,10]) — this is what keeps the run stable, by
# keeping SLURM off the bad hardware, exactly like the working runs landed on node-[2,4,5,7]:
#  - node-0: hosts the root SkyPilot cluster -> our Ray head fails there + GPU contention.
#  - node-8,9: faulty mlx5_7 (IbDev 4) IB rail -> NCCL QP/ECE handshake hangs, deadlocks init.
#SBATCH --exclude=node-0,node-8,node-9

beta=1
lr=3e-6
rollout_n=16
# GIFT length penalty: kl_div_sum = kl_div_sum / (response_length ** response_length_penalty).
# penalty=1 -> divide per-sample KL sum by response length (per-token mean) before group norm.
response_length_penalty=1
# Dynamic-sampling oversample: generate gen_batch_size prompts then filter to
# train_batch_size groups with non-degenerate accuracy (drops all-correct /
# all-wrong groups). gen = 3x train is the DAPO default.
train_batch_size=1024
gen_batch_size=$((train_batch_size * 3))
# Overlong-response shaping: responses within the last overlong_buffer_len
# tokens of max_response_length get a linear penalty up to -penalty_factor.
max_response_length=8192
overlong_buffer_len=1024
overlong_penalty_factor=1.0

# ---- Megatron engine knobs (replaces FSDP) ----
val_n=32
training_dtype=bfloat16          # revert: float16
all_param_offload=False          # A/B (job 1030 vs 1029): param_offload=True + gpu_mem_util=0.85 did
                                 # NOT speed rollout (it's straggler-tail-bound, not KV-limited) and
                                 # added a log_prob param-reload cost (+14s) -> reverted to baseline.
all_optimizer_offload=True       # keep optimizer states off-GPU during rollout
all_grad_offload=True            # frees ~15GB gradient buffers on GPU during update -> absorbs the
                                 # step-6 memory-growth OOM, with ZERO extra communication (unlike
                                 # shrinking micro_batch, which doubled per-microbatch NCCL = +48% slower).
recompute_granularity=full       # full: recompute ALL layers (incl. MoE MLP) in backward -> absorbs the
                                 # update_actor activation-growth OOM (jobs 1041/1042 OOM'd at step 5-12 in
                                 # _reduce_scatter_along_first_dim as response_length grew toward 8192).
                                 # selective (recompute attention only) was NOT enough. full requires
                                 # recompute_method+recompute_num_layers below (Megatron asserts this);
                                 # if reverting to selective, REMOVE those two override flags.
# Reuse the qwen3.5 venv directly (provides torch/ray/vllm/megatron/TE). PYTHONPATH=$VERL_DIR
# below makes `import verl` + recipe.dapo resolve to /workspace/zhichao/verl, overriding the
# venv's editable verl install — verified. No need to build/copy a separate venv.
ENV_DIR=/workspace/zhichao/verl-gift-qwen35/.venv
VERL_DIR=/workspace/zhichao/verl
YOUR_RUN_NAME="GIFT_Beta_${beta}_LR_${lr}_DynFilter_LenPen${response_length_penalty}_2N"
YOUR_PROJECT_NAME="Qwen3-chat"

# ---- CHANGED: data paths (runpod /workspace/zhichao/data layout) ----
math_train_path="/workspace/zhichao/data/dapo-math-17k.parquet"
aime24_test_path="/workspace/zhichao/data/aime-2024-dedup30-unique_aime24.parquet"
aime25_test_path="/workspace/zhichao/data/aime-2025-dedup30_aime25.parquet"
amc23_test_path="/workspace/zhichao/data/amc23-dedup40_amc23.parquet"

train_files="[$math_train_path]"
test_files="[$aime24_test_path,$aime25_test_path,$amc23_test_path]"

# Getting the node names
nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' '))

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
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

# ===================== CHANGED: IB / NCCL network setting =====================
# Compute a CONSISTENT NCCL_IB_HCA across all allocated nodes.
# Heterogeneous nodes can expose different active IB HCAs (e.g. node-7 has mlx5_5 up where
# others have mlx5_3). NCCL DEADLOCKS on the first cross-node collective if NCCL_IB_HCA
# differs between ranks (symptom: 2 nodes spin at 100% GPU, others block in ep_poll, then a
# gloo/NCCL timeout ~10 min into init). So gather each node's active HCAs and use the
# INTERSECTION uniformly on every rank.
echo "Computing consistent NCCL_IB_HCA across allocated nodes..."
_hca_detect='for d in $(ls /sys/class/infiniband/ 2>/dev/null | sort); do
  s=$(cat /sys/class/infiniband/$d/ports/1/state 2>/dev/null)
  l=$(cat /sys/class/infiniband/$d/ports/1/link_layer 2>/dev/null)
  r=$(cat /sys/class/infiniband/$d/ports/1/rate 2>/dev/null | awk "{print \$1+0}")
  [ "$s" = "4: ACTIVE" ] && [ "$l" = "InfiniBand" ] && [ "${r:-0}" -ge 200 ] && echo "$d"
done'
IB_HCA_COMMON=""
_hca_mismatch=0
_hca_first=""
for _nd in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
    _this=$(srun --nodes=1 --ntasks=1 -w "$_nd" --overlap bash -c "$_hca_detect" 2>/dev/null | sort -u)
    echo "  $_nd: $(echo "$_this" | paste -sd,)"
    if [ -z "$IB_HCA_COMMON" ]; then
        IB_HCA_COMMON="$_this"; _hca_first="$_this"
    else
        [ "$_this" != "$_hca_first" ] && _hca_mismatch=1
        IB_HCA_COMMON=$(comm -12 <(printf '%s\n' "$IB_HCA_COMMON") <(printf '%s\n' "$_this"))
    fi
done
IB_HCA_COMMON=$(printf '%s\n' "$IB_HCA_COMMON" | paste -sd,)
if [ "$_hca_mismatch" = 1 ]; then
    echo "WARNING: nodes expose DIFFERENT active IB HCAs (a NIC is down on at least one node)."
    echo "         Using the consistent intersection to avoid a NCCL deadlock: ${IB_HCA_COMMON}"
fi
if [ -z "$IB_HCA_COMMON" ]; then
    echo "PREFLIGHT FAILED: no IB HCA common to all allocated nodes; NCCL would hang. Aborting."
    exit 1
fi
echo "Consistent NCCL_IB_HCA = ${IB_HCA_COMMON}"
export IB_HCA_COMMON

# Static IB/NCCL env — propagated to every srun step via the default --export=ALL,
# so the Ray head/worker raylets (and the actors they spawn) inherit it.
export NCCL_PROTO=LL128
_RDMA_LIB=/workspace/zhichao/rdma_libs
export LD_LIBRARY_PATH="${_RDMA_LIB}:${_RDMA_LIB}/libibverbs:${LD_LIBRARY_PATH:-}"
[ -n "${IB_HCA_COMMON}" ] && export NCCL_IB_HCA="${IB_HCA_COMMON}"
export NCCL_IB_DISABLE=0
export NCCL_IB_GDR_LEVEL=2
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TIMEOUT=23
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_DEBUG=WARN
# Per-stage GPU memory logging: VERL_LOGGING_LEVEL=INFO un-suppresses log_gpu_memory_usage(),
# which prints allocated/reserved memory (with timestamps) at each stage boundary — used to
# profile per-stage GPU memory and to bucket nvidia-smi utilization samples by stage.
export VERL_LOGGING_LEVEL=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# verl force-injects CUDA_DEVICE_MAX_CONNECTIONS=1 into every Ray actor via runtime_env
# (constants_ppo.PPO_RAY_RUNTIME_ENV) UNLESS the launching shell already sets it
# (get_ppo_ray_runtime_env pops keys already present in os.environ). With =1 and
# virtual_pipeline_model_parallel_size=2 + overlap_p2p_comm=True, the single GPU work-
# submission channel serializes interleaved pipeline p2p sends/recvs and DEADLOCKS the
# first cross-node collective in _build_model_optimizer (symptom: 2 nodes spin 100% GPU,
# others block in a gloo recv that times out at 600s). Set =3 (proven value) so verl pops
# it from runtime_env and actors inherit it from the raylet. Must be set in the driver AND
# both ray-start raylets.
export CUDA_DEVICE_MAX_CONNECTIONS=3
# =============================================================================

# make sure we set environment variables before Ray initialization
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

# Set separate cache directories for each job to avoid conflicts
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torch_cache_${SLURM_JOB_ID}
# B200 (sm_100) workaround: vLLM auto-selects flashinfer TRTLLM-Gen, which JIT-
# downloads cubins under ~/.cache/flashinfer (NFS) and crashes on POSIX flock
# (ENOLCK). Force FlashAttention backend; redirect any other flashinfer cache
# to local /tmp so flock succeeds.
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_${SLURM_JOB_ID}
# Use the local HF cache; skip the Hub API (rate-limited when many ranks resolve
# the same model simultaneously, which crashed 58459 with HTTP 429).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# HF/datasets cache MUST live on shared /workspace: $HOME (/home/zhichao) is node-local
# and not writable on compute nodes, so datasets.load_dataset() crashed with
# PermissionError on os.makedirs('/home/zhichao'). Point all HF caches at /workspace.
export HF_HOME=/workspace/zhichao/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/zhichao/.cache/huggingface/datasets
# $HOME (/home/zhichao) is node-local and not writable on compute nodes; vLLM's
# model-info cache (~/.cache/vllm/modelinfos) and torch.compile cache mkdir there and
# crash with PermissionError. Redirect HOME + XDG_CACHE_HOME to shared, writable /workspace.
export HOME=/workspace/zhichao
export XDG_CACHE_HOME=/workspace/zhichao/.cache
# Megatron MoE update_actor forward OOM'd at step ~5 (fragmentation / "reserved but unallocated").
# NOTE: expandable_segments:True is INCOMPATIBLE with vLLM's CuMem memory pool (sleep mode) —
# it asserts-fails vLLM init. So use garbage_collection_threshold instead: proactively reclaims
# cached blocks under pressure to cut fragmentation OOMs, and is vLLM-pool-compatible.
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8
# wandb auth (compute nodes have no ~/.netrc) — avoids an auth failure at trainer init.
export WANDB_API_KEY=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
export WANDB_ENTITY=thefleet
# The container/SkyPilot env exports ROCR_VISIBLE_DEVICES (ROCm/AMD) alongside
# CUDA_VISIBLE_DEVICES. These are NVIDIA nodes, so ROCR is spurious, and verl's
# Worker._setup_env_cuda_visible_devices() hard-errors when BOTH are set. Drop it.
unset ROCR_VISIBLE_DEVICES

printenv


echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "source $ENV_DIR/bin/activate
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        export HF_HOME=/workspace/zhichao/.cache/huggingface
        export HF_DATASETS_CACHE=/workspace/zhichao/.cache/huggingface/datasets
        unset ROCR_VISIBLE_DEVICES
        export CUDA_DEVICE_MAX_CONNECTIONS=3
        export HOME=/workspace/zhichao
        export XDG_CACHE_HOME=/workspace/zhichao/.cache
        export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8
        # Pin Gloo/NCCL/TCPStore bootstrap sockets to the routable cluster NIC (the one
        # bearing head_node_ip's /24). Computed per-node: eth0 is a per-node 172.x Docker
        # bridge (NOT routable across nodes) and would hang cross-node CPU collectives.
        _CIF=\$(ip -o -4 addr show 2>/dev/null | awk '/ ${head_node_ip%.*}\./{print \$2; exit}')
        if [ -n \"\${_CIF}\" ]; then
          export GLOO_SOCKET_IFNAME=\${_CIF}
          export NCCL_SOCKET_IFNAME=\${_CIF}
          export TP_SOCKET_IFNAME=\${_CIF}
          echo \"[NET] SOCKET_IFNAME=\${_CIF}\"
        fi
        echo \"[NCCL] IB_HCA=\${NCCL_IB_HCA:-not_set} PROTO=\${NCCL_PROTO} GDR=\${NCCL_IB_GDR_LEVEL}\"
        ray start --head \
        --node-ip-address=\"$head_node_ip\" \
        --port=\"$port\" \
        --dashboard-port=\"$dashboard_port\" \
        --ray-client-server-port=30001 \
        --min-worker-port=20002 \
        --max-worker-port=29999 \
        --num-cpus \"${SLURM_CPUS_PER_TASK}\" \
        --num-gpus \"${SLURM_GPUS_PER_NODE}\" \
        --block" &
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "worker_num: $worker_num"

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        bash -c "source $ENV_DIR/bin/activate
            export HF_HUB_OFFLINE=1
            export TRANSFORMERS_OFFLINE=1
            export HF_HOME=/workspace/zhichao/.cache/huggingface
            export HF_DATASETS_CACHE=/workspace/zhichao/.cache/huggingface/datasets
            unset ROCR_VISIBLE_DEVICES
            export CUDA_DEVICE_MAX_CONNECTIONS=3
            export HOME=/workspace/zhichao
            export XDG_CACHE_HOME=/workspace/zhichao/.cache
            export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8
            _CIF=\$(ip -o -4 addr show 2>/dev/null | awk '/ ${head_node_ip%.*}\./{print \$2; exit}')
            if [ -n \"\${_CIF}\" ]; then
              export GLOO_SOCKET_IFNAME=\${_CIF}
              export NCCL_SOCKET_IFNAME=\${_CIF}
              export TP_SOCKET_IFNAME=\${_CIF}
              echo \"[NET] SOCKET_IFNAME=\${_CIF}\"
            fi
            echo \"[NCCL] IB_HCA=\${NCCL_IB_HCA:-not_set} PROTO=\${NCCL_PROTO} GDR=\${NCCL_IB_GDR_LEVEL}\"
            ray start --address \"$ip_head\" \
            --min-worker-port=20002 \
            --max-worker-port=29999 \
            --num-cpus \"${SLURM_CPUS_PER_TASK}\" \
            --num-gpus \"${SLURM_GPUS_PER_NODE}\" \
            --block" &
    sleep 5
done
sleep 10


# Ray initlization test (See whether any error in the above excution)
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
    export PYTHONPATH=$VERL_DIR:\${PYTHONPATH:-} && \
    export RAY_ADDRESS=\"$ip_head\" && \
    unset ROCR_VISIBLE_DEVICES && \
    export CUDA_DEVICE_MAX_CONNECTIONS=3 && \
    export HOME=/workspace/zhichao && \
    export XDG_CACHE_HOME=/workspace/zhichao/.cache && \
    export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8 && \
    cd $VERL_DIR && \
    python3 -m recipe.dapo.main_dapo --config-name=ppo_megatron_trainer \
    actor_rollout_ref.nccl_timeout=3600 \
    algorithm.adv_estimator=gift \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=gift \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.prompt_key=prompt \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    data.train_batch_size=$train_batch_size \
    +data.gen_batch_size=$gen_batch_size \
    data.max_prompt_length=2048 \
    data.max_response_length=$max_response_length \
    data.truncation='left' \
    +algorithm.filter_groups.enable=True \
    +algorithm.filter_groups.metric=acc \
    +algorithm.filter_groups.max_num_gen_batches=10 \
    actor_rollout_ref.model.path=/workspace/zhichao/models/Qwen3-30B-A3B-Base \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.response_length_penalty=$response_length_penalty \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30720 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=$beta \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.megatron.dtype=$training_dtype \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=null \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.context_parallel_size=1 \
    actor_rollout_ref.actor.megatron.param_offload=$all_param_offload \
    actor_rollout_ref.actor.megatron.optimizer_offload=$all_optimizer_offload \
    actor_rollout_ref.actor.megatron.grad_offload=$all_grad_offload \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=$recompute_granularity \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=fused \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.rollout.max_num_batched_tokens=20480 \
    actor_rollout_ref.rollout.max_model_len=10240 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=131072 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=$val_n \
    actor_rollout_ref.rollout.expert_parallel_size=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=131072 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.megatron.dtype=$training_dtype \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=null \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.context_parallel_size=1 \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.megatron.use_mbridge=True \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.attention_backend=fused \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=$overlong_buffer_len \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=$overlong_penalty_factor \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=$max_response_length \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$YOUR_PROJECT_NAME \
    trainer.experiment_name=$YOUR_RUN_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$SLURM_NNODES \
    trainer.val_before_train=True \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=1"
