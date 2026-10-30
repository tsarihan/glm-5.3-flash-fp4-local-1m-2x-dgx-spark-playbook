#!/usr/bin/env bash
# GLM-5.3-Flash FP4-Local on 2x DGX Spark, TP=2, vLLM b12x fork (local-inference-lab/vllm 8244d1c,
# built from source with eugr/spark-vllm-docker 0bdc9c9, FlashInfer 0213747d -> image vllm-node-b12x).
# FlashInfer must predate c46a0b50 (2026-10-17), which split b12x intrinsics into a package and
# dropped intrinsics.swizzle_block_scale that this vLLM fork calls (AttributeError at load).
# Flags follow tedngai/glm-5.3-flash-2x-dgx-spark recipe a810bf2 (glm-5.3-flash-tuned.yaml), except:
#   - max-model-len 1048576 (model native max_position_embeddings; recipe used 500000)
#   - GMU 0.85 (proven ceiling on these nodes; 0.90 wedged one; recipe used 0.87)
#   - our own docker run: named + kept containers (never --rm), RoCE NCCL env from our GLM lane.
#   - seccomp=unconfined: the b12x loader uses io_uring, which docker's default seccomp blocks
#     ("io_uring bounce initialization failed: Operation not permitted"). Narrower than --privileged.
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is REQUIRED: without it the caching allocator
# holds freed blocks during b12x state prep and drives MemAvailable to 0 GiB on both nodes after load.
# --recurrent-checkpoint-policy aligned (eugr #416): lets the hybrid-Mamba recurrent state
# checkpoint alongside KV blocks, so repeat prefix hits (same repo + new question) return in
# seconds instead of a full re-prefill; default "auto" loses the Mamba state at request end.
# usage: NODE_RANK=0 ./serve.sh   on rank-0 node (API head, port 8000)
#        NODE_RANK=1 ./serve.sh   on rank-1 node (headless)
set -euo pipefail
NODE_RANK="${NODE_RANK:?set NODE_RANK=0 on rank-0 node, 1 on rank-1 node}"
IMAGE="${IMAGE:-vllm-node-b12x:fi-b1f081e6}"
MODEL="${MODEL:-/data/models/glm-5.3-flash-fp4-local-nvidia}"
NAME="${NAME:-glm53-fp4local-r$NODE_RANK-$(date +%m%d-%H%M%S)}"
PORT="${PORT:-8000}"; MAXLEN="${MAXLEN:-1048576}"; GMU="${GMU:-0.85}"
SEQS="${SEQS:-8}"; CHUNK="${CHUNK:-8192}"; BLOCK="${BLOCK:-256}"
# Adaptive MTP depth (eugr #432; sandpig's 2x-Spark GLM: +22% hard prose, +32-42% mixed single
# stream, acceptance ~60%). Deeper drafts pay on predictable content (code repos). Fields exist in
# SpeculativeConfig at 8244d1c (verified). ADAPTIVE=0 -> fixed num_speculative_tokens=$MTP
# (old baseline MTP=1, acceptance 77.9%, 31.6-31.9 tok/s long answers).
ADAPTIVE="${ADAPTIVE:-1}"; MTP="${MTP:-5}"; MTP_WIN="${MTP_WIN:-8}"; MTP_INIT="${MTP_INIT:-5}"
# b12x loader: "auto" picks gds + a SharedReadGroup across the TP group, which hung rank 1 on the
# second Spark for 3 min (it never loaded). "bounce" = each node reads its own local shards via io_uring.
READ_MODE="${READ_MODE:-bounce}"
# text-only by default: rank 0 (API server + mm processor) dipped to ~6 GB free right after load
IMAGES="${IMAGES:-0}"; VIDEOS="${VIDEOS:-0}"   # per-request limits; 0/0 = text-only
MM="{\"image\":$IMAGES,\"video\":$VIDEOS}"
MASTER="${NODE0_FABRIC:?set to the rank-0 fabric IP}"; MPORT="${MPORT:-25300}"
case "$NODE_RANK" in 0) HOST_IP=${NODE0_FABRIC}; EXTRA="" ;; 1) HOST_IP=${NODE1_FABRIC:?set to the rank-1 fabric IP}; EXTRA="--headless" ;; esac
D=$HOME/llmruntimes/glm53-fp4local; JIT=$D/jit-cache; mkdir -p "$JIT"
docker inspect "$NAME" >/dev/null 2>&1 && { echo "container $NAME exists - refusing (never rm)"; exit 1; }
SPEC_FIXED="{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP,\"moe_backend\":\"humming\",\"attention_backend\":\"B12X\"}"
# 8244d1c asserts a per-batch-size schedule whenever dynamic spec decode is on (window-only
# adaptive dies at config time, "assert schedule is not None"); MTP_SCHED supplies it.
MTP_SCHED="${MTP_SCHED:-[[1,8,$MTP]]}"
SPEC_ADAPT="{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP,\"num_speculative_tokens_per_batch_size\":$MTP_SCHED,\"adaptive_speculative_tokens_window\":$MTP_WIN,\"adaptive_speculative_tokens_initial\":$MTP_INIT,\"moe_backend\":\"humming\",\"attention_backend\":\"B12X\"}"
if [[ "$ADAPTIVE" == "1" ]]; then SPEC="$SPEC_ADAPT"; else SPEC="$SPEC_FIXED"; fi
CMD="vllm serve /model --served-model-name glm-5.3-flash-fp4-local \
  --host 0.0.0.0 --port $PORT \
  --tensor-parallel-size 2 --decode-context-parallel-size 1 \
  --mamba-cache-mode align --enable-prefix-caching --recurrent-checkpoint-policy aligned \
  --enable-chunked-prefill \
  --dtype bfloat16 --kv-cache-dtype fp8 --quantization modelopt_mixed \
  --attention-backend B12X --block-size $BLOCK --moe-backend b12x --linear-backend b12x \
  --no-enable-flashinfer-autotune --load-format b12x \
  --model-loader-extra-config '{\"read_mode\":\"$READ_MODE\"}' \
  --max-model-len $MAXLEN --max-num-seqs $SEQS --max-num-batched-tokens $CHUNK \
  --speculative-config '$SPEC' \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
  --gpu-memory-utilization $GMU --limit-mm-per-prompt '$MM' \
  --nnodes 2 --node-rank $NODE_RANK --master-addr $MASTER --master-port $MPORT $EXTRA"
docker run -d --name "$NAME" --restart no --gpus all --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband \
  --security-opt seccomp=unconfined \
  --entrypoint /bin/bash \
  -v "$MODEL":/model:ro -v "$JIT":/root/.cache \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_HOST_IP=$HOST_IP \
  -e CUTE_DSL_ARCH=sm_121a -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_ENABLE_ROCE_ALLREDUCE=1 -e VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB \
  -e NCCL_BUFFSIZE=1048576 -e NCCL_MAX_CTAS=8 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_SSM_CONV_STATE_LAYOUT=DS \
  -e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_MEGA_AOT_ARTIFACT=1 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0 -e B12X_POLICY_MODE=auto \
  -e VLLM_B12X_MOE_FP4_FORCE_A16=1 -e B12X_W4A16_A4_PREFILL=1 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=roceP2p1s0f0 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ADDR_RANGE=${FABRIC_SUBNET:?e.g. 10.0.0.0/24} \
  -e NCCL_SOCKET_IFNAME=enP2p1s0f0np0 -e GLOO_SOCKET_IFNAME=enP2p1s0f0np0 -e TP_SOCKET_IFNAME=enP2p1s0f0np0 \
  -e UCX_NET_DEVICES=enP2p1s0f0np0 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ${EXTRA_ENV:-} \
  "$IMAGE" -c "$CMD"
echo "$NAME" > "$D/last-container-r$NODE_RANK"
echo "launched $NAME rank=$NODE_RANK maxlen=$MAXLEN gmu=$GMU mtp=$MTP port=$PORT"
