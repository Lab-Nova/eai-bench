#!/usr/bin/env bash
# GLM-5.3 on 8x B200 -- dpep (DP attention + EP8), radix + hierarchical cache,
# tool calling, conc=128. Every flag below is literal: no variables, no
# substitutions, no arrays.
set -euo pipefail

export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_DP_USE_GATHERV=1
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8192
export PYTHONNOUSERSITE=1

exec /share/bin/gpu-run 0,1,2,3,4,5,6,7 -- python3 -m sglang.launch_server \
  --model-path /share/GLM-5.3 \
  --served-model-name GLM-5.3 \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --data-parallel-size 8 \
  --enable-dp-attention \
  --enable-prefill-delayer \
  --prefill-delayer-max-delay-passes 205 \
  --prefill-delayer-ignore-max-prefill-bs \
  --expert-parallel-size 8 \
  --moe-a2a-backend megamoe \
  --moe-runner-backend auto \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend dsa \
  --dsa-prefill-backend fastermla \
  --dsa-decode-backend trtllm \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --max-running-requests 128 \
  --cuda-graph-max-bs-decode 128 \
  --mem-fraction-static 0.921 \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --context-length 1048576 \
  --enable-hierarchical-cache \
  --hicache-size 256 \
  --hicache-write-policy write_through \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --tokenizer-worker-num 8 \
  --enable-metrics \
  --enable-metrics-for-all-schedulers \
  --enable-cache-report \
  --flashinfer-allreduce-fusion-backend auto \
  --model-loader-extra-config '{"enable_multithread_load": true}'
