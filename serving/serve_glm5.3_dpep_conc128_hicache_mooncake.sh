#!/usr/bin/env bash
# GLM-5.3 on 8x B200, dp8/tp8, with a three-level KV cache: L1 on the GPUs, L2 in host
# memory (--hicache-ratio 2), L3 in a Mooncake store (8 x 192 GiB = 1.5 TiB).
# Starts the Mooncake master, then the sglang server. The master stops with the server.
#
# Environment:
#   MOONCAKE_MASTER_ADDR         default 127.0.0.1:50051; a non-local address joins that
#                                master instead of starting one
#   MOONCAKE_START_MASTER        auto (default) | 1 | 0
#   MOONCAKE_EXPECTED_CLIENTS    segments to wait for, default 8
#   MOONCAKE_RDMA_DEV            RoCE device, default mlx5_4
#   MC_GID_INDEX                 GID index, default 5
#   MOONCAKE_LOCAL_HOST          fabric VIP to advertise, default derived from the GID
#   HICACHE_PREFETCH_TIMEOUT_MAX L3 prefetch ceiling in seconds, default 120
#   ALLOW_MEM_OVERCOMMIT=1       skip the host-memory preflight
#
# Stop with Ctrl-C, or signal the process group:
#   pid=$(pgrep -f 'serve_glm5.3_dpep_conc128_hicache_mooncak[e].sh' | head -1)
#   kill -TERM -"$(ps -o pgid= -p "$pid" | tr -d ' ')"
set -euo pipefail

# -- preflight ---------------------------------------------------------------------
# Refuse to start unless MemAvailable covers L2 + L3.
mem_available_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
need_bytes=$(( 8 * 2 * 979456 * 55932 + 8*192*1024*1024*1024 ))
have_bytes=$(( mem_available_kb * 1024 ))
if [ "$need_bytes" -gt "$have_bytes" ] && [ "${ALLOW_MEM_OVERCOMMIT:-0}" != "1" ]; then
  echo "L2 + L3 want ~$((need_bytes / 1024 / 1024 / 1024)) GiB of host memory," \
       "MemAvailable is $((have_bytes / 1024 / 1024 / 1024)) GiB." >&2
  echo "Lower --hicache-ratio, lower global_segment_size, or set ALLOW_MEM_OVERCOMMIT=1." >&2
  exit 1
fi

# -- L3: the Mooncake master ---------------------------------------------------------
MOONCAKE_MASTER_ADDR="${MOONCAKE_MASTER_ADDR:-127.0.0.1:50051}"
case "${MOONCAKE_START_MASTER:-auto}" in
  auto) case "$MOONCAKE_MASTER_ADDR" in
          127.0.0.1:*|localhost:*) MOONCAKE_START_MASTER=1 ;;
          *)                       MOONCAKE_START_MASTER=0 ;;
        esac ;;
esac
mooncake_master_pid=""
report_pid=""
if [ "$MOONCAKE_START_MASTER" = 1 ]; then
  mooncake_master \
    --rpc_port=50051 \
    --metrics_port=9003 \
    --allocation_strategy=local_first \
    --eviction_high_watermark_ratio=0.95 \
    > /tmp/mooncake_master.log 2>&1 &
  mooncake_master_pid=$!
else
  echo "[mooncake] joining master at $MOONCAKE_MASTER_ADDR (not starting one here)"
fi

# Kill the wrapper's child, then the wrapper.
stop_background() {
  for pid in ${mooncake_master_pid:+"$mooncake_master_pid"} ${report_pid:+"$report_pid"}; do
    pkill -P "$pid" 2>/dev/null || true
    kill "$pid" 2>/dev/null || true
  done
}
trap stop_background EXIT

# Wait for the master to listen.
master_host="${MOONCAKE_MASTER_ADDR%:*}"
master_port="${MOONCAKE_MASTER_ADDR##*:}"
for _ in $(seq 1 100); do
  if (exec 3<>"/dev/tcp/$master_host/$master_port") 2>/dev/null; then break; fi
  if [ -n "$mooncake_master_pid" ] && ! kill -0 "$mooncake_master_pid" 2>/dev/null; then
    tail -5 /tmp/mooncake_master.log >&2
    echo "mooncake_master exited before it listened on $MOONCAKE_MASTER_ADDR" >&2
    exit 1
  fi
  sleep 0.1
done

# Report the mounted L3 capacity once every rank has mounted a segment, then exit.
if [ "$MOONCAKE_START_MASTER" = 1 ]; then
(
  expected="${MOONCAKE_EXPECTED_CLIENTS:-8}"
  clients=0
  for _ in $(seq 1 600); do
    read -r capacity clients <<<"$(curl -sf -m 2 http://127.0.0.1:9003/metrics 2>/dev/null |
      awk '/^master_total_capacity_bytes /{c=$2} /^master_active_clients /{n=$2} END{print c+0, n+0}' || true)"
    if [ "${clients:-0}" -ge "$expected" ]; then
      echo "[mooncake] L3 mounted: $((capacity / 1024 / 1024 / 1024)) GiB across ${clients} clients"
      exit 0
    fi
    sleep 5
  done
  echo "[mooncake] only ${clients:-0}/${expected} ranks mounted a segment after 50 minutes --" \
       "L3 is degraded; check /tmp/mooncake_master.log and the hicache lines in the server log" >&2
) &
report_pid=$!
fi

# -- the server ----------------------------------------------------------------------
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_DP_USE_GATHERV=1
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8192
export PYTHONNOUSERSITE=1

# Fabric VIP to advertise, taken from the low 32 bits of the device's IPv4-mapped GID.
MOONCAKE_RDMA_DEV="${MOONCAKE_RDMA_DEV:-mlx5_4}"
export MC_GID_INDEX="${MC_GID_INDEX:-5}"
mooncake_gid_path="/sys/class/infiniband/${MOONCAKE_RDMA_DEV}/ports/1/gids/${MC_GID_INDEX}"
if [ -z "${MOONCAKE_LOCAL_HOST:-}" ]; then
  if [ ! -r "$mooncake_gid_path" ]; then
    echo "no GID $MC_GID_INDEX on $MOONCAKE_RDMA_DEV ($mooncake_gid_path)." >&2
    echo "Set MOONCAKE_RDMA_DEV/MC_GID_INDEX for this host, or MOONCAKE_LOCAL_HOST." >&2
    exit 1
  fi
  gid="$(cat "$mooncake_gid_path")"
  MOONCAKE_LOCAL_HOST="$(printf '%d.%d.%d.%d' \
    "0x${gid:30:2}" "0x${gid:32:2}" "0x${gid:35:2}" "0x${gid:37:2}")"
fi
case "$MOONCAKE_LOCAL_HOST" in
  0.0.0.0|127.*) echo "GID $MC_GID_INDEX on $MOONCAKE_RDMA_DEV is not a fabric address" \
                      "($MOONCAKE_LOCAL_HOST); pick the RoCE device for this host." >&2
                 exit 1 ;;
esac
echo "[mooncake] RDMA $MOONCAKE_RDMA_DEV gid $MC_GID_INDEX, advertising $MOONCAKE_LOCAL_HOST"

# Do not put a comment between the flags below.
/share/bin/gpu-run 0,1,2,3,4,5,6,7 -- python3 -m sglang.launch_server \
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
  --max-queued-requests 320 \
  --mem-fraction-static 0.921 \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --context-length 1048576 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-write-policy write_through \
  --hicache-storage-backend mooncake \
  --hicache-storage-backend-extra-config "{\"master_server_address\": \"${MOONCAKE_MASTER_ADDR}\", \"metadata_server\": \"P2PHANDSHAKE\", \"protocol\": \"rdma\", \"device_name\": \"${MOONCAKE_RDMA_DEV}\", \"local_hostname\": \"${MOONCAKE_LOCAL_HOST}\", \"global_segment_size\": \"192gb\", \"prefetch_timeout_max\": ${HICACHE_PREFETCH_TIMEOUT_MAX:-120}}" \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --tokenizer-worker-num 8 \
  --enable-metrics \
  --enable-metrics-for-all-schedulers \
  --enable-cache-report \
  --flashinfer-allreduce-fusion-backend auto \
  --model-loader-extra-config '{"enable_multithread_load": true}'
