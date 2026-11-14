#!/usr/bin/env bash
# FluxMoE 动机实验复现 —— 扫描驱动
# 用法：
#   MODEL=allenai/OLMoE-1B-7B-0924-Instruct TP=1 UTILS="0.55 0.65 0.75 0.85" bash run_sweep.sh
#   MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1 TP=4 UTILS="0.60 0.70 0.80 0.90" bash run_sweep.sh
#
# 注意：
#   1) 需要 Linux / WSL / Git Bash；Windows 原生 CMD 不支持
#   2) BENCH 指向你的 vLLM 仓库里的 benchmarks/benchmark_serving.py
#   3) gpu-memory-utilization 是启动参数，每次换 util 都要重启服务

set -uo pipefail

# ---------------- 可调参数 ----------------
MODEL="${MODEL:-allenai/OLMoE-1B-7B-0924-Instruct}"
TP="${TP:-1}"
UTILS="${UTILS:-0.55 0.65 0.75 0.85}"       # KV 预算档位
CTXS="${CTXS:-1024 2048 4096}"              # 输入长度
CONCS="${CONCS:-16 32 64 128 256}"          # 并发
OUT_LEN="${OUT_LEN:-256}"                   # 固定输出长度
MAX_LEN="${MAX_LEN:-5120}"                  # max-model-len，需 > 最大 ctx + out_len
MAX_SEQS="${MAX_SEQS:-256}"
PORT="${PORT:-8000}"
REPEAT="${REPEAT:-1}"                       # 每配置重复次数（正式跑请设 3，取中位数）
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="$ROOT/results/raw"
LOGS="$ROOT/results/logs"
BENCH="${BENCH:-benchmarks/benchmark_serving.py}"

mkdir -p "$RAW" "$LOGS"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
  fi
}
trap cleanup EXIT INT TERM

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# ---------------- 服务端 ----------------
start_server() {
  local util="$1" logf="$2"
  log "启动 vLLM：util=$util TP=$TP  (日志: $logf)"
  vllm serve "$MODEL" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$util" \
    --max-num-seqs "$MAX_SEQS" \
    --max-model-len "$MAX_LEN" \
    --disable-log-requests \
    --trust-remote-code \
    --port "$PORT" \
    >"$logf" 2>&1 &
  SERVER_PID=$!

  # 最多等 20 分钟（大模型加载慢）
  for _ in $(seq 1 240); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
      log "服务就绪"
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "!! 服务启动失败，见 $logf"
      tail -n 30 "$logf" | sed 's/^/    /'
      return 1
    fi
    sleep 5
  done
  log "!! 启动超时"
  return 1
}

stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
  fi
  SERVER_PID=""
  sleep 8   # 等显存释放
}

# ---------------- 指标 ----------------
metric() {  # metric <name>
  curl -s "http://localhost:$PORT/metrics" 2>/dev/null \
    | awk -v k="$1" '$0 ~ "^"k" " {print $2; exit}'
}

gpu_temp_max() {
  nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null \
    | sort -n | tail -1
}
gpu_clock_min() {
  nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null \
    | sort -n | head -1
}

# 从启动日志里抓真实 KV 容量（百万 token 的原始 token 数）
kv_tokens_from_log() {
  local logf="$1"
  grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$logf" 2>/dev/null \
    | tail -1 | grep -oE '[0-9,]+' | tr -d ','
}

# ---------------- 单次测量 ----------------
run_one() {
  local util="$1" ctx="$2" conc="$3" kv_tokens="$4" r="$5"
  local tag="u${util}_c${ctx}_b${conc}_r${r}"
  local json="$RAW/${tag}.json"
  local meta="$RAW/${tag}.meta"
  local n_prompts=$(( conc * 2 ))
  (( n_prompts < 32 )) && n_prompts=32

  local p0
  p0="$(metric vllm:num_preemptions_total)"; p0="${p0:-0}"

  log "  → ctx=$ctx conc=$conc prompts=$n_prompts"
  python "$BENCH" \
    --backend vllm \
    --base-url "http://localhost:$PORT" \
    --model "$MODEL" \
    --tokenizer "$MODEL" \
    --dataset-name random \
    --random-input-len "$ctx" \
    --random-output-len "$OUT_LEN" \
    --num-prompts "$n_prompts" \
    --max-concurrency "$conc" \
    --request-rate inf \
    --ignore-eos \
    --save-result \
    --result-filename "$json" \
    >"$LOGS/${tag}.bench.log" 2>&1
  local rc=$?

  local p1 cache tmax cmin
  p1="$(metric vllm:num_preemptions_total)"; p1="${p1:-0}"
  cache="$(metric vllm:gpu_cache_usage_perc)"; cache="${cache:-0}"
  tmax="$(gpu_temp_max)"; cmin="$(gpu_clock_min)"

  cat >"$meta" <<EOF
model=$MODEL
tp=$TP
util=$util
kv_tokens=$kv_tokens
ctx=$ctx
conc=$conc
num_prompts=$n_prompts
out_len=$OUT_LEN
preempt_delta=$(awk -v a="$p1" -v b="$p0" 'BEGIN{print a-b}')
cache_usage=$cache
gpu_temp=$tmax
gpu_clock=$cmin
rc=$rc
EOF

  [[ $rc -ne 0 ]] && log "  !! 本次失败（rc=$rc），已在 CSV 里标记"
  return 0
}

# ---------------- 主循环 ----------------
log "==== 扫描开始 model=$MODEL TP=$TP ===="
command -v vllm >/dev/null 2>&1 || { echo "找不到 vllm 命令，请先 pip install vllm"; exit 1; }
[[ -f "$BENCH" ]] || { echo "找不到 $BENCH，请设置 BENCH=/路径/vllm/benchmarks/benchmark_serving.py"; exit 1; }
for util in $UTILS; do
  srv_log="$LOGS/server_u${util}.log"
  if ! start_server "$util" "$srv_log"; then
    log "跳过 util=$util"
    continue
  fi
  kv_tokens="$(kv_tokens_from_log "$srv_log")"
  [[ -z "${kv_tokens:-}" ]] && kv_tokens=0
  log "真实 KV 容量：${kv_tokens} tokens"

  for ctx in $CTXS; do
    for conc in $CONCS; do
      for r in $(seq 1 "$REPEAT"); do
        run_one "$util" "$ctx" "$conc" "$kv_tokens" "$r"
      done
    done
  done
  stop_server
done

log "==== 汇总 ===="
python "$ROOT/plot_results.py" --ingest "$RAW" --out "$ROOT/results/results.csv"
python "$ROOT/plot_results.py" --csv "$ROOT/results/results.csv" --outdir "$ROOT/results/figs"
log "完成。结果：$ROOT/results/results.csv ；图：$ROOT/results/figs/"
