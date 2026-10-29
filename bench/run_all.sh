#!/usr/bin/env bash
# Full benchmark suite against a running server. Writes one report per scenario
# into bench/results/ and prints a summary at the end.
#
#   bench/run_all.sh [url]
#
# Keep the machine otherwise idle: the model saturates one GPU, so any other
# GPU work skews these numbers.
set -u
URL="${1:-http://localhost:9001}"
PY="${PY:-server/.venv/bin/python}"
OUT="bench/results"
mkdir -p "$OUT"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$OUT/run-$STAMP.log"

say() { printf '\n\033[1m%s\033[0m\n' "$*" | tee -a "$LOG"; }

run() {
  local name="$1"; shift
  echo "--- $name: $*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG" | grep -E "mode=|throughput|latency|first chunk|audio per request|errors|^  [0-9]" || true
}

curl -sf "$URL/readyz" >/dev/null || { echo "server not ready at $URL"; exit 1; }

say "health"
curl -s "$URL/health" | tee -a "$LOG" | python3 -m json.tool | head -40

say "single-request latency vs num_step (full, 8 requests)"
for steps in 8 12 16 24 32; do
  run "full-c1-steps$steps" $PY bench/load_test.py --url "$URL" --mode full --concurrency 1 \
      --requests 8 --corpus en --num-step "$steps"
done

say "concurrency sweep, non-streaming (short sentences)"
for c in 1 2 4 8 16; do
  run "full-c$c" $PY bench/load_test.py --url "$URL" --mode full --concurrency "$c" --requests $((c * 6)) --corpus en
done

say "concurrency sweep, SSE streaming"
for c in 1 2 4 8 16; do
  run "sse-c$c" $PY bench/load_test.py --url "$URL" --mode sse --concurrency "$c" --requests $((c * 6)) --corpus en
done

say "WebSocket, LLM-style token input (8 words/s)"
for c in 1 4 8; do
  run "ws-c$c" $PY bench/load_test.py --url "$URL" --mode ws --concurrency "$c" --requests $((c * 3)) \
      --corpus en --token-words 1 --token-delay 0.125
done

say "Slovak"
for c in 1 4; do
  run "sk-full-c$c" $PY bench/load_test.py --url "$URL" --mode full --concurrency "$c" --requests $((c * 6)) \
      --corpus sk --lang sk --voice kroner
  run "sk-sse-c$c" $PY bench/load_test.py --url "$URL" --mode sse --concurrency "$c" --requests $((c * 6)) \
      --corpus sk --lang sk --voice kroner
done

say "long text (28 s of audio per request)"
for c in 1 4 8; do
  run "long-c$c" $PY bench/load_test.py --url "$URL" --mode full --concurrency "$c" --requests $((c * 3)) --corpus long
done

say "overload (64 concurrent long requests)"
run "overload" $PY bench/load_test.py --url "$URL" --mode full --concurrency 64 --requests 64 --corpus long

say "soak: 60 s at concurrency 8, mixed corpora"
run "soak" $PY bench/load_test.py --url "$URL" --mode sse --concurrency 8 --duration 60 --corpus mixed

say "GPU state"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | tee -a "$LOG"

say "done — full log: $LOG"
