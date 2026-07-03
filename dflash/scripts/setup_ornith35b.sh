#!/usr/bin/env bash
# Setup + serve Ornith-1.0-35B (AEON Ultimate Uncensored, GGUF) via llama-server.
#
# Usage (from repo root or dflash/):
#   bash dflash/scripts/setup_ornith35b.sh
#
# Why llama-server and not the DFlash binary:
#   Ornith-1.0-35B is a MoE (Qwen3.6-35B-A3B base) with a GatedDeltaNet SSM
#   layer — the qwen3next GGUF arch. The DFlash C++ engine only implements
#   dense Qwen3.5/3.6-27B, and its speculative draft is trained against
#   Qwen3.6-27B specifically. With only ~3B active params the MoE decodes
#   fast without speculation, so plain llama-server is the right engine.
#
# The NVFP4 upload of this model (AEON-7/...-NVFP4) requires Blackwell GPUs
# (sm_100/sm_120) and cannot run on the RTX 3090 (sm_86); this script uses
# the GGUF conversion of the same finetune instead.
#
# What this does:
#   1. Load HF_TOKEN from ~/.bashrc
#   2. Build llama-server from deps/llama.cpp (CUDA sm_86)
#   3. Download the Ornith GGUF quant (IQ4_XS, ~18.7 GB — fits 24 GB VRAM
#      with room for KV cache; Q4_K_M at 21.2 GB does not)
#   4. Smoke-test a completion through the server
#   5. Leave llama-server running on ORNITH_PORT for LiteLLM to route to
#
# Environment overrides:
#   CUDA_ARCH     CMake CUDA arch (default: 86 = RTX 3090)
#   ORNITH_QUANT  GGUF quant to download (default: IQ4_XS)
#   ORNITH_PORT   llama-server port (default: 8181 — DFlash keeps 8180)
#   ORNITH_CTX    context length (default: 32768)
#   N_CPU_MOE     expert layers to keep on CPU (default: 0; raise if OOM)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DFLASH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()    { printf '\033[1;32m[OK]\033[0m    %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
die()   { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

CUDA_ARCH="${CUDA_ARCH:-86}"
ORNITH_QUANT="${ORNITH_QUANT:-IQ4_XS}"
ORNITH_PORT="${ORNITH_PORT:-8181}"
ORNITH_CTX="${ORNITH_CTX:-32768}"
N_CPU_MOE="${N_CPU_MOE:-0}"

ORNITH_REPO="vcruz305/Ornith-1.0-35B-AEON-Ultimate-Uncensored-GGUF"
ORNITH_FILE="ornith-aeon-35b-${ORNITH_QUANT}.gguf"
MODEL_PATH="$DFLASH_DIR/models/$ORNITH_FILE"

# ── 1. HF_TOKEN ───────────────────────────────────────────────────────────────

if [[ -z "${HF_TOKEN:-}" ]]; then
    # shellcheck source=/dev/null
    source "$HOME/.bashrc" 2>/dev/null || true
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    die "HF_TOKEN not set. Add 'export HF_TOKEN=hf_...' to ~/.bashrc and re-run."
fi
info "HF_TOKEN loaded (${HF_TOKEN:0:8}...)"

# ── 2. Build llama-server ─────────────────────────────────────────────────────

cd "$DFLASH_DIR"

info "Initialising git submodules..."
git submodule update --init --recursive

LLAMA_BUILD="$DFLASH_DIR/deps/llama.cpp/build-server"
info "Configuring llama.cpp server build (sm_${CUDA_ARCH})..."
cmake -B "$LLAMA_BUILD" -S deps/llama.cpp \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DLLAMA_BUILD_TESTS=OFF

info "Building llama-server..."
cmake --build "$LLAMA_BUILD" --target llama-server -j
LLAMA_SERVER="$LLAMA_BUILD/bin/llama-server"
[[ -x "$LLAMA_SERVER" ]] || die "llama-server not found at $LLAMA_SERVER"
ok "llama-server built."

# ── 3. Download model ─────────────────────────────────────────────────────────

mkdir -p models
if [[ -f "$MODEL_PATH" ]]; then
    ok "Model already present: $MODEL_PATH"
else
    info "Downloading $ORNITH_REPO :: $ORNITH_FILE (~19 GB for IQ4_XS)..."
    python3 - "$ORNITH_REPO" "$ORNITH_FILE" "$DFLASH_DIR/models" <<'PYEOF'
import sys, os
from huggingface_hub import hf_hub_download
repo_id, filename, local_dir = sys.argv[1], sys.argv[2], sys.argv[3]
path = hf_hub_download(repo_id=repo_id, filename=filename,
                       local_dir=local_dir, token=os.environ["HF_TOKEN"])
print(path)
PYEOF
    ok "Model downloaded."
fi

# ── 4. Launch llama-server ────────────────────────────────────────────────────

MOE_ARGS=()
if [[ "$N_CPU_MOE" -gt 0 ]]; then
    MOE_ARGS=(--n-cpu-moe "$N_CPU_MOE")
    info "Keeping $N_CPU_MOE expert layers on CPU."
fi

info "Starting llama-server on port ${ORNITH_PORT} (ctx=${ORNITH_CTX})..."
"$LLAMA_SERVER" \
    --model "$MODEL_PATH" \
    --alias luce-ornith \
    --host 0.0.0.0 --port "$ORNITH_PORT" \
    --ctx-size "$ORNITH_CTX" \
    --n-gpu-layers 999 \
    --flash-attn on \
    --jinja \
    "${MOE_ARGS[@]}" &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM

# ── 5. Smoke test ─────────────────────────────────────────────────────────────

info "Waiting for server to load the model (can take ~1 min)..."
for i in $(seq 1 120); do
    if curl -sf "http://localhost:${ORNITH_PORT}/health" >/dev/null 2>&1; then
        break
    fi
    kill -0 "$SERVER_PID" 2>/dev/null || die "llama-server exited during load — check output above (OOM? try N_CPU_MOE=4)"
    sleep 2
done
curl -sf "http://localhost:${ORNITH_PORT}/health" >/dev/null \
    || die "server did not become healthy within 4 min"

# Ornith is a thinking model: tokens go to reasoning_content first, so the
# budget must be large enough to reach the final answer.
RESP=$(curl -sf "http://localhost:${ORNITH_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"luce-ornith","messages":[{"role":"user","content":"Say OK."}],"max_tokens":512}') \
    || die "smoke-test completion failed"
REPLY=$(python3 -c 'import json,sys; m=json.loads(sys.argv[1])["choices"][0]["message"]; print(m.get("content") or "(empty — raise max_tokens; reasoning used the budget)")' "$RESP")
ok "Smoke test reply: $REPLY"

ok "Ornith-1.0-35B serving on http://localhost:${ORNITH_PORT}/v1 (alias: luce-ornith)"
info "Route through LiteLLM: litellm --config $DFLASH_DIR/litellm_config.yaml --port 4000"
info "  → model 'coding' now maps to Ornith; 'qwen3.6-coding' still maps to DFlash on 8180."

wait "$SERVER_PID"
