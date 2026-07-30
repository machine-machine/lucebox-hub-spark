#!/usr/bin/env bash
# switch_model.sh — swap which model owns the RTX 3090 (24 GB fits one at a time).
#
#   switch_model.sh status    # show all backends + health
#   switch_model.sh ornith    # Ornith-1.0-35B AEON (llama-server, :8181)   — fast coding, 153 tok/s
#   switch_model.sh laguna    # Laguna-XS.2 33B (dflash_server, :8182)      — long-context + PFlash
#   switch_model.sh qwen      # Qwen3.6-27B (legacy DFlash server, :8180)   — DFlash spec decode
#
# systemd Conflicts= relations stop the other backends automatically; this
# script just starts the requested unit, re-arms its healthcheck timer (timer
# Requires= pulls the service, not the reverse), waits for health, and prints
# client connection info. LiteLLM (:4000) stays up across switches.

set -euo pipefail

declare -A UNIT=(   [ornith]=ornith  [laguna]=laguna  [qwen]=dflash )
declare -A PORT=(   [ornith]=8181    [laguna]=8182    [qwen]=8180 )
declare -A HEALTH=( [ornith]=/health [laguna]=/health [qwen]=/v1/models )
declare -A TIMER=(  [ornith]=ornith-healthcheck.timer [laguna]="" [qwen]=dflash-healthcheck.timer )
declare -A MODELS=( [ornith]="coding | ornith-coding" [laguna]="laguna-coding" [qwen]="qwen3.6-coding" )

status() {
    local b st hl
    printf '%-8s %-10s %-6s %s\n' BACKEND UNIT PORT HEALTH
    for b in ornith laguna qwen; do
        st=$(systemctl --user is-active "${UNIT[$b]}" 2>/dev/null || true)
        hl=down
        curl -sf -m 2 "http://localhost:${PORT[$b]}${HEALTH[$b]}" >/dev/null 2>&1 && hl=healthy
        printf '%-8s %-10s %-6s %s\n' "$b" "$st" "${PORT[$b]}" "$hl"
    done
    printf 'litellm  %-10s 4000\n' "$(systemctl --user is-active litellm 2>/dev/null || true)"
}

case "${1:-status}" in
    status) status ;;
    ornith|laguna|qwen)
        m=$1
        echo "Starting ${UNIT[$m]}.service (Conflicts= auto-stops the other backends)..."
        systemctl --user start "${UNIT[$m]}"
        [[ -n "${TIMER[$m]}" ]] && systemctl --user start "${TIMER[$m]}" 2>/dev/null || true
        echo -n "Waiting for health on :${PORT[$m]} (model load can take 1-3 min) "
        ok=""
        for _ in $(seq 1 100); do
            if curl -sf -m 2 "http://localhost:${PORT[$m]}${HEALTH[$m]}" >/dev/null 2>&1; then
                ok=1; echo " ready"; break
            fi
            echo -n "."
            sleep 3
        done
        [[ -n "$ok" ]] || { echo " TIMED OUT — journalctl --user -u ${UNIT[$m]} -n 50"; exit 1; }
        echo
        status
        echo
        echo "Clients: http://$(hostname -I | awk '{print $1}'):4000/v1  api_key=sk-local  model: ${MODELS[$m]}"
        ;;
    *) echo "usage: $0 {ornith|laguna|qwen|status}" >&2; exit 1 ;;
esac
