#!/usr/bin/env bash
# Healthcheck for the Ornith llama-server (:8181).
# Restarts ornith if the endpoint is unreachable, returns 5xx, or returns a
# 200 with zero completion_tokens.
#
# Grace period: llama-server answers 503 while loading the 18.7 GB GGUF
# (~1-2 min cold). Restarting during load would loop forever, so any failure
# within GRACE_SECONDS of the service becoming active is ignored.

set -euo pipefail

ENDPOINT="http://localhost:8181/v1/chat/completions"
PAYLOAD='{"model":"luce-ornith","messages":[{"role":"user","content":"ping"}],"max_tokens":3}'
LOG_TAG="ornith-healthcheck"
GRACE_SECONDS=180

started_usec=$(systemctl --user show ornith --property=ActiveEnterTimestampMonotonic --value)
if [[ -n "$started_usec" && "$started_usec" != "0" ]]; then
    now_usec=$(awk '{printf "%d", $1 * 1000000}' /proc/uptime)
    age=$(( (now_usec - started_usec) / 1000000 ))
    if (( age < GRACE_SECONDS )); then
        systemd-cat -t "$LOG_TAG" -p debug \
            echo "ornith up ${age}s < ${GRACE_SECONDS}s grace — skipping check"
        exit 0
    fi
fi

response=$(curl -s --max-time 30 \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
    -w "\n%{http_code}" \
    "$ENDPOINT" 2>/dev/null || echo -e "\n000")

http_code=$(tail -1 <<< "$response")
body=$(head -1 <<< "$response")

if [[ "$http_code" =~ ^5 ]] || [[ "$http_code" == "000" ]]; then
    systemd-cat -t "$LOG_TAG" -p err \
        echo "Ornith HTTP $http_code — restarting"
    systemctl --user restart ornith
    exit 0
fi

# completion_tokens counts reasoning tokens too, so >0 holds even though a
# 3-token reply from this thinking model has empty visible content.
completion_tokens=$(python3 -c "
import sys, json
try:
    d = json.loads('''$body''')
    print(d.get('usage', {}).get('completion_tokens', 0))
except Exception:
    print(0)
" 2>/dev/null || echo 0)

if [[ "$completion_tokens" -eq 0 ]]; then
    systemd-cat -t "$LOG_TAG" -p err \
        echo "Ornith returned 0 completion_tokens — restarting"
    systemctl --user restart ornith
else
    systemd-cat -t "$LOG_TAG" -p debug \
        echo "Ornith OK (HTTP $http_code, completion_tokens=$completion_tokens)"
fi
