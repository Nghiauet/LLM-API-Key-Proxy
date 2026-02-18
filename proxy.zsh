# =============================================================================
# LLM PROXY CONFIG
# =============================================================================
# Resolve project base dir. Accept explicit override, but if it points to this
# file (or is unset), normalize to the containing directory.
if [[ -z "${PROXY_BASE:-}" ]]; then
  export PROXY_BASE="${${(%):-%N}:A:h}"
elif [[ -f "$PROXY_BASE" ]]; then
  export PROXY_BASE="${PROXY_BASE:A:h}"
fi
export PROXY_ENV_FILE="${PROXY_ENV_FILE:-$PROXY_BASE/.env}"

_proxy_load_env() {
  local env_file line key value
  env_file="${PROXY_ENV_FILE:-$PROXY_BASE/.env}"
  export PROXY_ENV_FILE="$env_file"

  [ -f "$env_file" ] || return 0

  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    [[ -z "$line" ]] && continue
    [[ "$line" == \#* ]] && continue
    [[ "$line" != *=* ]] && continue

    key="${line%%=*}"
    value="${line#*=}"

    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue

    if [[ "$value" == \"*\" && "$value" == *\" && ${#value} -ge 2 ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" == \'*\' && ${#value} -ge 2 ]]; then
      value="${value:1:${#value}-2}"
    fi

    export "$key=$value"
  done < "$env_file"
}

# Load defaults from .env first, then fall back to hardcoded defaults.
_proxy_load_env
export PROXY_API_KEY="${PROXY_API_KEY:-REPLACE_WITH_PROXY_API_KEY}"
export PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
export PROXY_PORT="${PROXY_PORT:-8000}"
export PROXY_GIT_REPO="${PROXY_GIT_REPO:-https://github.com/Nghiauet/LLM-API-Key-Proxy.git}"
export PROXY_GIT_BRANCH="${PROXY_GIT_BRANCH:-main}"

# =============================================================================
# INTERNAL FUNCTIONS
# =============================================================================
_proxy_python() {
  # Ensure venv exists and return path to python
  if [ ! -x "$PROXY_BASE/venv/bin/python" ]; then
    echo " Creating venv in $PROXY_BASE/venv ..."
    (
      cd "$PROXY_BASE" || exit 1
      python3.12 -m venv venv 2>/dev/null || python3 -m venv venv || exit 1
      venv/bin/pip install --upgrade pip >/dev/null 2>&1
      [ -f requirements.txt ] && venv/bin/pip install -r requirements.txt >/dev/null 2>&1
    ) || { echo " Failed to create venv"; return 1; }
  fi
  echo "$PROXY_BASE/venv/bin/python"
}

_ensure_proxy() {
  if ! pgrep -f "proxy_app/main.py" >/dev/null 2>&1; then
    local py
    py="$(_proxy_python)" || return 1

    (
      cd "$PROXY_BASE" || exit 1
      PROXY_API_KEY="$PROXY_API_KEY" \
        nohup "$py" src/proxy_app/main.py \
          --host "$PROXY_HOST" \
          --port "$PROXY_PORT" \
          >/dev/null 2>&1 &
    )
  fi
}

# =============================================================================
# MAIN COMMAND
# =============================================================================
proxy() {
  # Reload .env on each command so runtime edits are picked up immediately.
  _proxy_load_env

  case "$1" in
    status)
      if pgrep -f "proxy_app/main.py" >/dev/null 2>&1; then
        echo " Proxy is running"
      else
        echo " Proxy is stopped"
      fi
      ;;

    version)
      if [ -f "$PROXY_BASE/src/proxy_app/main.py" ]; then
        (
          cd "$PROXY_BASE" || exit 1
          git log -1 --format=" Version: %h (%s, %ar)" 2>/dev/null
        )
      else
        echo " Proxy source not found at $PROXY_BASE"
      fi
      ;;

    start)
      if ! pgrep -f "proxy_app/main.py" >/dev/null 2>&1; then
        _ensure_proxy || return 1
        echo " Proxy started on http://$PROXY_HOST:$PROXY_PORT/v1"
      else
        echo " Proxy is already running"
      fi
      ;;

    stop)
      if pgrep -f "proxy_app/main.py" >/dev/null 2>&1; then
        pkill -f "proxy_app/main.py" 2>/dev/null
        echo " Proxy stopped"
      else
        echo " Proxy is not running"
      fi
      ;;

    restart)
      proxy stop
      proxy start
      ;;

    update)
      echo " Updating LLM-API-Key-Proxy..."

      pkill -f "proxy_app/main.py" 2>/dev/null
      local saved_key="$PROXY_API_KEY"

      mkdir -p "$(dirname "$PROXY_BASE")" || return 1

      if [ ! -d "$PROXY_BASE" ]; then
        git clone -b "$PROXY_GIT_BRANCH" \
          "$PROXY_GIT_REPO" \
          "$PROXY_BASE" || return 1
      fi

      cd "$PROXY_BASE" || return 1
      git fetch origin
      git checkout "$PROXY_GIT_BRANCH"
      git pull origin "$PROXY_GIT_BRANCH"

      # Ensure venv + deps
      _proxy_python >/dev/null || return 1

      # Ensure PROXY_API_KEY is present/updated in .env
      local env_file="${PROXY_ENV_FILE:-$PROXY_BASE/.env}"
      mkdir -p "$(dirname "$env_file")" || return 1
      if [ -f "$env_file" ]; then
        if grep -q "^PROXY_API_KEY=" "$env_file"; then
          sed -i.bak "s|^PROXY_API_KEY=.*|PROXY_API_KEY=\"$saved_key\"|" "$env_file"
          rm -f "${env_file}.bak"
        else
          echo "PROXY_API_KEY=\"$saved_key\"" >> "$env_file"
        fi
      else
        echo "PROXY_API_KEY=\"$saved_key\"" > "$env_file"
      fi

      echo " Updated successfully"
      ;;

    models)
      _ensure_proxy || return 1

      local response http_code models_json
      response=$(curl -s -w "\n%{http_code}" \
        -H "Authorization: Bearer $PROXY_API_KEY" \
        "http://$PROXY_HOST:$PROXY_PORT/v1/models" 2>/dev/null)

      http_code=$(echo "$response" | tail -1)
      models_json=$(echo "$response" | head -n -1)

      [[ "$http_code" != "200" ]] && echo " HTTP error $http_code" && return 1
      [ -z "$models_json" ] && echo " Empty response" && return 1

      python3 - "$models_json" <<'PY'
import json, sys, re
raw = sys.argv[1] if len(sys.argv) > 1 else ""
raw = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", raw)
data = json.loads(raw)

print(" Available models:")
for m in sorted(data.get("data", []), key=lambda x: x.get("id", "")):
    c = m.get("capabilities", {}) or {}
    flags = (
        ("[R]" if c.get("reasoning") else "") +
        ("[V]" if c.get("vision") else "") +
        ("[T]" if c.get("tool_choice") else "")
    )
    print(
        f"  • {m.get('id',''):50} "
        f"ctx:{m.get('context_length', 0):>8} "
        f"out:{m.get('max_completion_tokens', 0):>6} "
        f"{flags}"
    )
PY
      ;;

    ping)
      _ensure_proxy || return 1
      curl -s -o /dev/null -w " HTTP %{http_code}\n" \
        -H "Authorization: Bearer $PROXY_API_KEY" \
        "http://$PROXY_HOST:$PROXY_PORT/v1/models"
      ;;

    login)
      local py
      py="$(_proxy_python)" || return 1
      (cd "$PROXY_BASE" && "$py" src/proxy_app/main.py --add-credential)
      ;;

    tui)
      local py
      py="$(_proxy_python)" || return 1
      (cd "$PROXY_BASE" && "$py" src/proxy_app/main.py)
      ;;

    help|"")
      cat <<EOF
proxy {status|start|stop|restart|version|update|models|ping|login|tui|help}
  Base dir: $PROXY_BASE
  Env file: $PROXY_ENV_FILE
  API base: http://$PROXY_HOST:$PROXY_PORT/v1
  Update from: $PROXY_GIT_REPO ($PROXY_GIT_BRANCH)
EOF
      ;;

    *)
      echo " Unknown: $1. Use 'proxy help'"
      return 1
      ;;
  esac
}
