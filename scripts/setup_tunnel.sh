#!/usr/bin/env bash
# setup_tunnel.sh — install a durable systemd --user SSH tunnel for the shared MLflow.
#
#     localhost:LOCAL_PORT  ->  SERVER:REMOTE_PORT
#
# Assumes your SSH key is ALREADY authorized on the server (key copy is skipped).
# Idempotent: safe to re-run. Auto-detects whether the key is passphrase-protected
# and writes the matching unit:
#   - passphrase-LESS key -> boot-clean unit (no agent, starts unattended at boot)
#   - passphrase key       -> unit reaches the key via your ssh-agent (needs ssh-add)
#
# Usage:
#   bash setup_tunnel.sh
#   KEY=~/.ssh/id_ed25519 SERVER=jash@10.21.186.205 LOCAL_PORT=5555 REMOTE_PORT=5555 \
#       bash setup_tunnel.sh
set -euo pipefail

SERVER="${SERVER:-jash@10.21.186.205}"
LOCAL_PORT="${LOCAL_PORT:-5555}"
REMOTE_PORT="${REMOTE_PORT:-5555}"
KEY="${KEY:-$HOME/.ssh/id_ed25519}"          # your already-authorized key
UNIT_NAME="mlflow-tunnel.service"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT_PATH="$UNIT_DIR/$UNIT_NAME"

c_info=$'\033[1;36m'; c_warn=$'\033[1;33m'; c_err=$'\033[1;31m'; c_off=$'\033[0m'
log()  { printf '%s[setup-tunnel]%s %s\n' "$c_info" "$c_off" "$*"; }
warn() { printf '%s[setup-tunnel] WARN:%s %s\n' "$c_warn" "$c_off" "$*"; }
die()  { printf '%s[setup-tunnel] ERROR:%s %s\n' "$c_err" "$c_off" "$*" >&2; exit 1; }

# --- preflight -------------------------------------------------------------
command -v ssh       >/dev/null || die "ssh not found"
command -v systemctl >/dev/null || die "systemctl not found (needs systemd --user)"
[[ -f "$KEY" ]]                 || die "key not found: $KEY  (set KEY=/path/to/your/key)"
SERVER_HOST="${SERVER#*@}"
log "server=$SERVER  forward=localhost:$LOCAL_PORT -> ${SERVER_HOST}:$REMOTE_PORT"
log "using existing key: $KEY"

# --- detect passphrase on the key -----------------------------------------
# `ssh-keygen -y -P ""` succeeds only when the key is UNENCRYPTED.
if ssh-keygen -y -P "" -f "$KEY" >/dev/null 2>&1; then
  ENCRYPTED=0; log "key is passphrase-less -> boot-clean unit (no agent needed)"
else
  ENCRYPTED=1; warn "key is passphrase-protected -> unit will reach it via ssh-agent"
fi

# --- sanity: can this key authenticate? (non-fatal) -----------------------
log "testing key auth to $SERVER ..."
if ssh -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 \
       -o StrictHostKeyChecking=accept-new "$SERVER" true 2>/dev/null; then
  log "key authenticates to server ✓"
else
  warn "could not authenticate non-interactively"
  warn "  (fine if it's a passphrase key & the agent isn't loaded yet; otherwise check the key is in ${SERVER}:~/.ssh/authorized_keys)"
fi

# --- write the unit --------------------------------------------------------
mkdir -p "$UNIT_DIR"
if [[ -f "$UNIT_PATH" ]]; then
  cp -a "$UNIT_PATH" "$UNIT_PATH.bak.$(date +%s)"
  warn "existing unit backed up -> $UNIT_PATH.bak.*"
fi

if [[ "$ENCRYPTED" == 1 ]]; then
  AGENT_LINE=$'# Passphrase key -> reached through ssh-agent via a stable symlink.\n# If the agent restarts: ln -sfn "$SSH_AUTH_SOCK" ~/.ssh/agent/current && systemctl --user restart '"$UNIT_NAME"$'\nEnvironment=SSH_AUTH_SOCK=%h/.ssh/agent/current'
else
  AGENT_LINE='# Passphrase-less key -> no agent required; starts clean at boot.'
fi

log "writing unit: $UNIT_PATH"
cat > "$UNIT_PATH" <<EOF
[Unit]
Description=MLflow SSH tunnel localhost:$LOCAL_PORT -> ${SERVER_HOST}:$REMOTE_PORT (baselines matrix)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
${AGENT_LINE}
# Foreground ssh (NO -f): systemd supervises it directly.
#  ServerAlive*           -> ssh exits within ~45s of a dead link; Restart revives it
#  ExitOnForwardFailure   -> if the local bind fails, ssh exits -> systemd retries
#  BatchMode              -> never block on a prompt (fail fast -> retry), no tty
ExecStart=/usr/bin/ssh -N -L $LOCAL_PORT:localhost:$REMOTE_PORT \\
  -i $KEY \\
  -o IdentitiesOnly=yes \\
  -o ExitOnForwardFailure=yes \\
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \\
  -o StrictHostKeyChecking=accept-new \\
  -o BatchMode=yes \\
  $SERVER
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
EOF

# --- passphrase key: make sure the agent symlink + key are ready ----------
if [[ "$ENCRYPTED" == 1 ]]; then
  mkdir -p "$HOME/.ssh/agent"
  if [[ -n "${SSH_AUTH_SOCK:-}" && -S "${SSH_AUTH_SOCK:-/nonexistent}" ]]; then
    ln -sfn "$SSH_AUTH_SOCK" "$HOME/.ssh/agent/current"
    log "pointed ~/.ssh/agent/current -> $SSH_AUTH_SOCK"
  else
    warn "no live ssh-agent in this shell — start one, load the key, point the symlink:"
    echo "      eval \"\$(ssh-agent -s)\"; ssh-add $KEY"
    echo "      ln -sfn \"\$SSH_AUTH_SOCK\" ~/.ssh/agent/current"
    echo "      systemctl --user restart $UNIT_NAME"
  fi
  if ! ssh-add -l >/dev/null 2>&1; then
    warn "key not loaded in agent yet -> run:  ssh-add $KEY"
  fi
fi

# --- enable + start --------------------------------------------------------
log "enabling linger (so the tunnel starts at boot without you logging in)"
loginctl enable-linger "$USER" 2>/dev/null || warn "could not enable linger"
systemctl --user daemon-reload
systemctl --user enable "$UNIT_NAME" >/dev/null 2>&1 || true
systemctl --user restart "$UNIT_NAME"
sleep 2

# --- health check ----------------------------------------------------------
systemctl --user --no-pager --lines=0 status "$UNIT_NAME" || true
if command -v curl >/dev/null; then
  ok=0
  for _ in $(seq 1 10); do
    curl -sf "http://localhost:$LOCAL_PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    sleep 1
  done
  if [[ "$ok" == 1 ]]; then
    log "✅ tunnel healthy:  http://localhost:$LOCAL_PORT/health -> OK"
    log "point experiments at:  export MLFLOW_TRACKING_URI=http://localhost:$LOCAL_PORT"
  else
    warn "tunnel not answering /health yet — inspect:"
    warn "      journalctl --user -u $UNIT_NAME -n 50 --no-pager"
  fi
else
  warn "curl not installed; verify manually:  curl http://localhost:$LOCAL_PORT/health"
fi

log "done. handy commands:"
echo "  systemctl --user status $UNIT_NAME"
echo "  journalctl --user -u $UNIT_NAME -f"
echo "  systemctl --user restart $UNIT_NAME"
