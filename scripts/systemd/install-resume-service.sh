#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Install the on-boot auto-resume systemd service for the scaled
# batch runner. After a Spot eviction, simply `az vm start` the VM
# and the interrupted batch continues automatically.
#
# Run this ON the target VM (Linux/WSL2 with systemd), from anywhere:
#   bash ~/dataset-creator/scripts/systemd/install-resume-service.sh
#
# Idempotent. Requires sudo for writing to /etc/systemd/system.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/dataset-batch-resume.service"
UNIT_NAME="dataset-batch-resume.service"
UNIT_PATH="/etc/systemd/system/${UNIT_NAME}"

RUN_USER="${SUDO_USER:-$USER}"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"
RUN_HOME="${RUN_HOME:-$HOME}"

if [ ! -f "$TEMPLATE" ]; then
    echo "ERROR: template not found: $TEMPLATE" >&2
    exit 1
fi

if ! command -v systemctl >/dev/null 2>&1; then
    echo "ERROR: systemctl not found. This host has no systemd." >&2
    echo "       Fallback: after 'az vm start', SSH in and run:" >&2
    echo "         cd ~/dataset-creator && nohup bash scripts/run_scaled_batch.sh >> ~/batch_run.log 2>&1 &" >&2
    echo "       (RESUME=1 is the default, so it continues the interrupted batch.)" >&2
    exit 1
fi

echo "Installing ${UNIT_NAME} for user=${RUN_USER} home=${RUN_HOME}"
sed -e "s|__USER__|${RUN_USER}|g" -e "s|__HOME__|${RUN_HOME}|g" "$TEMPLATE" \
    | sudo tee "$UNIT_PATH" >/dev/null

sudo systemctl daemon-reload
sudo systemctl enable "$UNIT_NAME"

echo "✓ Installed and enabled."
echo "  It runs at boot ONLY if an interrupted batch is active"
echo "  (output/batch_state/.active_session exists)."
echo
echo "Verify:   systemctl status ${UNIT_NAME}"
echo "Dry test: sudo systemctl start ${UNIT_NAME}; journalctl -u ${UNIT_NAME} -n 20"
