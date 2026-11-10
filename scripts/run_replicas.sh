#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# Run several full scaled-batch replicas back-to-back, unattended.
#
# Designed for the "collect N replicas then walk away" workflow on a
# remote VM. If a batch is already running (launched separately), it
# waits for that one to finish first and then runs the requested number
# of ADDITIONAL replicas. Each replica is a fresh run_scaled_batch.sh
# invocation (new session), so they don't collide and the per-(host,
# service, scale) rows accumulate cleanly.
#
# Usage:
#   scripts/run_replicas.sh [N_ADDITIONAL]   # default 2 (=> 3 total with one already running)
#
# Writes a marker output/batch_state/.all_replicas_done on completion so a
# watcher / auto-shutdown can confirm the work is finished.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

REMAINING="${1:-2}"
ACTIVE="output/batch_state/.active_session"
DONE_FLAG="output/batch_state/.all_replicas_done"
mkdir -p output/batch_state
rm -f "$DONE_FLAG"

if [ -f "$ACTIVE" ]; then
    echo "[replicas] $(date '+%F %T') waiting for the in-progress batch to finish..."
    while [ -f "$ACTIVE" ]; do sleep 60; done
    echo "[replicas] $(date '+%F %T') in-progress batch finished."
fi

for i in $(seq 1 "$REMAINING"); do
    echo "[replicas] $(date '+%F %T') ===== starting additional replica ${i}/${REMAINING} ====="
    if bash scripts/run_scaled_batch.sh; then
        echo "[replicas] $(date '+%F %T') additional replica ${i} completed."
    else
        echo "[replicas] $(date '+%F %T') additional replica ${i} reported failures (continuing)."
    fi
done

echo "[replicas] $(date '+%F %T') ALL replicas complete."
touch "$DONE_FLAG"
