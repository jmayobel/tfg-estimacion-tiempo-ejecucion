#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Batch runner: invoke the orchestrator once per workload scale.
#
# Purpose
#   Produce the per-hardware "scaling curve" the prediction model
#   needs. One call here = N runs, one per preset in scripts/scales/.
#
# Resumability (for Spot/preemptible VMs)
#   Each completed scale writes a marker in output/batch_state/<session>/.
#   If the VM is evicted mid-batch and later restarted, re-running this
#   script RESUMES the same session: already-completed scales are skipped
#   and it continues with the remaining ones. This makes the batch safe to
#   re-launch after a Spot eviction without duplicating data.
#
# Env vars
#   SCALES        Space-separated scale names (default: "small medium large xlarge")
#   COOLDOWN_S    Seconds between runs (default: 120) so consecutive runs
#                 are independent.
#   ORCHESTRATOR  Script to invoke (default: scripts/orchestrator.sh).
#   BATCH_SESSION Session id. If unset, an active session is reused (resume)
#                 or a new timestamped one is created.
#   RESUME        1 (default) skip scales already marked done; 0 force all.
#   RESUME_ONLY   1 = only continue an existing active session; if none
#                 exists, exit 0 without starting a new batch. Used by the
#                 on-boot auto-resume service so a reboot never kicks off a
#                 brand-new batch by itself. Default 0.
#
# Examples
#   scripts/run_scaled_batch.sh                       # new (or resume active) batch
#   RESUME_ONLY=1 scripts/run_scaled_batch.sh         # continue only if interrupted
#   SCALES="medium large" COOLDOWN_S=60 scripts/run_scaled_batch.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output}"
STATE_DIR="${OUTPUT_DIR}/batch_state"
ACTIVE_FILE="${STATE_DIR}/.active_session"

SCALES="${SCALES:-small medium large xlarge}"
COOLDOWN_S="${COOLDOWN_S:-120}"
ORCHESTRATOR="${ORCHESTRATOR:-${SCRIPT_DIR}/orchestrator.sh}"
RESUME="${RESUME:-1}"
RESUME_ONLY="${RESUME_ONLY:-0}"

mkdir -p "$STATE_DIR"

if [ ! -x "$ORCHESTRATOR" ] && [ ! -f "$ORCHESTRATOR" ]; then
    echo "ERROR: orchestrator script not found at $ORCHESTRATOR" >&2
    exit 1
fi

# Verify all requested scale files exist before starting so we
# don't run for 2h and then fail on the last preset.
for scale in $SCALES; do
    if [ ! -f "${SCRIPT_DIR}/scales/${scale}.env" ]; then
        echo "ERROR: scale preset not found: scripts/scales/${scale}.env" >&2
        exit 1
    fi
done

# ------------------------------------------------------------
# Resolve the batch session (enables resume across evictions).
# ------------------------------------------------------------
if [ -n "${BATCH_SESSION:-}" ]; then
    SESSION="$BATCH_SESSION"
elif [ -f "$ACTIVE_FILE" ]; then
    SESSION="$(cat "$ACTIVE_FILE")"
else
    if [ "$RESUME_ONLY" = "1" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] RESUME_ONLY=1 and no active batch session — nothing to resume. Exiting."
        exit 0
    fi
    SESSION="batch-$(date '+%Y%m%d-%H%M%S')"
fi

SESSION_DIR="${STATE_DIR}/${SESSION}"
mkdir -p "$SESSION_DIR"
echo "$SESSION" > "$ACTIVE_FILE"

# ------------------------------------------------------------
# Compute which scales still need to run.
# ------------------------------------------------------------
remaining=""
skipped=""
for scale in $SCALES; do
    if [ "$RESUME" = "1" ] && [ -f "${SESSION_DIR}/${scale}.done" ]; then
        skipped="${skipped} ${scale}"
    else
        remaining="${remaining} ${scale}"
    fi
done
remaining="$(echo "$remaining" | xargs)"   # trim
skipped="$(echo "$skipped" | xargs)"

START_TS="$(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo "  Scaled batch run"
echo "  Session:   $SESSION"
echo "  Scales:    $SCALES"
[ -n "$skipped" ] && echo "  Already done (skipped on resume): $skipped"
echo "  To run:    ${remaining:-<none>}"
echo "  Cooldown:  ${COOLDOWN_S}s between scales"
echo "  Started:   $START_TS"
echo "============================================================"

if [ -z "$remaining" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] All scales already completed for session $SESSION. Clearing active marker."
    rm -f "$ACTIVE_FILE"
    exit 0
fi

total=0
ok=0
fail=0
n_remaining="$(echo "$remaining" | wc -w)"

for scale in $remaining; do
    total=$((total + 1))
    echo
    echo "------------------------------------------------------------"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting scale: ${scale}  (session $SESSION)"
    echo "------------------------------------------------------------"

    # Run the orchestrator with the scale preset sourced into its env.
    # Subshell so env doesn't leak between iterations.
    if (
        set -a
        if [ -f "${PROJECT_DIR}/.env" ]; then
            # shellcheck disable=SC1091
            source "${PROJECT_DIR}/.env"
        fi
        # shellcheck disable=SC1090
        source "${SCRIPT_DIR}/scales/${scale}.env"
        set +a
        bash "$ORCHESTRATOR"
    ); then
        ok=$((ok + 1))
        touch "${SESSION_DIR}/${scale}.done"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scale ${scale} completed (marked done)."
    else
        fail=$((fail + 1))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scale ${scale} FAILED." >&2
    fi

    # Cool down between scales — but skip after the last one.
    if [ "$total" -lt "$n_remaining" ] && [ "$COOLDOWN_S" -gt 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cooldown ${COOLDOWN_S}s before next scale..."
        sleep "$COOLDOWN_S"
    fi
done

# ------------------------------------------------------------
# If every requested scale now has a marker, the batch is complete:
# clear the active session so a future launch starts fresh (and the
# on-boot auto-resume service stays idle).
# ------------------------------------------------------------
all_done=1
for scale in $SCALES; do
    [ -f "${SESSION_DIR}/${scale}.done" ] || all_done=0
done
if [ "$all_done" = "1" ]; then
    rm -f "$ACTIVE_FILE"
    SESSION_STATE="COMPLETE — session cleared"
else
    SESSION_STATE="INCOMPLETE — session $SESSION still active (re-run to resume)"
fi

END_TS="$(date '+%Y-%m-%d %H:%M:%S')"
echo
echo "============================================================"
echo "  Batch finished"
echo "  Session:  $SESSION"
echo "  Started:  $START_TS"
echo "  Ended:    $END_TS"
echo "  Results:  ${ok}/${total} ok, ${fail} failed"
echo "  State:    $SESSION_STATE"
echo "============================================================"

if [ "$fail" -gt 0 ]; then
    exit 1
fi
