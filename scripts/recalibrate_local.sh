#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# TFG-Benchmark - Local Host Recalibrator
# 
# Runs the containerized calibration (pinned numpy==1.26.4) and backfills
# an existing datasets/<host>/ml_dataset.csv with standardized metrics.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <host-directory-name>"
    echo "Example: $0 macbook-air-m4"
    echo ""
    echo "Available host directories in datasets/:"
    ls -d "${PROJECT_DIR}/datasets"/*/ | xargs -n1 basename
    exit 1
fi

HOST_NAME="$1"
HOST_DIR="${PROJECT_DIR}/datasets/${HOST_NAME}"

if [ ! -d "$HOST_DIR" ]; then
    echo "ERROR: Directory $HOST_DIR does not exist."
    exit 1
fi

CSV_FILE="${HOST_DIR}/ml_dataset.csv"
if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: $CSV_FILE not found. Cannot backfill a missing dataset."
    exit 1
fi

echo "=============================================================================="
echo " Recalibrating host '$HOST_NAME' using containerized Python + NumPy runtime..."
echo "=============================================================================="

TEMP_CALIB="${PROJECT_DIR}/output/temp_calibration_${HOST_NAME}.json"
mkdir -p "${PROJECT_DIR}/output"

# Execute calibration in python:3.10-slim container
docker run --rm \
    -v "${PROJECT_DIR}:/workspace" \
    -w /workspace \
    python:3.10-slim \
    bash -c "pip install -q numpy==1.26.4 && python3 scripts/calibrate.py --print" > "$TEMP_CALIB"

# Verify JSON validity by searching for a key calibration metric
if ! grep -q "calib_gflops_st" "$TEMP_CALIB"; then
    echo "ERROR: Calibration in container failed or returned corrupt JSON."
    cat "$TEMP_CALIB"
    exit 1
fi

echo "In-container calibration output captured."
cat "$TEMP_CALIB"
echo "------------------------------------------------------------------------------"
echo "Injecting calibration metrics into ${CSV_FILE}..."
python3 "${SCRIPT_DIR}/backfill_calibration.py" --host "$HOST_DIR" --calib-json "$TEMP_CALIB"

# Cleanup
rm -f "$TEMP_CALIB"
echo "=============================================================================="
echo " ✓ Standardized recalibration and backfill complete for $HOST_NAME!"
echo "=============================================================================="
