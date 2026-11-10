#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Dataset Creator - Orchestrator
# Runs benchmark services sequentially, collects metrics via
# Prometheus, and exports to CSV.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/output"

# Each run gets its own timestamped subfolder so data accumulates across runs
RUN_ID="$(date '+%Y-%m-%d_%H-%M-%S')"
RUN_OUTPUT_DIR="${OUTPUT_DIR}/runs/${RUN_ID}"
LOG_FILE="${RUN_OUTPUT_DIR}/orchestrator.log"

# Ensure output dirs exist early so we can write the log
mkdir -p "$RUN_OUTPUT_DIR"

# Load config — but skip if WORKLOAD_SCALE is already in the environment,
# which means run_scaled_batch.sh already loaded .env *and* a scale preset
# on top. Re-sourcing .env here would clobber the preset overrides and the
# run would silently use the baseline workload sizes.
if [ -f "${PROJECT_DIR}/.env" ] && [ -z "${WORKLOAD_SCALE:-}" ]; then
    set -a
    source "${PROJECT_DIR}/.env"
    set +a
fi

# Defaults (override in .env)
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-2}"
HOSTNAME_LABEL="${HOSTNAME_LABEL:-$(hostname)}"

# Name of the scale preset used (small | medium | large | xlarge | custom).
# Recorded into run_metadata.json so build_ml_dataset.py can join scale
# back onto every (run, benchmark) row.
WORKLOAD_SCALE="${WORKLOAD_SCALE:-custom}"

# In Docker Desktop for Mac/Windows, host networking doesn't work like Linux.
# However, since Prometheus is published on the host, and we are using network_mode: "host",
# localhost actually works for reaching published ports on Docker Desktop.
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"

WAIT_AFTER_START="${WAIT_AFTER_START:-10}"

# Safety timeout to kill hung services (seconds, default 1 hour)
MAX_SERVICE_TIMEOUT="${MAX_SERVICE_TIMEOUT:-3600}"

# Per-service workload sizes (override in .env)
OLLAMA_REQUESTS="${OLLAMA_REQUESTS:-50}"
BUSQUEDA_REQUESTS="${BUSQUEDA_REQUESTS:-5}"
IPERF_BYTES="${IPERF_BYTES:-100G}"
SYSBENCH_EVENTS="${SYSBENCH_EVENTS:-10000}"
NETWORK_PING_COUNT="${NETWORK_PING_COUNT:-100}"
PYTORCH_GPU_EPOCHS="${PYTORCH_GPU_EPOCHS:-10}"
STRESSNG_OPS="${STRESSNG_OPS:-100000}"
FFMPEG_VIDEO_LENGTH="${FFMPEG_VIDEO_LENGTH:-120}"
K6_ITERATIONS="${K6_ITERATIONS:-1000}"
FIO_SIZE="${FIO_SIZE:-1G}"
ZIP_ROUNDS="${ZIP_ROUNDS:-3}"
BUILD_ROUNDS="${BUILD_ROUNDS:-3}"
REDIS_REQUESTS="${REDIS_REQUESTS:-1000000}"

# Ollama model to benchmark (pulled and run automatically)
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:1b}"

# Busqueda Caminos configuration
BUSQUEDA_URL="${BUSQUEDA_URL:-http://localhost:5005/process}"
BUSQUEDA_INSTANCE_FILE="${BUSQUEDA_INSTANCE_FILE:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging helpers
# - log/info/warn write to both the log file AND console
# - detail writes ONLY to the log file (suppressed from console)
# - err writes to console stderr AND log file
_ts() { date '+%Y-%m-%d %H:%M:%S'; }
_logfile() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"; }

log()    { local msg="[${GREEN}$( _ts )${NC}] $*"; echo -e "$msg"; _logfile "$*"; }
warn()   { local msg="[${YELLOW}$( _ts ) WARNING${NC}] $*"; echo -e "$msg"; _logfile "WARNING: $*"; }
err()    { local msg="[${RED}$( _ts ) ERROR${NC}] $*"; echo -e "$msg" >&2; _logfile "ERROR: $*"; }
info()   { local msg="[${BLUE}$( _ts )${NC}] $*"; echo -e "$msg"; _logfile "$*"; }
detail() { _logfile "$*"; }  # log-file only, no console output

# ============================================================
# Platform Detection
# ============================================================
detect_platform() {
    detail "Detecting platform..."

    PLATFORM="unknown"
    HAS_NVIDIA_GPU=false
    HAS_APPLE_NPU=false
    CPU_NAME="unknown"
    GPU_NAME="none"
    NPU_NAME="none"

    # Allow override via environment variable for Dockerized orchestrator
    if [[ "${HOST_OS:-}" == "darwin"* || "${HOST_OS:-}" == "macos" ]]; then
        PLATFORM="macos"
        detail "Platform: macOS (via HOST_OS)"
    elif [[ "${HOST_OS:-}" == "wsl2" ]] || grep -qi microsoft /proc/version 2>/dev/null; then
        PLATFORM="wsl2"
        detail "Platform: Windows 11 (WSL2)"
    elif [[ "$(uname)" == "Darwin" ]]; then
        PLATFORM="macos"
        detail "Platform: macOS"
    else
        PLATFORM="linux"
        detail "Platform: Linux"
    fi

    # Check for NVIDIA GPU (local or via Docker)
    if command -v nvidia-smi &>/dev/null; then
        HAS_NVIDIA_GPU=true
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "NVIDIA GPU")
        detail "NVIDIA GPU detected: ${GPU_NAME}"
    elif docker info 2>/dev/null | grep -qi "Runtimes.*nvidia"; then
        HAS_NVIDIA_GPU=true
        # nvidia-smi not on host (e.g., inside orchestrator container); query via Docker
        GPU_NAME=$(docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 \
            nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
        GPU_NAME="${GPU_NAME:-NVIDIA GPU (detected via Docker runtime)}"
        detail "NVIDIA GPU detected: ${GPU_NAME}"
    fi

    # Check for Apple Silicon / NPU
    if [[ "$PLATFORM" == "macos" ]]; then
        if sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -qi "apple" || [[ "$(docker info --format '{{.Architecture}}' 2>/dev/null)" == "aarch64" ]]; then
            HAS_APPLE_NPU=true
            NPU_NAME="Apple Neural Engine"
            CPU_NAME=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
            GPU_NAME="Apple GPU"
            detail "Apple Silicon detected with NPU: ${NPU_NAME}"
        fi
    fi

    # Get CPU name on Linux/WSL
    if [[ "$PLATFORM" == "wsl2" || "$PLATFORM" == "linux" ]]; then
        CPU_NAME=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo "Unknown CPU")
    fi

    info "Platform: ${PLATFORM} | CPU: ${CPU_NAME} | GPU: ${GPU_NAME} | NPU: ${NPU_NAME}"
}

# ============================================================
# Service Definitions
# ============================================================
# Format: service_key|profile|service_type|requires_gpu|is_gpu_variant|container_name|compose_service
# compose_service = the service name in docker-compose.services.yml (used by --exit-code-from)
declare -a SERVICES=(
    "ollama|ollama|Model ML|true|false|ds-ollama|ollama"
    "busqueda-caminos|busqueda-caminos|Compute|false|false|ds-busqueda-caminos|busqueda-caminos"
    "iperf3|iperf3|Network Benchmark|false|false|ds-iperf3-client|iperf3-client"
    "sysbench|sysbench|CPU Benchmark|false|false|ds-sysbench|ds-sysbench"
    "network-tools|network-tools|Network Tools|false|false|ds-network-tools|network-tools"
    "pytorch-gpu|pytorch-gpu|GPU Benchmark|true|true|ds-pytorch-gpu|pytorch-gpu-bench"
    "stress-ng|stress-ng|Stress Test|false|false|ds-stress-ng|stress-ng"
    "ffmpeg|ffmpeg|Media Processing|false|false|ds-ffmpeg|ffmpeg"
    "k6|k6|Load Testing|false|false|ds-k6|ds-k6"
    "ds-fio|fio|Disk Benchmark|false|false|ds-fio|ds-fio"
    "ds-7zip|7zip|Compression Benchmark|false|false|ds-7zip|ds-7zip"
    "ds-build|build|Compilation Benchmark|false|false|ds-build|ds-build"
    "ds-redis-benchmark|redis-benchmark|Memory Benchmark|false|false|ds-redis-benchmark|ds-redis-benchmark"
)

# ============================================================
# Monitoring Stack Management
# ============================================================
start_monitoring() {
    log "Starting monitoring stack..."

    cd "$PROJECT_DIR"

    # Start base monitoring
    docker compose -p ds-monitoring -f docker-compose.monitoring.yml up -d --quiet-pull --remove-orphans >> "$LOG_FILE" 2>&1

    # Start GPU exporter if NVIDIA GPU available
    if $HAS_NVIDIA_GPU; then
        detail "Starting NVIDIA DCGM exporter..."
        docker compose -p ds-gpu -f docker-compose.gpu.yml up -d --quiet-pull --remove-orphans >> "$LOG_FILE" 2>&1 || warn "DCGM exporter failed to start"
    fi

    # Start Apple host metrics exporter if on macOS
    if $HAS_APPLE_NPU; then
        detail "Starting Apple Silicon host metrics exporter..."
        info "Run on the macOS host: sudo python3 ${SCRIPT_DIR}/host_metrics.py"
        detail "Waiting up to ${WAIT_AFTER_START}s for host exporter to start..."
        
        # In Docker Desktop for Mac, host.docker.internal points to the host machine
        local host_url="http://localhost:9101/metrics"
        if [[ "${HOST_OS:-}" == "macos" ]]; then
            host_url="http://host.docker.internal:9101/metrics"
        fi

        local host_retries="$WAIT_AFTER_START"
        while [ "$host_retries" -gt 0 ]; do
            if curl -sf "$host_url" &>/dev/null; then
                detail "Host exporter is ready"
                break
            fi
            host_retries=$((host_retries - 1))
            sleep 1
        done
        if [ "$host_retries" -le 0 ]; then
            warn "Host exporter not responding on $host_url; continuing"
        fi
    fi

    # Wait for Prometheus to be ready
    detail "Waiting for Prometheus to be ready at ${PROMETHEUS_URL}..."
    local retries=30
    while ! curl -sf "${PROMETHEUS_URL}/-/ready" &>/dev/null; do
        retries=$((retries - 1))
        if [ $retries -le 0 ]; then
            err "Prometheus failed to start or is unreachable at ${PROMETHEUS_URL}"
            exit 1
        fi
        sleep 2
    done
    log "Monitoring stack ready (Prometheus at ${PROMETHEUS_URL})"
}

stop_monitoring() {
    detail "Stopping monitoring stack..."
    cd "$PROJECT_DIR"

    if $HAS_NVIDIA_GPU; then
        docker compose -p ds-gpu -f docker-compose.gpu.yml down --remove-orphans >> "$LOG_FILE" 2>&1 || true
    fi
    docker compose -p ds-monitoring -f docker-compose.monitoring.yml down --remove-orphans >> "$LOG_FILE" 2>&1
}

# ============================================================
# Service Execution
# ============================================================
get_service_size() {
    local container_name="$1"
    # Get the image size in MB
    docker inspect --format='{{.Image}}' "$container_name" 2>/dev/null | \
        xargs docker image inspect --format='{{.Size}}' 2>/dev/null | \
        awk '{printf "%.2f", $1/1048576}' || echo "0"
}

get_num_params() {
    local service_key="$1"
    # Only applicable for Ollama (model parameters)
    if [[ "$service_key" == "ollama" ]]; then
        # Try to get model info from Ollama API
        local info
        local actual_ollama_url="http://localhost:11434/api/show"
        if [[ -n "${HOST_OS:-}" && "${HOST_OS:-}" != "linux" ]]; then
            actual_ollama_url="http://host.docker.internal:11434/api/show"
        fi
        info=$(curl -sf "$actual_ollama_url" -d "{\"name\": \"${OLLAMA_MODEL}\"}" 2>/dev/null || echo "{}")
        echo "$info" | python3 -c 'import sys,json;d=json.load(sys.stdin);p=d.get("details",{}).get("parameter_size","0");s=p.upper().replace("B","e9").replace("M","e6").replace("K","e3");print(float(eval(s)))' 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

determine_architecture_type() {
    local service_key="$1"
    local requires_gpu="$2"

    if [[ "$requires_gpu" == "true" ]]; then
        if $HAS_NVIDIA_GPU; then
            echo "GPU"
        elif $HAS_APPLE_NPU; then
            echo "NPU"
        else
            echo "CPU"
        fi
    else
        echo "CPU"
    fi
}

run_service() {
    local service_key="$1"
    local profile="$2"
    local service_type="$3"
    local requires_gpu="$4"
    local is_gpu_variant="$5"
    local container_name="$6"
    local compose_service="$7"

    # Skip GPU-only services if no GPU
    if [[ "$is_gpu_variant" == "true" && "$HAS_NVIDIA_GPU" == "false" ]]; then
        warn "Skipping ${service_key} (requires NVIDIA GPU)"
        return 0
    fi

    # Handle Ollama: use native macOS Ollama if running on host, otherwise CPU container
    local actual_profile="$profile"
    local use_native_ollama=false
    if [[ "$service_key" == "ollama" && "$PLATFORM" == "macos" ]]; then
        # Check both localhost (if orchestrator is run natively) and host.docker.internal (if inside docker)
        if curl -sf http://localhost:11434/api/tags &>/dev/null || curl -sf http://host.docker.internal:11434/api/tags &>/dev/null; then
            use_native_ollama=true
            actual_profile="none"
            detail "Detected native Ollama running on macOS host. Leveraging native NPU/GPU acceleration."
        else
            actual_profile="ollama-cpu"
            warn "Native Ollama not running on macOS host. Falling back to CPU-only container."
        fi
    elif [[ "$service_key" == "ollama" && "$HAS_NVIDIA_GPU" == "false" ]]; then
        actual_profile="ollama-cpu"
    fi

    # Handle ffmpeg: use GPU variant if available. Promote requires_gpu so
    # the architecture_type column in the CSV (and the [START] log line)
    # reflect that this run actually went through NVENC, not libx264.
    if [[ "$service_key" == "ffmpeg" && "$HAS_NVIDIA_GPU" == "true" ]]; then
        actual_profile="ffmpeg-gpu"
        compose_service="ffmpeg-gpu"
        requires_gpu="true"
    fi

    local arch_type
    arch_type=$(determine_architecture_type "$service_key" "$requires_gpu")
    if [[ "$service_key" == "ollama" && "$PLATFORM" == "macos" ]]; then
        if $use_native_ollama; then
            arch_type="NPU"
        else
            arch_type="CPU"
        fi
    fi

    log "[START] ${service_key} (${service_type}, ${arch_type})"
    detail "  Profile: ${actual_profile} | Compose service: ${compose_service}"

    # Record start time
    local start_time
    start_time=$(date -u '+%Y-%m-%d %H:%M:%S')
    local start_epoch
    start_epoch=$(date +%s)

    # Start the service
    cd "$PROJECT_DIR"
    if [[ "$actual_profile" != "none" ]]; then
        docker compose -p ds-services -f docker-compose.services.yml --profile "$actual_profile" up -d --quiet-pull --remove-orphans >> "$LOG_FILE" 2>&1
    fi

    # Special handling for Ollama: send a fixed number of inference requests
    if [[ "$service_key" == "ollama" ]]; then
        if [[ "$actual_profile" != "none" ]]; then
            detail "Starting Ollama container..."
            docker compose -p ds-services -f docker-compose.services.yml --profile "$actual_profile" up -d --quiet-pull --remove-orphans >> "$LOG_FILE" 2>&1
            detail "Waiting for Ollama to initialize..."
            sleep 10
            detail "Pulling model: ${OLLAMA_MODEL}"
            docker exec ds-ollama ollama pull "$OLLAMA_MODEL" >> "$LOG_FILE" 2>&1 || warn "Model pull failed"
        else
            detail "Waiting for native Ollama to be ready..."
            sleep 2
            detail "Pulling model natively via HTTP API: ${OLLAMA_MODEL}"
            local native_ollama_pull_url="http://localhost:11434/api/pull"
            if [[ -n "${HOST_OS:-}" && "${HOST_OS:-}" != "linux" ]]; then
                native_ollama_pull_url="http://host.docker.internal:11434/api/pull"
            fi
            curl -sf -X POST "$native_ollama_pull_url" -d "{\"name\": \"${OLLAMA_MODEL}\", \"stream\": false}" >> "$LOG_FILE" 2>&1 || warn "Native model pull failed"
        fi
        
        local start_epoch
        start_epoch=$(date +%s)
        detail "Sending ${OLLAMA_REQUESTS} inference requests..."
        # Prepare per-request timing CSV
        local req_csv="${RUN_OUTPUT_DIR}/ollama_requests.csv"
        echo "request_id,start_time,end_time,elapsed_seconds" > "$req_csv"
        for _ollama_i in $(seq 1 "${OLLAMA_REQUESTS}"); do
            # If running in Docker on Mac/Windows, we need to use host.docker.internal instead of localhost
            local actual_ollama_url="http://localhost:11434/api/generate"
            if [[ -n "${HOST_OS:-}" && "${HOST_OS:-}" != "linux" ]]; then
                actual_ollama_url="http://host.docker.internal:11434/api/generate"
            fi
            detail "Ollama request ${_ollama_i}/${OLLAMA_REQUESTS}..."
            local req_start req_end req_elapsed
            req_start=$(date +%s.%3N)
            curl -sf "$actual_ollama_url" -d "{
                \"model\": \"${OLLAMA_MODEL}\",
                \"prompt\": \"Explain quantum computing in detail, covering qubits, superposition, entanglement, and quantum gates. Provide examples of real-world applications.\",
                \"stream\": false
            }" > /dev/null 2>&1 || true
            req_end=$(date +%s.%3N)
            req_elapsed=$(awk "BEGIN {print $req_end - $req_start}")
            echo "${_ollama_i},$req_start,$req_end,$req_elapsed" >> "$req_csv"
        done
        # Capture container metadata before removing
        local container_id=""
        local service_size="0"
        local service_name="ollama"
        if [[ "$actual_profile" != "none" ]]; then
            container_id=$(docker inspect --format='{{.Id}}' "$container_name" 2>/dev/null || echo "")
            service_size=$(get_service_size "$container_name")
            service_name=$(docker inspect --format='{{.Config.Image}}' "$container_name" 2>/dev/null | tr -d '\n')
            [[ -z "$service_name" ]] && service_name="$service_key"
            docker compose -p ds-services -f docker-compose.services.yml --profile "$actual_profile" down --remove-orphans >> "$LOG_FILE" 2>&1
        else
            service_name="ollama-native"
        fi
        local exit_code=0
        local end_epoch
        end_epoch=$(date +%s)
    elif [[ "$service_key" == "busqueda-caminos" ]]; then
        if [[ -z "$BUSQUEDA_INSTANCE_FILE" || ! -f "$BUSQUEDA_INSTANCE_FILE" ]]; then
            warn "BUSQUEDA_INSTANCE_FILE not set or not found; skipping busqueda-caminos load"
            return 0
        fi
        
        # If running in Docker on Mac/Windows, we need to use host.docker.internal instead of localhost
        local actual_busqueda_url="$BUSQUEDA_URL"
        if [[ "$actual_busqueda_url" == *"localhost"* && -n "${HOST_OS:-}" && "${HOST_OS:-}" != "linux" ]]; then
            actual_busqueda_url="${actual_busqueda_url/localhost/host.docker.internal}"
        fi

        detail "Starting busqueda-caminos container..."
        docker compose -p ds-services -f docker-compose.services.yml --profile "$actual_profile" up -d --quiet-pull --remove-orphans >> "$LOG_FILE" 2>&1
        detail "Waiting up to ${WAIT_AFTER_START}s for busqueda-caminos to be ready..."
        local bc_retries="$WAIT_AFTER_START"
        while [ "$bc_retries" -gt 0 ]; do
            if curl -sf -X POST -F "file=@${BUSQUEDA_INSTANCE_FILE}" "$actual_busqueda_url" &>/dev/null; then
                detail "busqueda-caminos is ready"
                break
            fi
            bc_retries=$((bc_retries - 1))
            sleep 1
        done
        if [ "$bc_retries" -le 0 ]; then
            warn "busqueda-caminos not responding on $actual_busqueda_url; continuing"
        fi
        
        local start_epoch
        start_epoch=$(date +%s)
        detail "Sending ${BUSQUEDA_REQUESTS} requests..."
        for _bc_i in $(seq 1 "${BUSQUEDA_REQUESTS}"); do
            detail "busqueda-caminos request ${_bc_i}/${BUSQUEDA_REQUESTS}..."
            curl -sf -X POST -F "file=@${BUSQUEDA_INSTANCE_FILE}" "$actual_busqueda_url" > /dev/null 2>&1 || {
                warn "busqueda-caminos request failed"
                sleep 1
            }
        done
        # Capture container metadata before removing
        local container_id
        container_id=$(docker inspect --format='{{.Id}}' "$container_name" 2>/dev/null || echo "")
        local service_size
        service_size=$(get_service_size "$container_name")
        local service_name
        service_name=$(docker inspect --format='{{.Config.Image}}' "$container_name" 2>/dev/null | tr -d '\n')
        [[ -z "$service_name" ]] && service_name="$service_key"
        docker compose -p ds-services -f docker-compose.services.yml --profile "$actual_profile" down --remove-orphans >> "$LOG_FILE" 2>&1
        local exit_code=0
        local end_epoch
        end_epoch=$(date +%s)
    else
        # Start the service and wait for it to finish naturally.
        # --exit-code-from waits for the container to finish; timeout is a safety net.
        local start_epoch
        start_epoch=$(date +%s)
        timeout "${MAX_SERVICE_TIMEOUT}" \
            docker compose -p ds-services -f docker-compose.services.yml --profile "$actual_profile" up --quiet-pull --remove-orphans --exit-code-from "$compose_service" >> "$LOG_FILE" 2>&1
        local exit_code=$?
        local end_epoch
        end_epoch=$(date +%s)
        if [[ "$exit_code" -eq 124 ]]; then
            warn "${service_key} hit safety timeout after ${MAX_SERVICE_TIMEOUT}s — stopping container"
            docker compose -p ds-services -f docker-compose.services.yml --profile "$actual_profile" stop >> "$LOG_FILE" 2>&1 || true
        fi
        
        # Capture container metadata before removing
        local container_id
        container_id=$(docker inspect --format='{{.Id}}' "$container_name" 2>/dev/null || echo "")
        local service_size
        service_size=$(get_service_size "$container_name")
        local service_name
        service_name=$(docker inspect --format='{{.Config.Image}}' "$container_name" 2>/dev/null | tr -d '\n')
        [[ -z "$service_name" ]] && service_name="$service_key"

        # Ensure the container is fully removed so it doesn't interfere with subsequent runs
        docker compose -p ds-services -f docker-compose.services.yml --profile "$actual_profile" down --remove-orphans >> "$LOG_FILE" 2>&1
    fi

    local execution_time
    execution_time=$((end_epoch - start_epoch))
    
    if [ $exit_code -ne 0 ]; then
        warn "[DONE]  ${service_key} - ${execution_time}s (exit code $exit_code)"
    else
        log "[DONE]  ${service_key} - ${execution_time}s"
    fi

    # Determine workload numeric value and unit
    local workload_value="0.0"
    local workload_unit="unknown"
    
    # helper to strip non-numeric for parsing like '100G' if needed
    case "$service_key" in
        "sysbench") workload_value="${SYSBENCH_EVENTS}"; workload_unit="events" ;;
        "stress-ng") workload_value="${STRESSNG_OPS}"; workload_unit="ops" ;;
        "ffmpeg") workload_value="${FFMPEG_VIDEO_LENGTH}"; workload_unit="seconds" ;;
        "k6") workload_value="${K6_ITERATIONS}"; workload_unit="iterations" ;;
        "ollama") workload_value="${OLLAMA_REQUESTS}"; workload_unit="requests" ;;
        "ds-fio") workload_value="$(echo ${FIO_SIZE} | sed 's/[^0-9.]*//g')"; workload_unit="GB_transfer" ;;
        "ds-7zip") workload_value="${ZIP_ROUNDS}"; workload_unit="rounds" ;;
        "ds-build") workload_value="${BUILD_ROUNDS}"; workload_unit="rounds" ;;
        "ds-redis-benchmark") workload_value="${REDIS_REQUESTS}"; workload_unit="requests" ;;
        "iperf3") workload_value="$(echo ${IPERF_BYTES} | sed 's/[^0-9.]*//g')"; workload_unit="GB_transfer" ;;
        "busqueda-caminos") workload_value="${BUSQUEDA_REQUESTS}"; workload_unit="requests" ;;
        "network-tools") workload_value="${NETWORK_PING_COUNT}"; workload_unit="tests" ;;
        "pytorch-gpu") workload_value="${PYTORCH_GPU_EPOCHS}"; workload_unit="epochs" ;;
    esac

    # Wait for final metrics to be scraped by Prometheus (2x scrape interval)
    sleep 12

    # Get additional service metadata
    local num_params
    num_params=$(get_num_params "$service_key")

    # Export metrics to CSV (verbose mode for debugging)
    detail "Exporting metrics for ${service_key} (container_id=${container_id:-EMPTY})..."
    detail "  epochs: start=${start_epoch} end=${end_epoch} step=${SAMPLE_INTERVAL}"
    python3 "${SCRIPT_DIR}/export_csv.py" \
        --prometheus-url "$PROMETHEUS_URL" \
        --start-time "$start_epoch" \
        --end-time "$end_epoch" \
        --service-key "$service_key" \
        --service-name "$service_name" \
        --service-type "$service_type" \
        --architecture-type "$arch_type" \
        --hostname "$HOSTNAME_LABEL" \
        --cpu-name "$CPU_NAME" \
        --gpu-name "$GPU_NAME" \
        --npu-name "$NPU_NAME" \
        --service-size "$service_size" \
        --num-params "$num_params" \
        --execution-time "$execution_time" \
        --container-name "$container_name" \
        --container-id "$container_id" \
        --workload-value "$workload_value" \
        --workload-unit "$workload_unit" \
        --output-dir "$RUN_OUTPUT_DIR" \
        --has-nvidia-gpu "$HAS_NVIDIA_GPU" \
        --has-apple-npu "$HAS_APPLE_NPU" \
        --step "$SAMPLE_INTERVAL" \
        --verbose >> "$LOG_FILE" 2>&1

    detail "CSV export completed for ${service_key}"
}

# ============================================================
# Main
# ============================================================
main() {
    # Clear previous log
    > "$LOG_FILE"

    log "============================================================"
    log "  Dataset Creator - Metrics Collection Pipeline"
    log "============================================================"

    detect_platform

    # Install Python dependencies
    detail "Installing Python dependencies..."
    # Try a sequence of installs that covers:
    #   1. Plain pip (works in most envs and inside virtualenvs)
    #   2. --user (works on systems where global install needs sudo)
    #   3. --user --break-system-packages (PEP 668: modern Debian/Ubuntu/WSL2
    #      mark the system Python as externally-managed and refuse plain pip)
    # Use `python3 -m pip` so the install lands in the same interpreter that
    # will run export_csv.py.
    if ! python3 -c "import requests, pandas, yaml" >/dev/null 2>&1; then
        python3 -m pip install -q requests pandas pyyaml >> "$LOG_FILE" 2>&1 \
            || python3 -m pip install -q --user requests pandas pyyaml >> "$LOG_FILE" 2>&1 \
            || python3 -m pip install -q --user --break-system-packages requests pandas pyyaml >> "$LOG_FILE" 2>&1 \
            || warn "pip install failed, ensure requests, pandas and pyyaml are available"
    fi
    # Verify so a silent install failure doesn't show up later as missing
    # telemetry CSVs.
    python3 -c "import requests, pandas, yaml" >/dev/null 2>&1 \
        || warn "Python deps still missing after install — export_csv.py will fail and per-service CSVs will not be written."

    # Capture hardware profile + per-run metadata so the ML dataset
    # builder can join hardware features and the workload scale label
    # onto every (run, benchmark) row without ambiguity.
    detail "Capturing hardware profile..."
    python3 "${SCRIPT_DIR}/hw_profile.py" --output-dir "$RUN_OUTPUT_DIR" >> "$LOG_FILE" 2>&1 \
        || warn "hw_profile.py failed; hardware features will be missing from this run"

    # Measured machine-capability calibration (~15s) merged into hw_profile.json,
    # so build_ml_dataset picks up the calib_* features. Skippable via
    # SKIP_CALIBRATION=1 (e.g. for quick test runs).
    if [ "${SKIP_CALIBRATION:-0}" != "1" ]; then
        detail "Running machine calibration micro-benchmark (in-container)..."
        # Run inside a docker container to ensure the micro-benchmark measures the CPU in the same virtualized runtime as the target benchmarks.
        docker run --rm \
            -v "${PROJECT_DIR}:/workspace" \
            -w /workspace \
            python:3.10-slim \
            bash -c "pip install -q numpy==1.26.4 && python3 scripts/calibrate.py --merge-hw output/runs/${RUN_ID}" >> "$LOG_FILE" 2>&1 \
            || warn "calibrate.py in-container failed; calib_* features will be missing from this run"
    fi

    detail "Writing run_metadata.json (scale=${WORKLOAD_SCALE})..."
    python3 - <<PYEOF >> "$LOG_FILE" 2>&1 || warn "Failed to write run_metadata.json"
import json, os
meta = {
    "run_id": "${RUN_ID}",
    "workload_scale": "${WORKLOAD_SCALE}",
    "hostname_label": "${HOSTNAME_LABEL}",
    "sample_interval_s": ${SAMPLE_INTERVAL},
    "workload_values": {
        "ollama_requests": ${OLLAMA_REQUESTS},
        "busqueda_requests": ${BUSQUEDA_REQUESTS},
        "iperf_bytes": "${IPERF_BYTES}",
        "sysbench_events": ${SYSBENCH_EVENTS},
        "network_ping_count": ${NETWORK_PING_COUNT},
        "pytorch_gpu_epochs": ${PYTORCH_GPU_EPOCHS},
        "stressng_ops": ${STRESSNG_OPS},
        "ffmpeg_video_length": ${FFMPEG_VIDEO_LENGTH},
        "k6_iterations": ${K6_ITERATIONS},
        "fio_size": "${FIO_SIZE}",
        "zip_rounds": ${ZIP_ROUNDS},
        "build_rounds": ${BUILD_ROUNDS},
        "redis_requests": ${REDIS_REQUESTS},
    },
}
with open(os.path.join("${RUN_OUTPUT_DIR}", "run_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
PYEOF

    # Start monitoring
    start_monitoring

    # Run each service sequentially.
    # SKIP_SERVICES (space/comma-separated keys) lets a host opt out of services
    # it can't run meaningfully — e.g. images with no arm64 variant on an ARM
    # host, where emulation would distort execution_time. Skipped services
    # simply produce no row for this host (the model handles missing combos).
    local _skip="${SKIP_SERVICES:-}"
    local skip_norm=" ${_skip//,/ } "
    for service_def in "${SERVICES[@]}"; do
        IFS='|' read -r key profile stype req_gpu is_gpu_var cname csvc <<< "$service_def"
        if [[ "$skip_norm" == *" $key "* ]]; then
            warn "Skipping ${key} (in SKIP_SERVICES)"
            continue
        fi
        run_service "$key" "$profile" "$stype" "$req_gpu" "$is_gpu_var" "$cname" "$csvc" || {
            warn "Service ${key} failed, continuing..."
        }
    done

    # Merge per-service CSVs from this run into a run-level combined file
    detail "Merging service CSVs for run ${RUN_ID}..."
    python3 "${SCRIPT_DIR}/merge_service_csvs.py" --run-dir "$RUN_OUTPUT_DIR" --output-dir "$OUTPUT_DIR" >> "$LOG_FILE" 2>&1

    # Stop monitoring
    stop_monitoring

    log "============================================================"
    log "  Collection complete!"
    log "  Run data:  ${RUN_OUTPUT_DIR}/"
    log "  Master:    ${OUTPUT_DIR}/all_services.csv"
    log "  Full log:  ${LOG_FILE}"
    log "============================================================"
}

# Handle cleanup on exit
cleanup() {
    warn "Interrupted, cleaning up..."
    cd "$PROJECT_DIR"
    docker compose -p ds-services -f docker-compose.services.yml --profile ollama --profile ollama-cpu --profile busqueda-caminos --profile iperf3 --profile sysbench --profile network-tools --profile pytorch-gpu --profile stress-ng --profile ffmpeg --profile ffmpeg-gpu --profile k6 down >> "$LOG_FILE" 2>&1 || true
    stop_monitoring >> "$LOG_FILE" 2>&1 || true
    exit 1
}
trap cleanup SIGINT SIGTERM

main "$@"
