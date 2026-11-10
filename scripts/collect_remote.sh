#!/bin/bash
# ============================================================================
# Remote Dataset Collection Script
# ============================================================================
# Triggers orchestrator.sh on remote VMs via SSH
# Reads remote_hosts.txt for list of machines
#
# Usage:
#   ./collect_remote.sh                    # Run on all hosts
#   ./collect_remote.sh hostname1 hostname2  # Run on specific hosts
#   ./collect_remote.sh --parallel         # Run all hosts in parallel
# ============================================================================

set -e

# Configuration
HOSTS_FILE="$(dirname "$0")/remote_hosts.txt"
REPO_DIR="$HOME/dataset-creator"
LOG_DIR="./logs/remote-collection"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# ============================================================================
# Parse Arguments
# ============================================================================

PARALLEL=false
SELECTED_HOSTS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel|-p)
            PARALLEL=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] [HOSTNAME...]"
            echo ""
            echo "Trigger dataset collection on remote VMs via SSH"
            echo ""
            echo "Options:"
            echo "  -p, --parallel           Run all hosts in parallel (use with caution)"
            echo "  -h, --help               Show this help message"
            echo ""
            echo "Arguments:"
            echo "  HOSTNAME...              Specific hostnames to run (from remote_hosts.txt)"
            echo "                           If not specified, runs on all hosts"
            echo ""
            echo "Examples:"
            echo "  $0                       # Run on all hosts sequentially"
            echo "  $0 azure-cpu-01          # Run on single host"
            echo "  $0 azure-cpu-01 azure-gpu-01  # Run on specific hosts"
            echo "  $0 --parallel            # Run on all hosts in parallel"
            exit 0
            ;;
        *)
            SELECTED_HOSTS+=("$1")
            shift
            ;;
    esac
done

# ============================================================================
# Pre-flight Checks
# ============================================================================

print_header "Pre-flight Checks"

# Check if hosts file exists
if [ ! -f "$HOSTS_FILE" ]; then
    print_error "Hosts file not found: $HOSTS_FILE"
    echo ""
    echo "Please create $HOSTS_FILE with your remote host configurations"
    echo "See the example in the file for format"
    exit 1
fi

# Count hosts
TOTAL_HOSTS=$(grep -v '^#' "$HOSTS_FILE" | grep -v '^[[:space:]]*$' | wc -l)
if [ "$TOTAL_HOSTS" -eq 0 ]; then
    print_error "No hosts configured in $HOSTS_FILE"
    echo "Please uncomment and configure at least one host"
    exit 1
fi

print_success "Hosts file found: $HOSTS_FILE"
print_info "Total configured hosts: $TOTAL_HOSTS"

# Create log directory
mkdir -p "$LOG_DIR"

echo ""

# ============================================================================
# Execute Remote Collection
# ============================================================================

print_header "Remote Dataset Collection"
echo ""

# Function to execute on a single host
execute_on_host() {
    local hostname=$1
    local ssh_user=$2
    local ip_address=$3
    local description=$4

    local log_file="$LOG_DIR/${hostname}_$(date +%Y%m%d-%H%M%S).log"

    echo ""
    print_header "Host: $hostname"
    echo "  Description: $description"
    echo "  Connection:  ${ssh_user}@${ip_address}"
    echo "  Log file:    $log_file"
    echo ""

    # Test SSH connection
    print_info "Testing SSH connection..."
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "${ssh_user}@${ip_address}" 'echo "Connection successful"' > /dev/null 2>&1; then
        print_error "Cannot connect to ${ssh_user}@${ip_address}"
        echo "  Please check:"
        echo "    - SSH key is configured for passwordless login"
        echo "    - IP address is correct"
        echo "    - VM is running"
        return 1
    fi
    print_success "SSH connection successful"

    # Check if repository exists
    print_info "Checking repository..."
    if ! ssh "${ssh_user}@${ip_address}" "test -d ~/dataset-creator"; then
        print_error "Repository not found at ~/dataset-creator"
        echo "  Please run setup-vm.sh on this host first"
        return 1
    fi
    print_success "Repository found"

    # Start collection
    print_info "Starting dataset collection..."
    print_info "This may take 1-2 hours depending on configured services..."

    # Execute orchestrator remotely and log output
    if ssh "${ssh_user}@${ip_address}" \
        "cd ~/dataset-creator && make run-docker" \
        2>&1 | tee "$log_file"; then

        print_success "Collection completed on $hostname"
        echo "  Log file: $log_file"
        return 0
    else
        print_error "Collection failed on $hostname"
        echo "  Check log file: $log_file"
        return 1
    fi
}

# Read hosts and execute
SUCCESS_COUNT=0
FAIL_COUNT=0
PIDS=()

while IFS=',' read -r hostname ssh_user ip_address description; do
    # Skip comments and empty lines
    [[ "$hostname" =~ ^#.*$ ]] && continue
    [[ -z "$hostname" ]] && continue

    # Trim whitespace
    hostname=$(echo "$hostname" | xargs)
    ssh_user=$(echo "$ssh_user" | xargs)
    ip_address=$(echo "$ip_address" | xargs)
    description=$(echo "$description" | xargs)

    # Skip if specific hosts requested and this isn't one
    if [ ${#SELECTED_HOSTS[@]} -gt 0 ]; then
        if [[ ! " ${SELECTED_HOSTS[@]} " =~ " ${hostname} " ]]; then
            continue
        fi
    fi

    # Execute (parallel or sequential)
    if [ "$PARALLEL" = true ]; then
        execute_on_host "$hostname" "$ssh_user" "$ip_address" "$description" &
        PIDS+=($!)
    else
        if execute_on_host "$hostname" "$ssh_user" "$ip_address" "$description"; then
            ((SUCCESS_COUNT++))
        else
            ((FAIL_COUNT++))
        fi
    fi

done < "$HOSTS_FILE"

# Wait for parallel jobs
if [ "$PARALLEL" = true ]; then
    print_info "Waiting for parallel jobs to complete..."
    for pid in "${PIDS[@]}"; do
        if wait "$pid"; then
            ((SUCCESS_COUNT++))
        else
            ((FAIL_COUNT++))
        fi
    done
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
print_header "Collection Summary"
echo ""
echo "  Successful: $SUCCESS_COUNT"
echo "  Failed:     $FAIL_COUNT"
echo "  Logs:       $LOG_DIR"
echo ""

if [ "$FAIL_COUNT" -gt 0 ]; then
    print_error "Some collections failed. Check logs in $LOG_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Review log files for errors"
    echo "  2. Fix any issues on failed hosts"
    echo "  3. Re-run this script for failed hosts only"
    exit 1
else
    print_success "All collections completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Download datasets: ./scripts/download_datasets.sh"
    echo "  2. Merge datasets: python3 scripts/merge_datasets.py datasets/* -o final_dataset.csv"
fi

echo ""
