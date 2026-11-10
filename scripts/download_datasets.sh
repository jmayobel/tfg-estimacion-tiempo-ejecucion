#!/bin/bash
# ============================================================================
# Remote Dataset Download Script
# ============================================================================
# Downloads CSV files from remote VMs to local machine
# Uses rsync for efficient transfer with progress display
#
# Usage:
#   ./download_datasets.sh                    # Download from all hosts
#   ./download_datasets.sh hostname1 hostname2  # Download from specific hosts
#   ./download_datasets.sh --output-dir ./data  # Custom output directory
# ============================================================================

set -e

# Configuration
HOSTS_FILE="$(dirname "$0")/remote_hosts.txt"
OUTPUT_DIR="./datasets"
REMOTE_OUTPUT_DIR="~/dataset-creator/output"

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

SELECTED_HOSTS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] [HOSTNAME...]"
            echo ""
            echo "Download dataset CSV files from remote VMs to local machine"
            echo ""
            echo "Options:"
            echo "  -o, --output-dir DIR     Output directory (default: ./datasets)"
            echo "  -h, --help               Show this help message"
            echo ""
            echo "Arguments:"
            echo "  HOSTNAME...              Specific hostnames to download from"
            echo "                           If not specified, downloads from all hosts"
            echo ""
            echo "Examples:"
            echo "  $0                              # Download from all hosts"
            echo "  $0 azure-cpu-01                 # Download from single host"
            echo "  $0 -o ./my-data azure-cpu-01    # Custom output directory"
            echo ""
            echo "Output Structure:"
            echo "  ./datasets/"
            echo "    ├── azure-cpu-01/"
            echo "    │   └── all_services.csv"
            echo "    ├── azure-gpu-01/"
            echo "    │   └── all_services.csv"
            echo "    └── windows-rtx3060/"
            echo "        └── all_services.csv"
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
    exit 1
fi

# Check if rsync is available
if ! command -v rsync &> /dev/null; then
    print_error "rsync is not installed"
    echo "Please install rsync:"
    echo "  macOS: brew install rsync"
    echo "  Ubuntu: sudo apt-get install rsync"
    exit 1
fi

print_success "rsync is available"

# Create output directory
mkdir -p "$OUTPUT_DIR"
print_success "Output directory: $OUTPUT_DIR"

echo ""

# ============================================================================
# Download from Hosts
# ============================================================================

print_header "Downloading Datasets"
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL_SIZE=0

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

    echo ""
    print_header "Host: $hostname"
    echo "  Description: $description"
    echo "  Connection:  ${ssh_user}@${ip_address}"
    echo ""

    # Create host-specific directory
    HOST_OUTPUT_DIR="${OUTPUT_DIR}/${hostname}"
    mkdir -p "$HOST_OUTPUT_DIR"

    # Test SSH connection
    print_info "Testing SSH connection..."
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "${ssh_user}@${ip_address}" 'echo "OK"' > /dev/null 2>&1; then
        print_error "Cannot connect to ${ssh_user}@${ip_address}"
        ((FAIL_COUNT++))
        continue
    fi
    print_success "Connected"

    # Check if remote output directory exists
    print_info "Checking remote output directory..."
    if ! ssh "${ssh_user}@${ip_address}" "test -d ${REMOTE_OUTPUT_DIR}"; then
        print_error "Remote output directory not found: ${REMOTE_OUTPUT_DIR}"
        echo "  Run ./scripts/collect_remote.sh first to generate datasets"
        ((FAIL_COUNT++))
        continue
    fi

    # Check if CSV files exist
    CSV_COUNT=$(ssh "${ssh_user}@${ip_address}" "find ${REMOTE_OUTPUT_DIR} -name '*.csv' 2>/dev/null | wc -l")
    if [ "$CSV_COUNT" -eq 0 ]; then
        print_error "No CSV files found in ${REMOTE_OUTPUT_DIR}"
        echo "  Run ./scripts/collect_remote.sh first to generate datasets"
        ((FAIL_COUNT++))
        continue
    fi

    print_success "Found $CSV_COUNT CSV file(s)"

    # Download with rsync
    print_info "Downloading datasets..."
    echo ""

    if rsync -avz --progress \
        "${ssh_user}@${ip_address}:${REMOTE_OUTPUT_DIR}/" \
        "${HOST_OUTPUT_DIR}/"; then

        print_success "Downloaded to: $HOST_OUTPUT_DIR"

        # Calculate size
        SIZE=$(du -sh "$HOST_OUTPUT_DIR" | cut -f1)
        print_info "Size: $SIZE"

        # List downloaded files
        print_info "Files:"
        find "$HOST_OUTPUT_DIR" -name '*.csv' -exec basename {} \; | while read -r file; do
            echo "    - $file"
        done

        ((SUCCESS_COUNT++))
    else
        print_error "Download failed for $hostname"
        ((FAIL_COUNT++))
    fi

done < "$HOSTS_FILE"

# ============================================================================
# Summary
# ============================================================================

echo ""
print_header "Download Summary"
echo ""
echo "  Successful:     $SUCCESS_COUNT"
echo "  Failed:         $FAIL_COUNT"
echo "  Output dir:     $OUTPUT_DIR"
echo ""

if [ "$SUCCESS_COUNT" -gt 0 ]; then
    echo "Downloaded datasets:"
    find "$OUTPUT_DIR" -name "all_services.csv" | while read -r file; do
        SIZE=$(du -h "$file" | cut -f1)
        ROWS=$(tail -n +2 "$file" 2>/dev/null | wc -l)
        echo "  - $file ($SIZE, $ROWS rows)"
    done
    echo ""
fi

if [ "$FAIL_COUNT" -gt 0 ]; then
    print_error "Some downloads failed"
    echo ""
    echo "Next steps:"
    echo "  1. Check SSH connectivity to failed hosts"
    echo "  2. Verify datasets exist on remote hosts"
    echo "  3. Re-run this script for failed hosts"
    exit 1
else
    print_success "All downloads completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Merge datasets:"
    echo "     python3 scripts/merge_datasets.py ${OUTPUT_DIR}/* -o final_dataset.csv"
    echo ""
    echo "  2. View merged dataset info:"
    echo "     head final_dataset.csv"
    echo "     wc -l final_dataset.csv"
fi

echo ""
