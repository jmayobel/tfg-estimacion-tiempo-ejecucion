#!/bin/bash
# ============================================================================
# Dataset Creator - Azure VM Setup Script
# ============================================================================
# Sets up a freshly deployed Azure Ubuntu 22.04 VM for dataset collection
# - Installs Docker Engine and Docker Compose
# - Installs Python 3 and dependencies
# - Clones the dataset-creator repository
# - Configures environment
# - (GPU VMs only) Installs NVIDIA drivers and DCGM Exporter
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Configuration
# ============================================================================

# REPO_URL can be passed as $1, via REPO_URL env var, or omitted entirely
# (in which case the user is expected to upload the project tree via scp/rsync).
REPO_URL="${1:-${REPO_URL:-https://github.com/Se-py/TFG-Benchmark.git}}"
REPO_DIR="${REPO_DIR:-$HOME/dataset-creator}"
HOSTNAME_LABEL="azure-vm-$(hostname)"

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
# Detect GPU
# ============================================================================

detect_gpu() {
    if lspci | grep -i nvidia > /dev/null; then
        return 0  # GPU detected
    else
        return 1  # No GPU
    fi
}

# ============================================================================
# System Update
# ============================================================================

print_header "Step 1: System Update"

print_info "Updating package lists..."
sudo apt-get update -qq

print_info "Upgrading existing packages..."
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y -qq

print_success "System updated"
echo ""

# ============================================================================
# Install Docker Engine
# ============================================================================

print_header "Step 2: Install Docker Engine"

if command -v docker &> /dev/null; then
    print_info "Docker already installed: $(docker --version)"
else
    print_info "Installing Docker Engine..."

    # Install prerequisites
    sudo apt-get install -y -qq \
        ca-certificates \
        curl \
        gnupg \
        lsb-release

    # Add Docker's official GPG key
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    # Set up Docker repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Install Docker Engine
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        docker-ce \
        docker-ce-cli \
        containerd.io \
        docker-buildx-plugin \
        docker-compose-plugin

    # Add current user to docker group
    sudo usermod -aG docker $USER

    print_success "Docker Engine installed: $(docker --version)"
fi

# Verify Docker Compose
if docker compose version &> /dev/null; then
    print_success "Docker Compose available: $(docker compose version)"
else
    print_error "Docker Compose plugin not installed"
    exit 1
fi

echo ""

# ============================================================================
# Install Python 3 and Dependencies
# ============================================================================

print_header "Step 3: Install Python and Dependencies"

print_info "Installing Python 3 and pip..."
sudo apt-get install -y -qq \
    python3 \
    python3-pip \
    python3-venv \
    git

PYTHON_VERSION=$(python3 --version)
print_success "Python installed: $PYTHON_VERSION"

print_info "Upgrading pip..."
python3 -m pip install --upgrade pip --quiet

print_success "Python environment ready"
echo ""

# ============================================================================
# Install NVIDIA Drivers (GPU VMs only)
# ============================================================================

if detect_gpu; then
    print_header "Step 4: Install NVIDIA Drivers and Tools (GPU Detected)"

    print_info "NVIDIA GPU detected"
    lspci | grep -i nvidia

    echo ""
    print_info "Installing NVIDIA drivers..."

    # Add NVIDIA repository
    sudo add-apt-repository ppa:graphics-drivers/ppa -y
    sudo apt-get update -qq

    # Install NVIDIA driver
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        nvidia-driver-535 \
        nvidia-utils-535

    print_success "NVIDIA driver installed"

    # Install NVIDIA Container Toolkit
    print_info "Installing NVIDIA Container Toolkit..."

    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-container-toolkit

    # Configure Docker to use NVIDIA runtime
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker

    print_success "NVIDIA Container Toolkit installed"

    echo ""
    print_info "${YELLOW}⚠️  You may need to reboot for NVIDIA drivers to fully activate${NC}"
    print_info "${YELLOW}   Run: sudo reboot${NC}"
    echo ""
else
    print_header "Step 4: NVIDIA Drivers (Skipped - No GPU Detected)"
    print_info "This is a CPU-only VM, skipping GPU setup"
    echo ""
fi

# ============================================================================
# Clone Repository
# ============================================================================

print_header "Step 5: Clone Dataset Creator Repository"

if [ -d "$REPO_DIR/.git" ]; then
    print_info "Repository already cloned at: $REPO_DIR"
    if [ -n "$REPO_URL" ]; then
        print_info "Pulling latest changes..."
        cd "$REPO_DIR"
        git pull --ff-only || print_info "Could not pull updates (continuing)"
    fi
elif [ -d "$REPO_DIR" ]; then
    print_info "Found non-git project tree at: $REPO_DIR (assuming uploaded manually)"
else
    if [ -z "$REPO_URL" ]; then
        print_error "REPO_URL is not set and $REPO_DIR does not exist."
        echo ""
        echo "Either:"
        echo "  1) Re-run with the repo URL:    bash setup-vm.sh https://github.com/<you>/dataset-creator.git"
        echo "  2) Set the env var:             REPO_URL=... bash setup-vm.sh"
        echo "  3) Upload the project to $REPO_DIR (scp/rsync) before running this script."
        echo ""
        exit 1
    fi

    print_info "Cloning repository..."
    print_info "URL: $REPO_URL"
    git clone "$REPO_URL" "$REPO_DIR"
    print_success "Repository cloned to: $REPO_DIR"
fi

cd "$REPO_DIR"
echo ""

# ============================================================================
# Install Python Dependencies
# ============================================================================

print_header "Step 6: Install Python Dependencies"

if [ -f "scripts/requirements.txt" ]; then
    print_info "Installing production dependencies..."
    pip3 install -r scripts/requirements.txt --quiet
    print_success "Production dependencies installed"
fi

if [ -f "requirements-dev.txt" ]; then
    print_info "Installing development dependencies..."
    pip3 install -r requirements-dev.txt --quiet
    print_success "Development dependencies installed"
fi

echo ""

# ============================================================================
# Configure Environment
# ============================================================================

print_header "Step 7: Configure Environment"

if [ ! -f ".env" ]; then
    print_info "Creating .env file from template..."

    if [ -f ".env.example" ]; then
        cp .env.example .env
    else
        # Create minimal .env if example doesn't exist
        cat > .env << EOF
# Dataset Creator Configuration
SAMPLE_INTERVAL=2
MAX_SERVICE_TIMEOUT=3600
HOSTNAME_LABEL=$HOSTNAME_LABEL
PROMETHEUS_URL=http://localhost:9090
WAIT_AFTER_START=10
EOF
    fi

    # Update HOSTNAME_LABEL
    sed -i "s/^HOSTNAME_LABEL=.*/HOSTNAME_LABEL=$HOSTNAME_LABEL/" .env

    print_success ".env file created"
else
    print_info ".env file already exists"
fi

print_info "Configured hostname: $HOSTNAME_LABEL"
echo ""

# ============================================================================
# Test Docker
# ============================================================================

print_header "Step 8: Test Docker Installation"

print_info "Testing Docker..."

# For docker commands to work without sudo in this session, we need newgrp or re-login
# For now, use sudo for verification
if sudo docker run --rm hello-world > /dev/null 2>&1; then
    print_success "Docker is working correctly"
else
    print_error "Docker test failed"
    echo "You may need to log out and log back in for docker group changes to take effect"
fi

echo ""

# ============================================================================
# Summary
# ============================================================================

print_header "Setup Complete!"

echo ""
echo "Summary:"
echo "  ✓ System updated"
echo "  ✓ Docker Engine installed: $(docker --version 2>/dev/null || echo 'N/A')"
echo "  ✓ Docker Compose installed: $(docker compose version 2>/dev/null || echo 'N/A')"
echo "  ✓ Python installed: $(python3 --version)"
echo "  ✓ Repository cloned: $REPO_DIR"
echo "  ✓ Environment configured: $HOSTNAME_LABEL"
if detect_gpu; then
    echo "  ✓ NVIDIA drivers installed (reboot recommended)"
fi
echo ""

print_info "Next steps:"
echo ""
echo "1. ${GREEN}Log out and log back in${NC} (for docker group changes to take effect):"
echo "   exit"
echo "   ssh $USER@$(hostname -I | awk '{print $1}')"
echo ""
echo "2. ${GREEN}Test Docker without sudo:${NC}"
echo "   docker run --rm hello-world"
echo ""

if detect_gpu; then
    echo "3. ${GREEN}Reboot to activate NVIDIA drivers:${NC}"
    echo "   sudo reboot"
    echo ""
    echo "4. ${GREEN}After reboot, verify GPU:${NC}"
    echo "   nvidia-smi"
    echo ""
    echo "5. ${GREEN}Start GPU monitoring stack:${NC}"
    echo "   cd ~/dataset-creator"
    echo "   docker compose -f docker-compose.monitoring.yml -f docker-compose.gpu.yml up -d"
    echo ""
fi

echo "6. ${GREEN}Run the dataset creator:${NC}"
echo "   cd ~/dataset-creator"
echo "   ./scripts/orchestrator.sh"
echo ""
echo "7. ${GREEN}Collect results:${NC}"
echo "   Results will be in: ~/dataset-creator/output/all_services.csv"
echo ""

print_success "Setup script completed successfully!"
echo ""
