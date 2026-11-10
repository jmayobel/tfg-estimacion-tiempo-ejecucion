#!/bin/bash
# ============================================================================
# Azure VM Deployment Script for Dataset Creator
# ============================================================================
# Deploys Azure VM using Bicep template with configurable parameters
# Usage: ./deploy.sh [--resource-group RG_NAME] [--location LOCATION]
# ============================================================================

set -e  # Exit on error

# Default values
RESOURCE_GROUP="dataset-creator-rg"
LOCATION="westeurope"
TEMPLATE_FILE="vm-template.bicep"
PARAMETERS_FILE="parameters.json"
DEPLOYMENT_NAME="dataset-creator-deployment-$(date +%Y%m%d-%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

while [[ $# -gt 0 ]]; do
    case $1 in
        --resource-group|-g)
            RESOURCE_GROUP="$2"
            shift 2
            ;;
        --location|-l)
            LOCATION="$2"
            shift 2
            ;;
        --parameters|-p)
            PARAMETERS_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -g, --resource-group NAME    Resource group name (default: dataset-creator-rg)"
            echo "  -l, --location LOCATION      Azure region (default: westeurope)"
            echo "  -p, --parameters FILE        Parameters file (default: parameters.json)"
            echo "  -h, --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                        # Use defaults"
            echo "  $0 -g my-rg -l westus2                   # Custom resource group and location"
            echo "  $0 -p parameters-gpu.json                # Use GPU parameters file"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Pre-flight Checks
# ============================================================================

print_header "Pre-flight Checks"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    print_error "Azure CLI is not installed"
    echo "Please install from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi
print_success "Azure CLI is installed"

# Check if logged in to Azure
if ! az account show &> /dev/null; then
    print_error "Not logged in to Azure"
    echo "Please run: az login"
    exit 1
fi
print_success "Logged in to Azure"

# Check if template file exists
if [ ! -f "$TEMPLATE_FILE" ]; then
    print_error "Template file not found: $TEMPLATE_FILE"
    exit 1
fi
print_success "Template file found: $TEMPLATE_FILE"

# Check if parameters file exists
if [ ! -f "$PARAMETERS_FILE" ]; then
    print_error "Parameters file not found: $PARAMETERS_FILE"
    echo "Please create $PARAMETERS_FILE or use --parameters to specify a different file"
    exit 1
fi
print_success "Parameters file found: $PARAMETERS_FILE"

# Get current subscription
SUBSCRIPTION=$(az account show --query name -o tsv)
print_info "Using subscription: $SUBSCRIPTION"

echo ""

# ============================================================================
# Create Resource Group
# ============================================================================

print_header "Creating Resource Group"

if az group show --name "$RESOURCE_GROUP" &> /dev/null; then
    print_info "Resource group '$RESOURCE_GROUP' already exists"
else
    print_info "Creating resource group: $RESOURCE_GROUP in $LOCATION"
    az group create \
        --name "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --output none
    print_success "Resource group created"
fi

echo ""

# ============================================================================
# Deploy Bicep Template
# ============================================================================

print_header "Deploying VM"

print_info "Deployment name: $DEPLOYMENT_NAME"
print_info "Resource group: $RESOURCE_GROUP"
print_info "Location: $LOCATION"
print_info "Template: $TEMPLATE_FILE"
print_info "Parameters: $PARAMETERS_FILE"

echo ""
print_info "Starting deployment (this may take 5-10 minutes)..."
echo ""

# Deploy with output captured
DEPLOYMENT_OUTPUT=$(az deployment group create \
    --name "$DEPLOYMENT_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --template-file "$TEMPLATE_FILE" \
    --parameters "@$PARAMETERS_FILE" \
    --output json)

if [ $? -eq 0 ]; then
    print_success "Deployment completed successfully"
else
    print_error "Deployment failed"
    exit 1
fi

echo ""

# ============================================================================
# Extract Outputs
# ============================================================================

print_header "Deployment Information"

# Parse outputs
PUBLIC_IP=$(echo "$DEPLOYMENT_OUTPUT" | jq -r '.properties.outputs.publicIP.value')
FQDN=$(echo "$DEPLOYMENT_OUTPUT" | jq -r '.properties.outputs.fqdn.value')
SSH_COMMAND=$(echo "$DEPLOYMENT_OUTPUT" | jq -r '.properties.outputs.sshCommand.value')
VM_SIZE=$(echo "$DEPLOYMENT_OUTPUT" | jq -r '.properties.outputs.vmSize.value')
IS_GPU_VM=$(echo "$DEPLOYMENT_OUTPUT" | jq -r '.properties.outputs.isGpuVM.value')

echo ""
echo "VM Details:"
echo "  VM Size:       $VM_SIZE"
echo "  GPU Enabled:   $IS_GPU_VM"
echo "  Public IP:     $PUBLIC_IP"
echo "  FQDN:          $FQDN"
echo ""
echo "SSH Access:"
echo "  Command:       $SSH_COMMAND"
echo ""

# Save connection info to file
CONNECTION_FILE="connection-info.txt"
cat > "$CONNECTION_FILE" << EOF
# Dataset Creator VM Connection Information
# Generated: $(date)

Resource Group: $RESOURCE_GROUP
Location:       $LOCATION
VM Size:        $VM_SIZE
GPU Enabled:    $IS_GPU_VM
Public IP:      $PUBLIC_IP
FQDN:           $FQDN

SSH Command:
  $SSH_COMMAND

Next Steps:
  1. Wait ~2 minutes for VM to fully boot
  2. Test SSH connection: $SSH_COMMAND
  3. Run setup script:
     scp setup-vm.sh ${SSH_COMMAND#ssh }:~/
     ssh ${SSH_COMMAND#ssh } 'bash ~/setup-vm.sh'
  4. Verify installation:
     ssh ${SSH_COMMAND#ssh } 'docker --version && python3 --version'

EOF

print_success "Connection info saved to: $CONNECTION_FILE"

# ============================================================================
# Post-Deployment Instructions
# ============================================================================

print_header "Next Steps"

echo ""
echo "1. ${GREEN}Wait for VM to boot${NC} (~2 minutes)"
echo ""
echo "2. ${GREEN}Test SSH connection:${NC}"
echo "   $SSH_COMMAND"
echo ""
echo "3. ${GREEN}Setup the VM:${NC}"
echo "   cd azure"
echo "   ./setup-vm.sh $PUBLIC_IP"
echo ""
echo "4. ${GREEN}Alternative manual setup:${NC}"
echo "   scp setup-vm.sh ${SSH_COMMAND#ssh }:~/"
echo "   ssh ${SSH_COMMAND#ssh } 'bash ~/setup-vm.sh'"
echo ""
echo "5. ${GREEN}Verify installation:${NC}"
echo "   ssh ${SSH_COMMAND#ssh } 'docker --version && python3 --version'"
echo ""

if [ "$IS_GPU_VM" = "true" ]; then
    echo "${YELLOW}Note: This is a GPU VM. The setup script will install NVIDIA drivers.${NC}"
    echo "${YELLOW}      After setup, verify GPU with: ssh ${SSH_COMMAND#ssh } 'nvidia-smi'${NC}"
    echo ""
fi

print_success "Deployment complete!"
echo ""
print_info "Connection info saved to: $CONNECTION_FILE"
print_info "Keep this file for future reference"
echo ""

# ============================================================================
# Cost Warning
# ============================================================================

echo -e "${YELLOW}⚠️  Remember to stop or deallocate the VM when not in use to minimize costs:${NC}"
echo "   az vm deallocate --resource-group $RESOURCE_GROUP --name <vm-name>"
echo "   az vm start --resource-group $RESOURCE_GROUP --name <vm-name>"
echo ""
