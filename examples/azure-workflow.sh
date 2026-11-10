#!/bin/bash
# ============================================================================
# Example End-to-End Azure Workflow
# ============================================================================
# This script demonstrates the complete workflow for:
# 1. Deploying 2 Azure VMs (1 CPU, 1 GPU)
# 2. Setting up both VMs
# 3. Configuring remote hosts
# 4. Running dataset collection on all machines
# 5. Downloading datasets
# 6. Merging with local data
# 7. Generating final dataset
#
# THIS IS AN EXAMPLE - Review and customize before running!
# ============================================================================

set -e  # Exit on error

# ============================================================================
# Configuration (CUSTOMIZE THESE)
# ============================================================================

# Azure configuration
RESOURCE_GROUP="dataset-creator-rg"
LOCATION="westeurope"

# SSH public key path
SSH_PUBLIC_KEY=$(cat ~/.ssh/id_rsa.pub)

# Local dataset directory
LOCAL_OUTPUT_DIR="./output_local"  # Your local dataset directory

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ============================================================================
# Step-by-Step Workflow
# ============================================================================

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}Dataset Creator - Azure Workflow Example${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo "This script will:"
echo "  1. Deploy 2 Azure VMs (CPU and GPU)"
echo "  2. Setup both VMs"
echo "  3. Configure remote hosts"
echo "  4. Trigger dataset collection"
echo "  5. Download results"
echo "  6. Merge with local data"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# ============================================================================
# Step 1: Deploy Azure VMs
# ============================================================================

echo -e "\n${GREEN}Step 1: Deploying Azure VMs${NC}"
echo "-----------------------------------"

cd azure || exit 1

# Deploy CPU VM
echo "Deploying CPU VM (Standard_D4s_v3)..."
cat > parameters-cpu.json <<EOF
{
  "\$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "vmName": {"value": "dataset-cpu-01"},
    "vmSize": {"value": "Standard_D4s_v3"},
    "adminUsername": {"value": "azureuser"},
    "sshPublicKey": {"value": "$SSH_PUBLIC_KEY"},
    "osDiskType": {"value": "StandardSSD_LRS"},
    "osDiskSizeGB": {"value": 128},
    "enableDataDisk": {"value": false},
    "enableAcceleratedNetworking": {"value": true},
    "environment": {"value": "production"},
    "projectTag": {"value": "dataset-creator"}
  }
}
EOF

./deploy.sh -p parameters-cpu.json -g "$RESOURCE_GROUP" -l "$LOCATION" > deployment-cpu.log 2>&1 &
PID_CPU=$!

# Deploy GPU VM
# NOTE: Standard_NC4as_T4_v3 requires quota in the NCASv3_T4 family. Azure for
# Students subscriptions start at quota 0 for this family — request an increase
# (minimum 4 vCPUs) via Portal -> Subscriptions -> Usage + quotas before running.
echo "Deploying GPU VM (Standard_NC4as_T4_v3 - Tesla T4)..."
cat > parameters-gpu.json <<EOF
{
  "\$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "vmName": {"value": "dataset-gpu-01"},
    "vmSize": {"value": "Standard_NC4as_T4_v3"},
    "adminUsername": {"value": "azureuser"},
    "sshPublicKey": {"value": "$SSH_PUBLIC_KEY"},
    "osDiskType": {"value": "Premium_LRS"},
    "osDiskSizeGB": {"value": 256},
    "enableDataDisk": {"value": true},
    "dataDiskSizeGB": {"value": 512},
    "enableAcceleratedNetworking": {"value": true},
    "environment": {"value": "production"},
    "projectTag": {"value": "dataset-creator"}
  }
}
EOF

./deploy.sh -p parameters-gpu.json -g "$RESOURCE_GROUP" -l "$LOCATION" > deployment-gpu.log 2>&1 &
PID_GPU=$!

# Wait for both deployments
echo "Waiting for deployments to complete (this may take 5-10 minutes)..."
wait $PID_CPU
wait $PID_GPU

# Extract IPs from deployment outputs
CPU_VM_IP=$(grep "Public IP:" deployment-cpu.log | awk '{print $NF}')
GPU_VM_IP=$(grep "Public IP:" deployment-gpu.log | awk '{print $NF}')

echo -e "${GREEN}✓ VMs deployed${NC}"
echo "  CPU VM: azureuser@$CPU_VM_IP"
echo "  GPU VM: azureuser@$GPU_VM_IP"

cd ..

# ============================================================================
# Step 2: Setup VMs
# ============================================================================

echo -e "\n${GREEN}Step 2: Setting up VMs${NC}"
echo "-----------------------------------"

echo "Setting up CPU VM..."
ssh -o StrictHostKeyChecking=no azureuser@$CPU_VM_IP 'bash -s' < azure/setup-vm.sh > setup-cpu.log 2>&1

echo "Setting up GPU VM (includes NVIDIA drivers)..."
ssh -o StrictHostKeyChecking=no azureuser@$GPU_VM_IP 'bash -s' < azure/setup-vm.sh > setup-gpu.log 2>&1

echo -e "${GREEN}✓ VMs configured${NC}"
echo ""
echo -e "${YELLOW}⚠️  GPU VM requires reboot for NVIDIA drivers${NC}"
read -p "Reboot GPU VM now? (y/N) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ssh azureuser@$GPU_VM_IP 'sudo reboot'
    echo "Waiting 2 minutes for VM to reboot..."
    sleep 120
fi

# ============================================================================
# Step 3: Configure Remote Hosts
# ============================================================================

echo -e "\n${GREEN}Step 3: Configuring remote hosts${NC}"
echo "-----------------------------------"

cat > scripts/remote_hosts.txt <<EOF
# Generated by azure-workflow.sh on $(date)
# Azure VMs
azure-cpu-01,azureuser,$CPU_VM_IP,Azure Standard_D4s_v3 CPU VM
azure-gpu-01,azureuser,$GPU_VM_IP,Azure Standard_NC4as_T4_v3 (Tesla T4) GPU VM

# Local machines (UNCOMMENT AND CUSTOMIZE IF NEEDED)
#windows-rtx3060,user,192.168.1.100,Local Windows WSL2 with RTX 3060
#macos-m4-pro,user,192.168.1.101,Local macOS M4 Pro
EOF

echo -e "${GREEN}✓ Remote hosts configured${NC}"
cat scripts/remote_hosts.txt

# ============================================================================
# Step 4: Trigger Dataset Collection
# ============================================================================

echo -e "\n${GREEN}Step 4: Triggering dataset collection${NC}"
echo "-----------------------------------"

echo "Starting collection on all remote hosts..."
echo "This will take 1-2 hours depending on service durations."
echo ""

./scripts/collect_remote.sh

echo -e "${GREEN}✓ Collection completed on all hosts${NC}"

# ============================================================================
# Step 5: Download Datasets
# ============================================================================

echo -e "\n${GREEN}Step 5: Downloading datasets${NC}"
echo "-----------------------------------"

./scripts/download_datasets.sh

echo -e "${GREEN}✓ Datasets downloaded to ./datasets/${NC}"
ls -lh datasets/

# ============================================================================
# Step 6: Merge with Local Data
# ============================================================================

echo -e "\n${GREEN}Step 6: Merging datasets${NC}"
echo "-----------------------------------"

# Check if local output exists
if [ -d "$LOCAL_OUTPUT_DIR" ]; then
    echo "Merging Azure datasets with local data..."
    python3 scripts/merge_datasets.py \
        datasets/azure-cpu-01 \
        datasets/azure-gpu-01 \
        "$LOCAL_OUTPUT_DIR" \
        -o final_dataset.csv
else
    echo "No local dataset found, merging Azure datasets only..."
    python3 scripts/merge_datasets.py \
        datasets/azure-cpu-01 \
        datasets/azure-gpu-01 \
        -o final_dataset.csv
fi

echo -e "${GREEN}✓ Final dataset created: final_dataset.csv${NC}"

# ============================================================================
# Step 7: Summary
# ============================================================================

echo ""
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}Workflow Complete!${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo "Final dataset: final_dataset.csv"
echo ""

# Show dataset summary
python3 -c "
import pandas as pd
df = pd.read_csv('final_dataset.csv')
print('Dataset Summary:')
print(f'  Total rows: {len(df):,}')
print(f'  Hostnames: {df[\"hostname\"].nunique()}')
print(f'  Services: {df[\"service_name\"].nunique()}')
print(f'  Columns: {len(df.columns)}')
print('')
print('Rows per hostname:')
for hostname, count in df['hostname'].value_counts().items():
    print(f'  {hostname}: {count:,}')
"

echo ""
echo "Next steps:"
echo "  1. Analyze dataset: python3 -c 'import pandas as pd; df = pd.read_csv(\"final_dataset.csv\"); print(df.head())'"
echo "  2. Stop Azure VMs to save costs: make azure-stop"
echo "  3. Delete resources when done: az group delete -g $RESOURCE_GROUP --yes"
echo ""

# ============================================================================
# Cost Warning
# ============================================================================

echo -e "${YELLOW}⚠️  Remember to stop or delete Azure VMs when not in use!${NC}"
echo ""
echo "Stop VMs:"
echo "  az vm deallocate -g $RESOURCE_GROUP -n dataset-cpu-01"
echo "  az vm deallocate -g $RESOURCE_GROUP -n dataset-gpu-01"
echo ""
echo "Delete all resources:"
echo "  az group delete -g $RESOURCE_GROUP --yes --no-wait"
echo ""
