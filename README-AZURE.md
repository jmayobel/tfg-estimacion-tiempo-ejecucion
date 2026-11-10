# Dataset Creator - Azure VM Deployment Guide

Complete guide for deploying the Dataset Creator on Azure Virtual Machines (both CPU-only and GPU-enabled).

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [VM Deployment](#vm-deployment)
- [VM Setup](#vm-setup)
- [Running Dataset Collection](#running-dataset-collection)
- [Multi-Machine Workflow](#multi-machine-workflow)
- [Troubleshooting](#troubleshooting)
- [Cost Optimization](#cost-optimization)
- [Advanced Topics](#advanced-topics)

---

## Prerequisites

Before you begin, ensure you have:

### Local Machine Requirements

1. **Azure Subscription**
   - Active Azure subscription ([create free account](https://azure.microsoft.com/free/))
   - Sufficient quota for desired VM SKUs (especially GPU VMs)

2. **Azure CLI**

   ```bash
   # macOS
   brew install azure-cli

   # Windows
   winget install Microsoft.AzureCLI

   # Linux
   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
   ```

3. **SSH Key Pair**
   ```bash
   # Generate if you don't have one
   ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

   # View your public key
   cat ~/.ssh/id_rsa.pub
   ```

4. **Required Tools**
   - `jq` (JSON processor): `brew install jq` (macOS) or `sudo apt-get install jq` (Linux)
   - `rsync` (for dataset download): Usually pre-installed

### Knowledge Requirements

- Basic understanding of Linux command line
- Familiarity with Docker and containerization
- Basic Azure concepts (Resource Groups, VMs, regions)

---

## Quick Start

**TL;DR**: Deploy, setup, run, and collect

```bash
# 1. Configure SSH public key
cd azure
nano parameters.json  # Add your SSH public key

# 2. Deploy VM
./deploy.sh

# 3. Setup VM (use IP from deployment output)
./setup-vm.sh  # Will prompt for VM IP

# 4. SSH into VM and run dataset collection
ssh azureuser@<vm-ip>
cd ~/dataset-creator
./scripts/orchestrator.sh

# 5. Download results (from local machine)
cd ~/dataset-creator
./scripts/download_datasets.sh

# 6. Merge with local datasets
python3 scripts/merge_datasets.py datasets/* output_local -o final_dataset.csv
```

---

## VM Deployment

### Step 1: Choose VM Size

Select the appropriate VM SKU based on your workload:

> **Azure for Students subscription note:** the regional vCPU cap is **6 vCPUs**, so any SKU > 4 vCPU (e.g. `Standard_D8s_v3`) will be rejected. All NC* GPU SKUs start at quota 0 and require a one-time quota request before they can be deployed.

#### CPU-Only VMs

| SKU | vCPU | RAM | Storage | Use Case | Cost/hour* |
|-----|------|-----|---------|----------|------------|
| `Standard_D2s_v3` | 2 | 8 GB | 16 GB SSD | Minimal (testing only) | ~$0.10 |
| `Standard_D4s_v3` | 4 | 16 GB | 32 GB SSD | **Recommended CPU (Intel)** | ~$0.19 |
| `Standard_D4as_v4` | 4 | 16 GB | 32 GB SSD | Recommended CPU (AMD EPYC) | ~$0.18 |
| `Standard_D4ds_v4` | 4 | 16 GB | + local NVMe | Best for FIO disk benchmarks | ~$0.21 |

#### GPU VMs (quota request required before first use)

| SKU | vCPU | RAM | GPU | Use Case | Cost/hour* |
|-----|------|-----|-----|----------|------------|
| `Standard_NC4as_T4_v3` | 4 | 28 GB | Tesla T4 (16GB) | **Recommended GPU** | ~$0.53 |
| `Standard_NC8as_T4_v3` | 8 | 56 GB | Tesla T4 (16GB) | More CPU alongside same GPU | ~$0.75 |
| `Standard_NC6s_v3` | 6 | 112 GB | Tesla V100 (16GB) | Faster GPU, rarely granted to students | ~$3.06 |

> Retired SKUs (`Standard_NC6/12/24` — Tesla K80) are intentionally excluded: they're unavailable in most regions and incompatible with the `nvidia-driver-535` package that `setup-vm.sh` installs.

\**Prices are approximate for West Europe. Check [Azure Pricing Calculator](https://azure.microsoft.com/pricing/calculator/) for exact rates.*

**Recommendation for Azure for Students:** start with `Standard_D4s_v3` (no quota request needed). To run the GPU dataset, file a quota request for the `Standard NCASv3_T4 Family` (4 vCPUs in `westeurope`) — usually approved within 24h for academic projects — then deploy `Standard_NC4as_T4_v3`.

### Step 2: Configure Deployment Parameters

Edit `azure/parameters.json`:

```bash
cd azure
nano parameters.json
```

**Required changes**:

```json
{
  "parameters": {
    "vmName": {
      "value": "dataset-creator-vm"  // Choose a unique name
    },
    "vmSize": {
      "value": "Standard_D4s_v3"  // Or your chosen SKU
    },
    "sshPublicKey": {
      "value": "ssh-rsa AAAAB3NzaC1yc2EA..."  // ⚠️ REPLACE WITH YOUR PUBLIC KEY
    }
  }
}
```

**Get your SSH public key**:
```bash
cat ~/.ssh/id_rsa.pub
```

Copy the entire output (starting with `ssh-rsa`) and paste it into the `sshPublicKey` value.

### Step 3: Deploy the VM

```bash
cd azure
./deploy.sh
```

**What this does**:
1. Creates Azure Resource Group (default: `dataset-creator-rg`)
2. Deploys Ubuntu 22.04 LTS VM
3. Configures network security groups (SSH port 22, Prometheus port 9090)
4. Assigns static public IP
5. Outputs connection information

**Example output**:
```
============================================================================
Deployment Information
============================================================================

VM Details:
  VM Size:       Standard_D4s_v3
  GPU Enabled:   false
  Public IP:     20.123.45.67
  FQDN:          dataset-creator-vm.westeurope.cloudapp.azure.com

SSH Access:
  Command:       ssh azureuser@20.123.45.67
```

**Connection info saved to**: `azure/connection-info.txt`

### Step 4: Wait for VM Boot

After deployment, wait ~2 minutes for the VM to fully boot before connecting.

---

## VM Setup

### Option 1: Automated Setup (Recommended)

```bash
cd azure
./setup-vm.sh
```

The script will:
1. Prompt for VM IP address
2. Copy setup script to VM via SCP
3. Execute setup remotely via SSH
4. Install Docker, Python, and dependencies
5. Clone the dataset-creator repository
6. Configure the environment
7. (GPU VMs) Install NVIDIA drivers

**Expected duration**: 5-10 minutes

### Option 2: Manual Setup

If you prefer manual control:

```bash
# 1. Copy setup script to VM
VM_IP="20.123.45.67"  # Replace with your VM IP
scp azure/setup-vm.sh azureuser@$VM_IP:~/

# 2. SSH into VM
ssh azureuser@$VM_IP

# 3. Run setup script
bash ~/setup-vm.sh

# 4. Log out and back in (for docker group changes)
exit
ssh azureuser@$VM_IP

# 5. Verify installation
docker --version
python3 --version
ls ~/dataset-creator
```

### Post-Setup Verification

#### CPU-Only VMs

```bash
ssh azureuser@<vm-ip>

# Test Docker
docker run --rm hello-world

# Verify Python
python3 --version  # Should be 3.10+

# Check repository
cd ~/dataset-creator
ls scripts/  # Should show orchestrator.sh

# Check environment
cat .env  # Verify HOSTNAME_LABEL is set
```

#### GPU VMs

```bash
ssh azureuser@<vm-ip>

# Verify NVIDIA driver
nvidia-smi
# Should show GPU details (Tesla K80, V100, etc.)

# Test NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Verify DCGM Exporter
cd ~/dataset-creator
docker compose -f docker-compose.gpu.yml up -d
curl http://localhost:9400/metrics | grep dcgm
```

**GPU Setup Note**: You may need to reboot after the initial setup for NVIDIA drivers to fully activate:

```bash
sudo reboot
# Wait 2 minutes, then reconnect
```

---

## Running Dataset Collection

### Manual Execution (On VM)

SSH into the VM and run the orchestrator:

```bash
ssh azureuser@<vm-ip>
cd ~/dataset-creator

# Optional: configure busqueda-caminos load (if using that service)
# Ensure the instance file exists on the VM
# BUSQUEDA_URL=http://localhost:5000/process
# BUSQUEDA_INSTANCE_FILE="/home/azureuser/E-VRPTW Instances/your_instance.txt"

# Start dataset collection inside a Docker container
make run-docker
```

**What happens**:

1. Starts monitoring stack (Prometheus, cAdvisor, Node-Exporter)
2. For GPU VMs: Starts DCGM Exporter or Apple metrics
3. Runs 13 benchmark services sequentially (Ollama, FFmpeg, Sysbench, etc.)
4. Collects metrics every 2 seconds during execution
5. Exports metrics to CSV files in `output/` directory
6. Merges per-service CSVs into `output/all_services.csv`

**Duration**: ~3-4 hours with 15-minute services; shorter if you reduce per-service durations

**Monitoring progress**:
```bash
# In another SSH session
ssh azureuser@<vm-ip>
tail -f ~/dataset-creator/logs/orchestrator.log

# Or watch Docker containers
watch docker ps
```

### Remote Execution (From Local Machine)

Use the remote collection script to trigger execution without manual SSH:

```bash
# 1. Configure remote hosts
cd ~/dataset-creator
nano scripts/remote_hosts.txt

# Add your VM:
# azure-cpu-01,azureuser,20.123.45.67,Azure D4s_v3 CPU VM

# 2. Trigger remote collection
./scripts/collect_remote.sh azure-cpu-01

# 3. Or collect from all configured hosts
./scripts/collect_remote.sh
```

The script will:
- SSH into the VM
- Execute `make run-docker`
- Stream logs to local machine
- Save logs to `logs/remote-collection/azure-cpu-01_<timestamp>.log`

---

## Multi-Machine Workflow

### Architecture: Azure VMs + Local Machines

```
┌─────────────────────────────────────────────────────────┐
│ Local Machine (macOS M4 Pro)                            │
│ - Controls orchestration                                │
│ - Aggregates datasets                                   │
└─────────────────────────────────────────────────────────┘
             │
             │ SSH
             ├─────────────────────┬─────────────────────┐
             ↓                     ↓                     ↓
     ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
     │ Azure CPU VM   │    │ Azure GPU VM   │    │ Local Windows  │
     │ (D4s_v3)       │    │ (NC6)          │    │ (RTX 3060)     │
     │ East US        │    │ East US        │    │ WSL2           │
     └────────────────┘    └────────────────┘    └────────────────┘
             │                     │                     │
             └─────────────────────┴─────────────────────┘
                                   │
                          Datasets Downloaded
                                   │
                                   ↓
                         ┌──────────────────┐
                         │ Merged Dataset   │
                         │ final_dataset.csv│
                         └──────────────────┘
```

### Step-by-Step Workflow

#### 1. Configure Remote Hosts

Edit `scripts/remote_hosts.txt`:

```bash
# hostname,ssh_user,ip_address,description
azure-cpu-01,azureuser,20.123.45.67,Azure D4s_v3 CPU-only
azure-gpu-01,azureuser,20.123.45.68,Azure NC6 with Tesla K80
windows-rtx3060,user,192.168.1.100,Local Windows WSL2 with RTX 3060
macos-m4-pro,user,192.168.1.101,Local macOS M4 Pro
```

**Important**: Ensure:
- SSH key-based authentication is configured for all hosts
- `HOSTNAME_LABEL` in each machine's `.env` matches the hostname in this file
- Dataset creator is installed at `~/dataset-creator` on all hosts

#### 2. Trigger Collection on All Machines

**Sequential execution** (safer, easier to monitor):
```bash
./scripts/collect_remote.sh
```

**Parallel execution** (faster, but harder to monitor):
```bash
./scripts/collect_remote.sh --parallel
```

**Specific hosts only**:
```bash
./scripts/collect_remote.sh azure-cpu-01 azure-gpu-01
```

#### 3. Download Datasets

Once collection completes:

```bash
./scripts/download_datasets.sh
```

**This creates**:
```
datasets/
├── azure-cpu-01/
│   └── all_services.csv
├── azure-gpu-01/
│   └── all_services.csv
├── windows-rtx3060/
│   └── all_services.csv
└── macos-m4-pro/
    └── all_services.csv
```

#### 4. Merge Datasets

Combine all datasets into a single CSV:

```bash
python3 scripts/merge_datasets.py \
    datasets/azure-cpu-01 \
    datasets/azure-gpu-01 \
    datasets/windows-rtx3060 \
    datasets/macos-m4-pro \
    -o final_dataset.csv
```

**Merge process**:
- Concatenates all CSVs
- Deduplicates based on `hostname + timestamp + service_name`
- Sorts by `hostname`, `service_name`, `timestamp`
- Outputs summary statistics

**Example output**:
```
============================================================
Merged Dataset: final_dataset.csv
  Total rows: 142,500
  Hostnames: 4
  Services: 13
  Columns: 33

  Rows per hostname:
    azure-cpu-01: 35,000
    azure-gpu-01: 36,500
    windows-rtx3060: 35,500
    macos-m4-pro: 35,500

  Rows per service:
    ollama: 12,000
    ffmpeg: 10,500
    sysbench: 11,000
    ...
============================================================
```

#### 5. Analyze Results

```bash
# View schema
head -n 1 final_dataset.csv

# Count rows
wc -l final_dataset.csv

# Filter by hostname
grep "azure-gpu-01" final_dataset.csv > azure_gpu_only.csv

# Filter by service
grep "ollama" final_dataset.csv > ollama_only.csv

# Load into Python/Pandas
python3
>>> import pandas as pd
>>> df = pd.read_csv('final_dataset.csv')
>>> df.info()
>>> df.groupby('hostname')['gpu_usage'].mean()
```

---

## Troubleshooting

### Deployment Issues

#### Error: "SSH public key is invalid"

**Symptom**: Deployment fails with SSH key error

**Solution**:
```bash
# Ensure your public key is correctly formatted
cat ~/.ssh/id_rsa.pub
# Should start with: ssh-rsa AAAAB3NzaC1yc2...

# Regenerate if needed
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa
```

#### Error: "Quota exceeded for VM size"

**Symptom**: Deployment fails with quota error for GPU VMs

**Solution**:
1. Check current quota: Azure Portal → Subscriptions → Usage + quotas
2. Request quota increase: Support → New support request → Quota
3. Alternative: Choose a different region with available quota
4. Temporary: Use CPU-only VM while waiting for quota approval

#### Error: "Location does not support VM size"

**Symptom**: Deployment fails because region doesn't support chosen SKU

**Solution**:
```bash
# List available VM sizes in a region
az vm list-sizes --location westeurope --output table | grep NC

# Check your current quota for a SKU family before deploying:
az vm list-usage --location westeurope -o table | grep -E "NCAS|DSv3|Total Regional"

# Try alternative regions:
# - GPU VMs: westeurope, northeurope, francecentral, eastus, westus2
# - CPU VMs: available in all regions
```

### Setup Issues

#### Error: "Docker installation fails"

**Symptom**: `setup-vm.sh` fails during Docker installation

**Solution**:
```bash
# SSH into VM
ssh azureuser@<vm-ip>

# Manually install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Log out and back in
exit
ssh azureuser@<vm-ip>

# Verify
docker run --rm hello-world
```

#### Error: "nvidia-smi not found" (GPU VMs)

**Symptom**: GPU not detected after setup

**Solution**:
```bash
# Reboot the VM (required for driver activation)
sudo reboot

# Wait 2 minutes, then reconnect
ssh azureuser@<vm-ip>

# Verify GPU
nvidia-smi

# If still not working, manually install drivers
sudo apt-get update
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt-get install nvidia-driver-535 -y
sudo reboot
```

#### Error: "Repository clone fails"

**Symptom**: `setup-vm.sh` cannot clone repository

**Solution**:
```bash
# Update REPO_URL in setup-vm.sh
nano azure/setup-vm.sh
# Change: REPO_URL="https://github.com/YOUR_USERNAME/dataset-creator.git"

# Or manually clone on VM
ssh azureuser@<vm-ip>
git clone https://github.com/YOUR_USERNAME/dataset-creator.git ~/dataset-creator
```

### Execution Issues

#### Error: "Prometheus not accessible"

**Symptom**: Orchestrator fails with "Connection refused" to Prometheus

**Solution**:
```bash
ssh azureuser@<vm-ip>
cd ~/dataset-creator

# Check if monitoring stack is running
docker ps | grep prometheus

# Start monitoring stack manually
docker compose -f docker-compose.monitoring.yml up -d

# Wait 10 seconds
sleep 10

# Verify Prometheus
curl http://localhost:9090/-/healthy

# Then run orchestrator
./scripts/orchestrator.sh
```

#### Error: "Permission denied" for Docker

**Symptom**: Docker commands fail with permission error

**Solution**:
```bash
# Add user to docker group (if not done)
sudo usermod -aG docker $USER

# Log out and back in
exit
ssh azureuser@<vm-ip>

# Verify (should work without sudo)
docker ps
```

#### Error: "Out of disk space"

**Symptom**: Docker operations fail due to lack of space

**Solution**:
```bash
# Check disk usage
df -h

# Clean Docker images and containers
docker system prune -a --volumes -f

# If OS disk is full, consider:
# 1. Increase OS disk size in Azure Portal
# 2. Add data disk (set enableDataDisk=true in parameters.json)
```

### SSH Connection Issues

#### Error: "Connection timed out"

**Symptom**: Cannot SSH to VM

**Solution**:
```bash
# 1. Verify VM is running
az vm list -g dataset-creator-rg --output table

# 2. Start VM if deallocated
az vm start -g dataset-creator-rg -n dataset-creator-vm

# 3. Check Network Security Group
# Azure Portal → VM → Networking → Verify port 22 is allowed

# 4. Verify your IP isn't blocked
# If you have a dynamic IP, it may have changed
```

#### Error: "Host key verification failed"

**Symptom**: SSH warns about changed host key

**Solution**:
```bash
# Remove old host key
ssh-keygen -R <vm-ip>

# Or remove old key manually
nano ~/.ssh/known_hosts
# Delete line with the VM IP

# Reconnect
ssh azureuser@<vm-ip>
```

---

## Cost Optimization

### Minimize Costs

1. **Stop VMs when not in use**:
   ```bash
   # Deallocate VM (stops billing for compute)
   az vm deallocate -g dataset-creator-rg -n dataset-creator-vm

   # Start when needed
   az vm start -g dataset-creator-rg -n dataset-creator-vm
   ```

2. **Use Azure Spot VMs** (up to 90% discount):
   - Edit `azure/vm-template.bicep`
   - Add spot VM configuration
   - Note: Can be evicted with 30-second notice

3. **Choose cheaper regions**:
   - US regions often cheaper than Europe/Asia
   - Check [Azure Pricing Calculator](https://azure.microsoft.com/pricing/calculator/)

4. **Delete resources after collection**:
   ```bash
   # Delete entire resource group
   az group delete -g dataset-creator-rg --yes --no-wait
   ```

5. **Use auto-shutdown**:
   - Azure Portal → VM → Auto-shutdown
   - Set daily shutdown time (e.g., 6 PM)

### Cost Breakdown

**Example**: Running a full CPU + GPU dataset collection (sequential, not parallel)

| Resource | SKU | Duration | Cost |
|----------|-----|----------|------|
| CPU VM | Standard_D4s_v3 | 6 hours | $1.15 |
| AMD CPU VM | Standard_D4as_v4 | 6 hours | $1.08 |
| GPU VM | Standard_NC4as_T4_v3 | 6 hours | $3.18 |
| Storage | SSD 128GB | 1 day | $0.38 |
| Network | Egress 10GB | - | $0.87 |
| **Total for full TFG dataset** | | | **~$6.66** |

**Hours your $100 Azure for Students credit buys** (each SKU run alone, 24/7):
- `Standard_D4s_v3`: ~525 hours
- `Standard_D4as_v4`: ~555 hours
- `Standard_NC4as_T4_v3`: ~190 hours

**Recommendation**: deallocate VMs the moment a benchmark run finishes (`make azure-stop`). The orchestrator is sequential — there's no benefit to keeping VMs hot between runs.

---

## Advanced Topics

### Custom VM Sizes

To use VM sizes not listed in the template:

1. Edit `azure/vm-template.bicep`
2. Add your desired SKU to the `@allowed` list:
   ```bicep
   @allowed([
     'Standard_D4s_v3'
     'Standard_NC4as_T4_v3'
     'Standard_F16s_v2'  // Add this
   ])
   param vmSize string
   ```

3. Update `parameters.json` with the new SKU
4. Verify your subscription has quota for the SKU family:
   ```bash
   az vm list-usage --location westeurope -o table | grep "<family-name>"
   ```

### Multiple VMs in Parallel

> **Azure for Students:** the 6-vCPU regional cap means you usually cannot run a 4-vCPU CPU VM and a 4-vCPU GPU VM in the same region at the same time. Run them sequentially, or place them in two different regions.

Deploy multiple VMs simultaneously (paid subscriptions, or split across regions):

```bash
# CPU VM
cd azure
cp parameters.json parameters-cpu.json
# Edit: vmName="dataset-cpu-01", vmSize="Standard_D4s_v3"
./deploy.sh -p parameters-cpu.json &

# GPU VM (in new terminal) - requires NCASv3_T4 quota
cd azure
cp parameters.json parameters-gpu.json
# Edit: vmName="dataset-gpu-01", vmSize="Standard_NC4as_T4_v3"
./deploy.sh -p parameters-gpu.json &

# Wait for both to complete
wait
```

### Automated Backup to Azure Storage

Store datasets in Azure Blob Storage for long-term retention:

```bash
# Create storage account
az storage account create \
  --name datasetcreatorblob \
  --resource-group dataset-creator-rg \
  --location westeurope \
  --sku Standard_LRS

# Upload dataset
az storage blob upload \
  --account-name datasetcreatorblob \
  --container-name datasets \
  --name final_dataset.csv \
  --file final_dataset.csv
```

### VPN Setup for Private Networking

For enhanced security, place VMs in a private VNet accessible only via VPN:

1. Create Azure VPN Gateway
2. Configure point-to-site VPN
3. Update `vm-template.bicep` to use private IPs only
4. Connect via VPN before SSH

See [Azure VPN Gateway documentation](https://docs.microsoft.com/azure/vpn-gateway/) for details.

### GPU-Specific Optimizations

For maximum GPU utilization:

1. **Increase Ollama workload**:
   ```bash
   # Edit .env on VM
   OLLAMA_REQUESTS=10  # More inference requests for thorough testing
   ```

2. **Run GPU-intensive services only**:
   ```bash
   # Edit .env
   OLLAMA_REQUESTS=10
   FFMPEG_VIDEO_LENGTH=300
   PYTORCH_GPU_EPOCHS=20

   # To skip CPU-only services, remove them from the SERVICES array
   # in orchestrator.sh (e.g., remove sysbench, 7zip entries)
   ```

3. **Monitor GPU metrics in real-time**:
   ```bash
   ssh azureuser@<vm-ip>
   watch -n 2 nvidia-smi
   ```

### Spot Instance Configuration

Use Azure Spot VMs for up to 90% cost savings:

**Note**: Spot VMs can be evicted with 30 seconds notice when Azure needs the capacity.

Edit `azure/vm-template.bicep` and add:

```bicep
resource vm 'Microsoft.Compute/virtualMachines@2023-09-01' = {
  name: vmName
  location: location
  properties: {
    priority: 'Spot'
    evictionPolicy: 'Deallocate'
    billingProfile: {
      maxPrice: -1  // Pay up to on-demand price
    }
    // ... rest of configuration
  }
}
```

**Recommendation**: Only use Spot VMs for:
- Non-critical, interruptible workloads
- Batch processing that can resume
- Development/testing

---

## FAQ

**Q: Can I use Windows Server VMs instead of Linux?**
A: No, the dataset creator requires Docker Engine which runs natively on Linux. Windows Server would require WSL2 setup, making it unnecessarily complex.

**Q: How do I access Prometheus/Grafana remotely?**
A: SSH port forwarding:
```bash
ssh -L 9090:localhost:9090 azureuser@<vm-ip>
# Then access http://localhost:9090 in your browser
```

**Q: Can I run multiple dataset collections simultaneously on the same VM?**
A: No, the orchestrator is designed for sequential service execution to avoid metric interference.

**Q: How do I update the dataset creator code on Azure VMs?**
A:
```bash
ssh azureuser@<vm-ip>
cd ~/dataset-creator
git pull origin main
pip3 install -r scripts/requirements.txt --upgrade
```

**Q: What happens if a service fails during collection?**
A: The orchestrator continues with the next service. Failed services will have no data in the final CSV. Check logs in `output/<service>_errors.log`.

**Q: Can I collect from Azure VMs and local machines simultaneously?**
A: Yes! Use `./scripts/collect_remote.sh --parallel` to trigger collection on all configured hosts at once.

---

## Next Steps

1. **Deploy your first Azure VM**: [VM Deployment](#vm-deployment)
2. **Run dataset collection**: [Running Dataset Collection](#running-dataset-collection)
3. **Integrate with local machines**: [Multi-Machine Workflow](#multi-machine-workflow)
4. **Analyze your datasets**: See main [README.md](README.md) for analysis tools

---

## Support & Resources

- **Azure Documentation**: https://docs.microsoft.com/azure/virtual-machines/
- **Azure Pricing Calculator**: https://azure.microsoft.com/pricing/calculator/
- **Bicep Language Reference**: https://docs.microsoft.com/azure/azure-resource-manager/bicep/
- **Dataset Creator Issues**: https://github.com/YOUR_USERNAME/dataset-creator/issues

---

**Generated with Dataset Creator v1.0**
