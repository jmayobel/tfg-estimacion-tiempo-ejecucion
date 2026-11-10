# Dataset Creator - Windows WSL2 Setup Guide

Complete guide for running the Dataset Creator on Windows with WSL2 and NVIDIA GPU support.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [GPU Setup](#gpu-setup)
- [Running the Dataset Creator](#running-the-dataset-creator)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

1. **Windows 10/11** (Build 19044 or higher)
2. **WSL2** (Windows Subsystem for Linux 2)
3. **Docker Desktop for Windows** (with WSL2 backend)
4. **NVIDIA GPU** (Optional but recommended)
5. **NVIDIA Drivers** (Latest version for your GPU)

### System Requirements

- **RAM**: 16GB minimum, 32GB recommended
- **Disk Space**: 50GB free space minimum
- **GPU**: NVIDIA GPU with CUDA support (optional)
- **CPU**: Multi-core processor (4+ cores recommended)

---

## Installation

### Step 1: Install WSL2

Open PowerShell as Administrator and run:

```powershell
# Enable WSL
wsl --install

# Or if WSL is already installed, update to WSL2
wsl --set-default-version 2

# Install Ubuntu (recommended)
wsl --install -d Ubuntu-22.04
```

Restart your computer when prompted.

### Step 2: Install Docker Desktop

1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop/
2. Install with **WSL2 backend enabled**
3. Start Docker Desktop
4. In Settings → General, ensure "Use WSL 2 based engine" is checked
5. In Settings → Resources → WSL Integration, enable integration with your specific Ubuntu instance (e.g., `Ubuntu-22.04` or `Ubuntu-Dataset-Creator`).

> **Note on Multiple WSL Instances**: If you already have a primary WSL2 instance and want to create a fresh one just for this project, you can install a new instance (e.g., `wsl --install -d Ubuntu-24.04`). Docker Desktop can share its engine across multiple WSL2 instances simultaneously. Just make sure to enable the toggle for your new instance in Docker Desktop's WSL Integration settings.

### Step 3: Set Up Ubuntu WSL2

Open Ubuntu from the Start menu and run:

```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install Python 3, pip, git, and make
sudo apt install -y python3 python3-pip python3-venv git make

# Verify installations
python3 --version  # Should be 3.10 or higher
docker --version   # Should show Docker version
make --version     # Should show Make version
```

### Step 4: Clone the Repository

```bash
# Navigate to your projects directory
cd ~
mkdir -p Projects
cd Projects

# Clone the repository (replace with your repo URL)
git clone <your-repo-url> dataset-creator
cd dataset-creator
```

---

## GPU Setup

### Step 1: Install NVIDIA CUDA Toolkit in WSL2

```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb

# Update and install CUDA toolkit
sudo apt update
sudo apt install -y cuda-toolkit-12-3
```

### Step 2: Verify GPU Access

```bash
# Check NVIDIA driver in WSL2
nvidia-smi

# Should show your GPU(s) and driver version
```

### Step 3: Configure Docker for GPU Access

The NVIDIA Container Toolkit should work automatically with Docker Desktop on WSL2.

Verify with:

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

If successful, you should see your GPU information.

---

## Running the Dataset Creator

### Step 1: Configure Settings

Edit the `.env` file in the project root:

```bash
# Edit with your preferred editor
nano .env
```

Set your configuration:

```bash
# Machine identifier
HOSTNAME_LABEL=windows-wsl2-nvidia

# Sampling configuration
SAMPLE_INTERVAL=2
MAX_SERVICE_TIMEOUT=3600  # Safety timeout in seconds

# GPU-specific services (adjust workload sizes if needed)
OLLAMA_REQUESTS=5
OLLAMA_MODEL=llama3.2:1b
FFMPEG_VIDEO_LENGTH=120
PYTORCH_GPU_EPOCHS=10
```

### Step 2: Run the Orchestrator

The easiest and most compatible way to run the orchestrator is using Docker. This ensures all dependencies are perfectly matched.

```bash
# Start the complete data collection pipeline inside a Docker container
make run-docker
```

The orchestrator will:
1. ✅ Detect your Windows WSL2 environment
2. ✅ Start Prometheus monitoring stack
3. ✅ Launch NVIDIA DCGM exporter for GPU metrics
4. ✅ Run each benchmark service sequentially
5. ✅ Collect metrics from Prometheus
6. ✅ Export to CSV files in `./output/`

### Step 4: Monitor Progress

Open a new terminal and watch the logs:

```bash
# Watch Docker containers
docker ps

# View Prometheus metrics
# Open browser: http://localhost:9090

# View cAdvisor
# Open browser: http://localhost:8080
```

### Expected Runtime

With default settings (5 minutes per service × 13 services):
- **Total time**: ~65-75 minutes
- **Output**: `./output/all_services.csv`
- **File size**: Varies (typically 5-20MB depending on sample rate)

---

## Verification

### Check Output Files

```bash
# List generated CSV files
ls -lh output/

# Should see:
# - Individual service CSVs (ollama.csv, ffmpeg.csv, etc.)
# - all_services.csv (merged file)

# View first few rows
head -n 5 output/all_services.csv

# Count total rows
wc -l output/all_services.csv
```

### Verify GPU Metrics

```bash
# Check for GPU columns in output
head -n 1 output/all_services.csv | tr ',' '\n' | grep gpu

# Should see:
# gpu_usage
# gpu_memory_usage
# gpu_memory_total
# gpu_temp
# gpu_clock_value
# gpu_name
```

### Run Tests (Optional)

```bash
# Run test suite
make test

# Generate coverage report
make coverage
```

---

## Troubleshooting

### Docker Issues

**Problem**: "Cannot connect to Docker daemon"

```bash
# Solution 1: Restart Docker Desktop from Windows

# Solution 2: Check Docker service
sudo service docker status
sudo service docker start

# Solution 3: Add user to docker group
sudo usermod -aG docker $USER
# Then log out and back in
```

**Problem**: "permission denied while trying to connect to Docker"

```bash
# Fix permissions
sudo chmod 666 /var/run/docker.sock
```

### GPU Issues

**Problem**: GPU not detected

```bash
# Check NVIDIA driver in Windows (PowerShell)
nvidia-smi

# Check in WSL2
nvidia-smi

# Reinstall CUDA toolkit if needed
sudo apt install --reinstall cuda-toolkit-12-3
```

**Problem**: DCGM exporter fails to start

```bash
# Check GPU compatibility
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# View DCGM logs
docker logs dcgm-exporter
```

### Prometheus Issues

**Problem**: No metrics collected

```bash
# Check Prometheus is running
docker ps | grep prometheus

# Check Prometheus targets
# Browser: http://localhost:9090/targets
# All should be "UP"

# View Prometheus logs
docker logs prometheus
```

### Memory Issues

**Problem**: Out of memory errors

```bash
# Increase WSL2 memory limit
# Create/edit: C:\Users\<YourUsername>\.wslconfig

[wsl2]
memory=16GB
processors=4
swap=8GB

# Then restart WSL2 (PowerShell as Admin):
wsl --shutdown
```

### Service-Specific Issues

**Problem**: Ollama fails to start

```bash
# Check Ollama logs
docker logs ollama

# Pull model manually
docker exec ollama ollama pull llama3.2:1b
```

**Problem**: Some services timeout

```bash
# Increase safety timeout in .env file
MAX_SERVICE_TIMEOUT=7200  # 2 hours instead of 1

# Or increase per-service workload:
FFMPEG_VIDEO_LENGTH=300
```

### Output Issues

**Problem**: CSV files are empty

```bash
# Check Prometheus has data
# Browser: http://localhost:9090/graph
# Run query: container_cpu_usage_seconds_total

# Check orchestrator logs
tail -f orchestrator.log

# Run export script manually with verbose logging
python3 scripts/export_csv.py --verbose \
  --prometheus-url http://localhost:9090 \
  --start-time "2024-01-01 10:00:00" \
  --end-time "2024-01-01 10:05:00" \
  --service-key test \
  --service-name test \
  --service-type benchmark \
  --architecture-type CPU \
  --hostname test-host \
  --cpu-name "Test CPU" \
  --gpu-name "Test GPU" \
  --npu-name "none" \
  --service-size 0 \
  --num-params 0 \
  --execution-time 300 \
  --container-name test-container \
  --output-dir ./output \
  --step 2
```

---

## Performance Optimization

### For Faster Data Collection

```bash
# Reduce sample interval (more data points)
SAMPLE_INTERVAL=1

# Use smaller workloads (less data)
OLLAMA_REQUESTS=3
SYSBENCH_EVENTS=5000

# Run only specific services
# Remove unwanted services from the SERVICES array
```

### For Better GPU Utilization

```bash
# Use GPU-intensive models for Ollama
OLLAMA_MODEL=llama3.2:3b  # Larger model

# Increase FFmpeg encoding complexity
# (Edit docker-compose.services.yml FFmpeg command)
```

---

## Next Steps

After successful data collection:

1. **Analyze your data**:
   ```bash
   # Use pandas to explore
   python3 -c "import pandas as pd; df = pd.read_csv('output/all_services.csv'); print(df.describe())"
   ```

2. **Merge with macOS data** (if collecting from both platforms):
   ```bash
   python3 scripts/merge_datasets.py output_windows/ output_mac/ -o dataset_merged.csv
   ```

3. **Use for ML training**:
   - Load CSV into your preferred ML framework
   - Features: CPU, GPU, memory, network, disk metrics
   - Target: Service type, performance classification, etc.

---

## Getting Help

- **GitHub Issues**: Report bugs or ask questions
- **Logs**: Check `./logs/` directory for detailed logs
- **Verbose Mode**: Add `--verbose` flag to scripts for debug output

---

## Additional Resources

- [Docker Desktop WSL2 Backend](https://docs.docker.com/desktop/wsl/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Main README](README.md) - Project overview
- [macOS Setup Guide](README-MACOS.md) - For Apple Silicon setup
