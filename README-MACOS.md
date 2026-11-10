# Dataset Creator - macOS Apple Silicon Setup Guide

Complete guide for running the Dataset Creator on macOS with Apple Silicon (M1/M2/M3/M4) and GPU/NPU metrics collection.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Apple Silicon Metrics Setup](#apple-silicon-metrics-setup)
- [Running the Dataset Creator](#running-the-dataset-creator)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

1. **macOS** 12.0 (Monterey) or later
2. **Apple Silicon** (M1, M2, M3, or M4 chip)
3. **Docker Desktop for Mac** (Apple Silicon version)
4. **Homebrew** package manager
5. **Python 3.10+**

### System Requirements

- **RAM**: 16GB minimum (24GB for M4 Pro/Max)
- **Disk Space**: 50GB free space minimum
- **CPU**: Apple Silicon (M1/M2/M3/M4)
- **macOS**: Monterey (12.0) or later

---

## Installation

### Step 1: Install Homebrew

If not already installed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Add Homebrew to PATH (follow instructions shown after install)
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

### Step 2: Install Docker Desktop

```bash
# Download and install Docker Desktop for Apple Silicon
# Visit: https://www.docker.com/products/docker-desktop/

# Or install via Homebrew
brew install --cask docker

# Start Docker Desktop from Applications folder
open -a Docker
```

Wait for Docker to start (you'll see the Docker icon in the menu bar).

### Step 3: Install Python and Git

```bash
# Install Python 3 (if not already installed)
brew install python@3.11 git

# Verify installations
python3 --version  # Should be 3.10 or higher
git --version
docker --version
```

### Step 4: Clone the Repository

```bash
# Navigate to your projects directory
cd ~/Projects  # Or your preferred location
# Create directory if it doesn't exist
mkdir -p ~/Projects
cd ~/Projects

# Clone the repository (replace with your repo URL)
git clone <your-repo-url> dataset-creator
cd dataset-creator
```

### Step 4: Clone the Repository

```bash
# Navigate to your projects directory
cd ~/Projects  # Or your preferred location
# Create directory if it doesn't exist
mkdir -p ~/Projects
cd ~/Projects

# Clone the repository (replace with your repo URL)
git clone <your-repo-url> dataset-creator
cd dataset-creator
```

---

## Apple Silicon Metrics Setup

The Dataset Creator can collect detailed GPU and Neural Engine (NPU) metrics on Apple Silicon using the `powermetrics` command.

### Step 1: Understand Permission Requirements

⚠️ **Important**: Collecting Apple Silicon GPU/NPU metrics requires `sudo` (root) access because:
- `powermetrics` is a privileged command
- It accesses hardware performance counters
- It reads thermal sensors

### Step 2: Test Powermetrics Access

```bash
# Test if powermetrics works
sudo powermetrics --samplers gpu_power,thermal,ane_power -i 2000 -n 1

# You should see output like:
# GPU HW active residency:  12.67%
# GPU idle residency:  87.33%
# ANE Power: 500 mW
# CPU die temperature: 45.30 C
```

If this works, you're ready to collect Apple Silicon metrics!

### Step 3: Optional - Passwordless Sudo for Convenience

⚠️ **Security Warning**: Only do this on your personal development machine.

```bash
# Edit sudoers file
sudo visudo

# Add this line at the end (replace 'yourusername' with your actual username):
yourusername ALL=(ALL) NOPASSWD: /usr/bin/powermetrics

# Save and exit (Ctrl+X, then Y, then Enter in nano)
```

Now you can run `sudo powermetrics` without password prompts.

### Step 4: Alternative - Use ioreg (No Sudo Required)

If you can't use `sudo`, the system will fall back to `ioreg`:
- ✅ Works without sudo
- ✅ Collects temperature data
- ❌ Less accurate GPU utilization
- ❌ No ANE (Neural Engine) metrics

---

## Running the Dataset Creator

### Step 1: Configure Settings

Edit the `.env` file in the project root:

```bash
# Edit with your preferred editor
nano .env
# Or use VS Code
code .env
```

Set your configuration:

```bash
# Machine identifier
HOSTNAME_LABEL=macos-m4-pro  # Adjust for your chip

# Sampling configuration
SAMPLE_INTERVAL=2
MAX_SERVICE_TIMEOUT=3600  # Safety timeout in seconds

# Ollama settings (uses Apple Neural Engine)
OLLAMA_REQUESTS=5
OLLAMA_MODEL=llama3.2:1b

# Other service workloads
FFMPEG_VIDEO_LENGTH=120  # Video encoding on Apple GPU (seconds of video)
SYSBENCH_EVENTS=10000
```

### Step 2: Start Apple Silicon Metrics Collector

Open a **new Terminal window** and run:

```bash
cd ~/Projects/dataset-creator
python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt

# Start the Apple Silicon metrics exporter
sudo python3 scripts/host_metrics.py --port 9101 --interval 2

# Or with verbose logging:
sudo python3 scripts/host_metrics.py --port 9101 --interval 2 --verbose
```

**Keep this terminal window open** during data collection. You should see:

```
Apple Silicon Host Metrics Exporter
  GPU: Apple M4 Pro
  Total Memory: 24576 MB
  Port: 9101
  Interval: 2s
  Running as root: True
  Started: powermetrics collector (sudo)
  Started: ioreg thermal collector
  Started: IOKit GPU collector

Serving metrics at http://0.0.0.0:9101/metrics
```

### Step 3: Run the Orchestrator

In your **original terminal**:

The easiest and most compatible way to run the orchestrator is using Docker. This ensures all dependencies are perfectly matched.

```bash
# Ensure metrics collector is running (check other terminal)

# Start the complete data collection pipeline inside a Docker container
make run-docker
```

The orchestrator will:
1. ✅ Detect your macOS Apple Silicon environment
2. ✅ Start Prometheus monitoring stack
3. ✅ Connect to Apple Silicon metrics exporter (port 9101)
4. ✅ Run each benchmark service sequentially
5. ✅ Collect CPU, GPU, and ANE metrics from Prometheus
6. ✅ Export to CSV files in `./output/`

### Step 5: Monitor Progress

Open a new terminal and watch:

```bash
# Watch Docker containers
docker ps

# View Prometheus metrics
# Open browser: http://localhost:9090

# Check Apple Silicon metrics
curl http://localhost:9101/metrics

# View debug endpoint
curl http://localhost:9101/debug | jq
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

### Verify Apple Silicon Metrics

```bash
# Check for Apple-specific columns
head -n 1 output/all_services.csv | tr ',' '\n' | grep -E '(gpu|npu|ane)'

# Should see:
# gpu_usage (Apple GPU utilization)
# gpu_memory_usage
# gpu_memory_total
# gpu_temp
# gpu_clock_value
# npu_usage (Apple Neural Engine)
# npu_memory_usage
# npu_temp
```

### Check Metric Quality

```bash
# Install dependencies if needed
pip install pandas

# Analyze collected data
python3 << EOF
import pandas as pd
df = pd.read_csv('output/all_services.csv')

print("Dataset Summary:")
print(f"Total rows: {len(df)}")
print(f"Services: {df['service_name'].nunique()}")
print(f"\nApple GPU metrics:")
print(df[['gpu_usage', 'gpu_temp']].describe())
print(f"\nApple ANE metrics:")
print(df[['npu_usage']].describe())
EOF
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
# Solution 1: Start Docker Desktop
open -a Docker

# Solution 2: Check Docker status
docker ps

# Solution 3: Restart Docker Desktop
# Quit from menu bar, then restart
```

**Problem**: Docker containers are slow

```bash
# Increase Docker resources in Docker Desktop settings:
# Docker Desktop → Preferences → Resources
# - CPUs: 4-8 (depending on your Mac)
# - Memory: 8-16GB
# - Swap: 2GB
# - Disk: 100GB
```

### Apple Silicon Metrics Issues

**Problem**: "powermetrics not found"

```bash
# powermetrics should be built-in on macOS
# Verify path:
which powermetrics
# Should show: /usr/bin/powermetrics

# Try running directly:
sudo /usr/bin/powermetrics --help
```

**Problem**: GPU metrics always showing 0

```bash
# Check if metrics collector is running
curl http://localhost:9101/metrics | grep apple_gpu

# Check debug output
curl http://localhost:9101/debug | jq

# Restart with verbose logging
sudo python3 scripts/host_metrics.py --verbose
```

**Problem**: "Permission denied" for powermetrics

```bash
# You must use sudo
sudo python3 scripts/host_metrics.py

# Or set up passwordless sudo (see installation section)
```

**Problem**: ANE (Neural Engine) metrics missing

```bash
# ANE metrics depend on powermetrics with sudo
# Check if running as root:
curl http://localhost:9101/debug | jq .apple_ane_utilization

# If 0.0, ensure:
# 1. Running with sudo
# 2. Using ANE-capable workload (Ollama with ML model)
# 3. Model is actually running inference
```

### Prometheus Issues

**Problem**: Prometheus not scraping Apple metrics

```bash
# Check Prometheus targets
# Browser: http://localhost:9090/targets
# Look for: host.docker.internal:9101

# Test metrics endpoint
curl http://host.docker.internal:9101/metrics

# Check Prometheus config
cat prometheus/prometheus.yml | grep -A 5 host.docker.internal
```

**Problem**: No data in Prometheus

```bash
# Check if metrics exporter is running
lsof -i :9101

# Restart metrics exporter
sudo python3 scripts/host_metrics.py

# View Prometheus logs
docker logs prometheus
```

### Service-Specific Issues

**Problem**: Ollama fails to start or crashes

```bash
# Check Ollama logs
docker logs ollama

# Pull model manually
docker exec ollama ollama pull llama3.2:1b

# Try smaller model if out of memory
# Edit .env: OLLAMA_MODEL=llama3.2:1b
```

**Problem**: FFmpeg encoding is very slow

```bash
# This is normal on CPU-only encoding
# Apple Silicon doesn't support NVENC (NVIDIA encoder)

# To speed up:
# 1. Use shorter video duration
# 2. Lower resolution
# Edit docker-compose.services.yml FFmpeg command
```

**Problem**: Some services not available for ARM64

```bash
# Some Docker images may not support ARM64
# Check orchestrator output for warnings

# Skip unsupported services by commenting them out in:
# docker-compose.services.yml
```

### Memory Issues

**Problem**: "Out of memory" errors

```bash
# Check system memory
top -l 1 | grep PhysMem

# Close other applications

# Increase Docker memory limit:
# Docker Desktop → Preferences → Resources → Memory
# Set to at least 8GB

# Or reduce concurrent services
# Edit orchestrator to run fewer services
```

### Performance Issues

**Problem**: Mac gets very hot

```bash
# This is normal during intensive benchmarking
# Monitor temperatures:
sudo powermetrics --samplers thermal -i 5000 -n 1

# To reduce heat:
# 1. Reduce service durations
# 2. Increase time between services
# 3. Use cooling pad
# 4. Run at night when ambient temperature is lower
```

---

## Performance Optimization

### For Faster Data Collection

```bash
# Reduce sample interval (more granular data)
SAMPLE_INTERVAL=1

# Use smaller workloads (less total data)
OLLAMA_REQUESTS=3
SYSBENCH_EVENTS=5000

# Run only CPU-intensive services
# Remove network/disk benchmarks from the SERVICES array
```

### For Better GPU/ANE Utilization

```bash
# Use larger models for Ollama
OLLAMA_MODEL=llama3.2:3b  # or llama2:7b

# Run multiple inference requests
# Edit docker-compose.services.yml Ollama command

# Increase FFmpeg complexity
# Use higher resolution or bitrate
```

### For Production Data Collection

```bash
# Larger workloads for stable metrics
OLLAMA_REQUESTS=10
SYSBENCH_EVENTS=50000

# Higher sample rate
SAMPLE_INTERVAL=1

# Run overnight to avoid thermal throttling
# Use at command or cron job:
echo "./scripts/orchestrator.sh" | at 02:00
```

---

## Next Steps

After successful data collection:

1. **Analyze your data**:
   ```bash
   python3 << EOF
   import pandas as pd
   df = pd.read_csv('output/all_services.csv')

   print("Services collected:")
   print(df['service_name'].value_counts())

   print("\nApple GPU utilization:")
   print(df.groupby('service_name')['gpu_usage'].agg(['mean', 'max', 'std']))

   print("\nApple ANE utilization:")
   print(df.groupby('service_name')['npu_usage'].agg(['mean', 'max', 'std']))
   EOF
   ```

2. **Merge with Windows data** (if collecting from both platforms):
   ```bash
   python3 scripts/merge_datasets.py \
     /path/to/windows/output \
     /path/to/macos/output \
     -o dataset_merged.csv
   ```

3. **Compare platforms**:
   ```bash
   python3 << EOF
   import pandas as pd
   df = pd.read_csv('dataset_merged.csv')

   print("Performance by platform:")
   print(df.groupby(['hostname', 'service_name'])['cpu_usage'].mean().unstack())
   EOF
   ```

---

## Advanced: Automated Runs

### Using launchd (macOS Task Scheduler)

Create a plist file for automated daily runs:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.dataset-creator.daily</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/dataset-creator/scripts/orchestrator.sh</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>2</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
</dict>
</plist>
```

Load the job:
```bash
launchctl load ~/Library/LaunchAgents/com.dataset-creator.daily.plist
```

---

## Getting Help

- **GitHub Issues**: Report bugs or ask questions
- **Logs**: Check `./logs/` directory for detailed logs
- **Verbose Mode**: Add `--verbose` flag to scripts for debug output
- **Apple Developer Forums**: For Apple Silicon specific issues

---

## Additional Resources

- [Docker Desktop for Mac](https://docs.docker.com/desktop/mac/)
- [Apple Silicon Performance](https://developer.apple.com/metal/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Main README](README.md) - Project overview
- [Windows Setup Guide](README-WINDOWS.md) - For Windows WSL2 setup

---

## Apple Silicon Specific Notes

### M1/M2/M3/M4 Differences

- **M1/M2**: 8-16 GPU cores, 16-core ANE
- **M3**: Up to 40 GPU cores, enhanced ray tracing
- **M4**: Up to 40 GPU cores, improved ANE performance

The metrics collector automatically adapts to your chip.

### Thermal Management

Apple Silicon uses aggressive thermal throttling:
- Monitor temperature with: `sudo powermetrics --samplers thermal`
- Optimal operating range: < 80°C
- If throttling occurs, data quality may vary

### Memory Architecture

Apple Silicon uses unified memory (shared between CPU/GPU/ANE):
- `gpu_memory_usage` and `npu_memory_usage` are estimates
- Ratios: GPU ~30%, ANE ~10% of utilization-based allocation
- Actual memory usage may vary by workload
