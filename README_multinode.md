# Multi-Node Dataset Collection Guide

This guide explains how to collect performance metrics from two different environments (Windows 11 WSL2 with NVIDIA GPU, and macOS M4 with Apple Silicon) and merge them into a unified dataset.

## Prerequisites

### Windows 11 (WSL2)
1.  **WSL2 Installed**: Ensure you are running in a WSL2 environment (Ubuntu recommended).
2.  **Docker Desktop**: Installed and integrated with WSL2.
3.  **NVIDIA Drivers**: Installed on Windows.
4.  **NVIDIA Container Toolkit**: Installed in WSL2 to allow Docker to access the GPU.
    ```bash
    # Verification
    nvidia-smi
    docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
    ```
5.  **Python 3**: Installed (`sudo apt install python3 python3-pip`).

### macOS (Apple Silicon M4)
1.  **Docker Desktop**: Installed and running.
2.  **Python 3**: Installed.
3.  **Sudo Access**: Required for `powermatrics` (GPU/NPU monitoring).

---

## 1. Run Collection on Windows (WSL2)

1.  Open your WSL2 terminal and navigate to the project directory.
2.  Install Python dependencies:
    ```bash
    pip3 install -r scripts/requirements.txt
    ```
3.  Run the orchestrator:
    ```bash
    # Optional: Set workload sizes (defaults are in scripts/orchestrator.sh)
    # export OLLAMA_REQUESTS=5
    
    ./scripts/orchestrator.sh
    ```
    - This will start the services, monitoring stack (Prometheus + Cadvisor + DCGM Exporter), and benchmark workloads.
    - It will automatically detect the NVIDIA GPU.
4.  **Output**: accurate CSV files will be generated in the `output/` directory.
5.  **Transfer**: Copy the `output` directory (or its contents) to your macOS machine (e.g., to a folder named `output_windows`).

---

## 2. Run Collection on macOS

1.  Open your terminal and navigate to the project directory.
2.  Install Python dependencies:
    ```bash
    pip3 install -r scripts/requirements.txt
    ```
3.  **Start Host Metrics Exporter** (Crucial for Apple Silicon metrics):
    ```bash
    sudo python3 scripts/host_metrics.py
    ```
    - Leave this running in a separate terminal window.
    - It requires your password to access `powermetrics`.
4.  **Run Orchestrator** (in a new terminal window):
    ```bash
    ./scripts/orchestrator.sh
    ```
    - It will detect macOS and collect metrics from the running `host_metrics.py`.
5.  **Output**: CSV files will be generated in `output/`. Rename this directory to `output_mac` to avoid confusion when merging.

---

## 3. Merge Datasets

Once you have the output directories from both machines on one computer:

1.  Run the merge script:
    ```bash
    python3 scripts/merge_datasets.py ./output_windows ./output_mac -o final_dataset.csv
    ```

2.  **Result**: `final_dataset.csv` will contain the combined metrics from both environments, with `hostname`, `cpu_name`, `gpu_name`, and `npu_name` columns distinguishing the data points.

---

---

## 4. Azure VM Integration (Optional)

For deploying dataset collection on Azure VMs in addition to local machines, see the comprehensive [README-AZURE.md](README-AZURE.md) guide.

### Quick Azure Workflow

**1. Deploy Azure VMs:**
```bash
cd azure
# Edit parameters.json with your SSH public key
./deploy.sh
```

**2. Configure remote hosts:**
```bash
# Edit scripts/remote_hosts.txt
nano scripts/remote_hosts.txt

# Add your VMs:
# azure-cpu-01,azureuser,20.123.45.67,Azure D4s_v3 CPU VM
# azure-gpu-01,azureuser,20.123.45.68,Azure NC6 GPU VM
```

**3. Trigger collection on all machines** (local + Azure):
```bash
./scripts/collect_remote.sh
```

**4. Download datasets from Azure VMs:**
```bash
./scripts/download_datasets.sh
```

**5. Merge all datasets** (Windows + macOS + Azure):
```bash
python3 scripts/merge_datasets.py \
    datasets/azure-cpu-01 \
    datasets/azure-gpu-01 \
    output_windows \
    output_mac \
    -o final_dataset.csv
```

### Example: 4-Machine Setup

| Machine | Platform | GPU | HOSTNAME_LABEL |
|---------|----------|-----|----------------|
| Local Windows | WSL2 | RTX 3060 | windows-rtx3060 |
| Local macOS | M4 Pro | Apple GPU | macos-m4-pro |
| Azure CPU VM | Ubuntu 22.04 | None | azure-cpu-01 |
| Azure GPU VM | Ubuntu 22.04 | Tesla K80 | azure-gpu-01 |

**Final dataset**: 4 hardware configurations, comprehensive metrics across CPU/GPU/NPU workloads.

See [README-AZURE.md](README-AZURE.md) for complete deployment guide, troubleshooting, and cost optimization strategies.

---

## Troubleshooting

-   **"powermetrics not found"**: Ensure you are on macOS and using `sudo`.
-   **No GPU metrics on Windows**: Check `docker-compose.gpu.yml` logs (`docker compose -f docker-compose.gpu.yml logs`). Ensure NVIDIA Container Toolkit is set up.
-   **Empty CSVs**: Check Prometheus availability at `http://localhost:9090` during the run.
-   **Azure SSH connection fails**: Verify VM is running (`az vm list -g dataset-creator-rg`), check NSG rules allow port 22, verify SSH key is configured correctly.
-   **Remote collection hangs**: Check if orchestrator is already running on the remote host, ensure sufficient disk space on VMs.
