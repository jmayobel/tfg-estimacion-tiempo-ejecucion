#!/usr/bin/env python3
"""
Dataset Creator - Apple Silicon Host Metrics Exporter
Runs on macOS host (NOT in Docker) to expose GPU, ANE (NPU),
and thermal metrics as a Prometheus-compatible HTTP endpoint.

Usage:
    sudo python3 host_metrics.py [--port 9101] [--interval 2]

Requires sudo for powermetrics access.
"""

import argparse
import json
import logging
import os
import platform
import re
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# Helper Functions
# ============================================================
def run_command_safe(cmd, timeout=5):
    """
    Execute a subprocess command safely with consistent error handling.

    Args:
        cmd: Command and arguments as list
        timeout: Timeout in seconds (default: 5)

    Returns:
        str: stdout as string if successful, None on failure
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        if result.returncode == 0:
            return result.stdout.strip()
        logger.debug(f"Command returned non-zero exit code {result.returncode}: {' '.join(cmd)}")
        return None
    except subprocess.TimeoutExpired:
        logger.debug(f"Command timeout after {timeout}s: {' '.join(cmd)}")
        return None
    except FileNotFoundError:
        logger.debug(f"Command not found: {cmd[0]}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error running command {' '.join(cmd)}: {e}")
        return None


def size_to_mb(size_str):
    """Convert a top(1) size string like '3500M' or '12G' to MB."""
    num = float(re.findall(r"[\d.]+", size_str)[0])
    return num * 1024.0 if "G" in size_str.upper() else num


def extract_metric(text, patterns, parse_func=float):
    """
    Try multiple regex patterns and return the first match.

    Args:
        text: Text to search in
        patterns: List of regex patterns to try
        parse_func: Function to parse matched value (default: float)

    Returns:
        Parsed value if match found, None otherwise
    """
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return parse_func(match.group(1))
            except (ValueError, IndexError) as e:
                logger.debug(f"Error parsing matched value with pattern {pattern}: {e}")
                continue
    return None


# ============================================================
# Constants
# ============================================================

# Memory and power constants
DEFAULT_TOTAL_MEMORY_MB = 24576.0  # M4 chip default (24GB)
BYTES_TO_MB = 1024 * 1024

# Apple Neural Engine (ANE) power limit, used to scale power draw to a
# utilization percentage.
MAX_ANE_POWER_MW = 8000.0
MAX_ANE_POWER_W = MAX_ANE_POWER_MW / 1000.0

# Memory utilization estimation ratios for unified memory
# These are conservative estimates based on typical usage patterns
GPU_MEMORY_UTILIZATION_RATIO = 0.3  # Estimated GPU memory as % of total (30%)
ANE_MEMORY_UTILIZATION_RATIO = 0.1  # Estimated ANE memory as % of total (10%)

# Default server configuration
DEFAULT_PORT = 9101
DEFAULT_COLLECTION_INTERVAL_SECONDS = 2


# ============================================================
# Global metrics store
# ============================================================
metrics = {
    "apple_gpu_utilization": 0.0,
    "apple_gpu_memory_used_mb": 0.0,
    "apple_gpu_memory_total_mb": 0.0,
    "apple_gpu_temperature_celsius": 0.0,
    "apple_gpu_clock_mhz": 0.0,
    "apple_ane_utilization": 0.0,
    "apple_ane_memory_used_mb": 0.0,
    "apple_ane_memory_total_mb": 0.0,
    "apple_ane_temperature_celsius": 0.0,
    "apple_cpu_temperature_celsius": 0.0,
    "apple_host_cpu_usage": 0.0,
    "apple_host_memory_usage_mb": 0.0,
}

metrics_lock = threading.Lock()


# ============================================================
# System Info
# ============================================================
def get_total_memory_mb():
    """
    Get total system memory in MB using sysctl.

    Returns:
        float: Total system memory in megabytes. Falls back to
            DEFAULT_TOTAL_MEMORY_MB if detection fails.
    """
    # Try direct value queries first
    for key in ["hw.memsize", "hw.physmem"]:
        output = run_command_safe(["sysctl", "-n", key])
        if output and output.isdigit():
            return int(output) / BYTES_TO_MB

    # Fallback: parse key-value format
    output = run_command_safe(["sysctl", "hw.memsize"])
    if output:
        # Output: "hw.memsize: 25769803776"
        try:
            parts = output.split(":")
            if len(parts) == 2:
                return int(parts[1].strip()) / BYTES_TO_MB
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing sysctl hw.memsize output: {e}")

    logger.info(
        "Could not detect memory size, using default %.0fGB",
        DEFAULT_TOTAL_MEMORY_MB / 1024,
    )
    return DEFAULT_TOTAL_MEMORY_MB


def get_gpu_info():
    """
    Get GPU name from system_profiler.

    Returns:
        str: GPU model name (e.g., "Apple M4 Pro"). Falls back to "Apple GPU" if detection fails.
    """
    output = run_command_safe(
        ["system_profiler", "SPDisplaysDataType", "-json"],
        timeout=10,
    )
    if output:
        try:
            data = json.loads(output)
            displays = data.get("SPDisplaysDataType", [])
            if displays:
                return displays[0].get("sppci_model", "Apple GPU")
        except json.JSONDecodeError as e:
            logger.warning(f"Error parsing system_profiler JSON output: {e}")
        except (KeyError, IndexError) as e:
            logger.debug(f"GPU info not found in system_profiler output: {e}")

    logger.info("Using default GPU name: Apple GPU")
    return "Apple GPU"


# ============================================================
# Metrics Collection via powermetrics (text output)
# ============================================================
def collect_powermetrics_text(interval):
    """
    Run powermetrics with text output and parse GPU/ANE/thermal data.
    This is more reliable than plist format for M-series chips.
    Requires sudo.

    Args:
        interval (int): Sampling interval in seconds

    Note:
        Runs in infinite loop. Updates global metrics dict with collected data.
        Requires sudo/root privileges to execute powermetrics.
    """
    total_mem = get_total_memory_mb()

    cmd_base = [
        "powermetrics",
        "--samplers",
        "gpu_power,thermal,ane_power",
        "-i",
        str(interval * 1000),
        "-n",
        "1",
    ]

    while True:
        try:
            # Run single-shot powermetrics
            # This takes 'interval' seconds to complete
            result = subprocess.run(cmd_base, capture_output=True, text=True, check=False)

            if result.returncode != 0:
                print(f"ERROR: powermetrics failed: {result.stderr}")
                time.sleep(1)
                continue

            lines = result.stdout.splitlines()
            if lines:
                parse_sample(lines, total_mem)

        except FileNotFoundError:
            print("ERROR: powermetrics not found.")
            break
        except Exception as e:
            print(f"ERROR: Exception in polling loop: {e}")
            time.sleep(1)


def parse_sample(lines, total_mem):
    """
    Parse a single powermetrics text sample.

    Args:
        lines (list): Lines of powermetrics text output
        total_mem (float): Total system memory in MB for memory estimation

    Note:
        Updates global metrics dict with parsed GPU, ANE, and thermal data.
        Uses regex patterns to extract metrics from various powermetrics output formats.
    """
    new_metrics = {}
    text = "\n".join(lines)

    # ----- GPU Utilization -----
    # M4 format: "GPU HW active residency:  12.67%"
    # Other formats: "GPU Active: 42%", "GPU Busy: 15%"
    gpu_util = extract_metric(
        text,
        [
            r"GPU\s+HW\s+active\s+residency\s*:\s*([\d.]+)\s*%",
            r"GPU\s+[Aa]ctive\s*(?:residency)?\s*[:=]\s*([\d.]+)\s*%",
            r"GPU\s+[Bb]usy\s*[:=]\s*([\d.]+)\s*%",
        ],
    )
    if gpu_util is not None:
        new_metrics["apple_gpu_utilization"] = gpu_util

    # ----- GPU Idle residency (fallback to compute utilization) -----
    # M4 format: "GPU idle residency:  87.33%"
    if "apple_gpu_utilization" not in new_metrics:
        idle = extract_metric(text, [r"GPU\s+idle\s+residency\s*:\s*([\d.]+)\s*%"])
        if idle is not None:
            new_metrics["apple_gpu_utilization"] = max(0.0, 100.0 - idle)

    # ----- GPU Frequency -----
    # M4 format: "GPU HW active frequency: 338 MHz"
    gpu_freq = extract_metric(
        text,
        [
            r"GPU\s+HW\s+active\s+frequency\s*:\s*(\d+)\s*MHz",
            r"GPU\s+(?:HW\s+)?[Aa]ctive\s+[Ff]req(?:uency)?\s*[:=]\s*(\d+)\s*MHz",
            r"GPU\s+[Ff]req(?:uency)?\s*[:=]\s*(\d+)\s*MHz",
        ],
    )
    if gpu_freq is not None:
        new_metrics["apple_gpu_clock_mhz"] = gpu_freq

    # ----- ANE (Neural Engine) Power -----
    # "ANE Power: 500 mW" or "ane_power: 0.5 W"; scaled to a utilization
    # percentage against the chip's max ANE power.
    ane_power_mw = extract_metric(text, [r"ANE\s+[Pp]ower\s*[:=]\s*([\d.]+)\s*mW"])
    if ane_power_mw is None:
        ane_power_w = extract_metric(text, [r"ANE\s+[Pp]ower\s*[:=]\s*([\d.]+)\s*W"])
        if ane_power_w is not None:
            ane_power_mw = ane_power_w * 1000.0
    if ane_power_mw is not None:
        new_metrics["apple_ane_utilization"] = min(100.0, (ane_power_mw / MAX_ANE_POWER_MW) * 100)

    # ----- Thermal / Temperature -----
    # SMC sampler outputs lines like:
    #   CPU die temperature: 45.30 C
    #   GPU die temperature: 42.10 C
    #   SOC Die Temp Avg: 40.5 C
    # Or SMC keys like:
    #   Tp09 (CPU efficiency core 1): 38.13 C
    #   Tg0p (GPU): 35.50 C

    # Direct die temperature readings
    cpu_temp = extract_metric(
        text,
        [
            r"CPU\s+die\s+temp(?:erature)?\s*:\s*([\d.]+)\s*C",
            r"CPU\s+temp(?:erature)?\s*:\s*([\d.]+)\s*C",
            r"Tp\d+\s*\(CPU[^)]*\)\s*:\s*([\d.]+)\s*C",
        ],
    )
    if cpu_temp is not None:
        new_metrics["apple_cpu_temperature_celsius"] = cpu_temp

    gpu_temp = extract_metric(
        text,
        [
            r"GPU\s+die\s+temp(?:erature)?\s*:\s*([\d.]+)\s*C",
            r"GPU\s+temp(?:erature)?\s*:\s*([\d.]+)\s*C",
            r"Tg\d+\w*\s*\(GPU[^)]*\)\s*:\s*([\d.]+)\s*C",
        ],
    )
    if gpu_temp is not None:
        new_metrics["apple_gpu_temperature_celsius"] = gpu_temp

    # ANE temperature
    ane_temp = extract_metric(
        text,
        [
            r"ANE\s+die\s+temp(?:erature)?\s*:\s*([\d.]+)\s*C",
            r"ANE\s+temp(?:erature)?\s*:\s*([\d.]+)\s*C",
            r"Ta\d+\w*\s*\(ANE[^)]*\)\s*:\s*([\d.]+)\s*C",
        ],
    )
    if ane_temp is not None:
        new_metrics["apple_ane_temperature_celsius"] = ane_temp

    # SOC temperature as fallback for CPU and GPU
    soc_temp = extract_metric(
        text,
        [r"SOC\s+(?:Die\s+)?[Tt]emp(?:erature)?\s*(?:Avg)?\s*[:=]\s*([\d.]+)"],
    )
    if soc_temp is not None:
        if "apple_cpu_temperature_celsius" not in new_metrics:
            new_metrics["apple_cpu_temperature_celsius"] = soc_temp
        if "apple_gpu_temperature_celsius" not in new_metrics:
            new_metrics["apple_gpu_temperature_celsius"] = soc_temp

    # Generic SMC temperature fallback: find any temperature reading
    if "apple_cpu_temperature_celsius" not in new_metrics:
        fallback_temp = extract_metric(
            text,
            [r"die\s+temp(?:erature)?\s*(?:\d)?\s*:\s*([\d.]+)\s*C"],
        )
        if fallback_temp is not None:
            new_metrics["apple_cpu_temperature_celsius"] = fallback_temp

    # ----- Memory estimates (unified memory) -----
    gpu_util = new_metrics.get("apple_gpu_utilization", 0)
    new_metrics["apple_gpu_memory_total_mb"] = total_mem
    new_metrics["apple_gpu_memory_used_mb"] = (
        (gpu_util / 100.0) * total_mem * GPU_MEMORY_UTILIZATION_RATIO
    )

    ane_util = new_metrics.get("apple_ane_utilization", 0)
    new_metrics["apple_ane_memory_total_mb"] = total_mem
    new_metrics["apple_ane_memory_used_mb"] = (
        (ane_util / 100.0) * total_mem * ANE_MEMORY_UTILIZATION_RATIO
    )

    with metrics_lock:
        metrics.update(new_metrics)

    # Debug print
    debug_vals = {k: v for k, v in new_metrics.items() if v != 0 and v != total_mem}
    if debug_vals:
        print(f"  Updated: {debug_vals}")


# ============================================================
# Fallback: ioreg for thermals (no sudo needed)
# ============================================================
def collect_ioreg_thermals():
    """
    Collect thermal data via ioreg as fallback (no sudo needed).

    Note:
        Runs in infinite loop. Updates global metrics dict with CPU temperature.
        Uses ioreg to read AppleSMCTemperatureSensor. Does not require root privileges.
        Falls back method when powermetrics is not available.
    """
    while True:
        # Try to get SMC temperature sensors
        output = run_command_safe(["ioreg", "-rc", "AppleSMCTemperatureSensor"])
        if output:
            temps = re.findall(r'"CurrentValue"\s*=\s*(\d+)', output)
            if temps:
                try:
                    # First sensor is usually CPU proximity
                    cpu_temp = int(temps[0]) / 100.0 if int(temps[0]) > 100 else float(temps[0])
                    with metrics_lock:
                        if metrics.get("apple_cpu_temperature_celsius") is None:
                            metrics["apple_cpu_temperature_celsius"] = cpu_temp
                except (ValueError, IndexError) as e:
                    logger.debug(f"Error parsing ioreg thermal value: {e}")

        # Also try iStats gem if available
        output = run_command_safe(["istats", "cpu", "temp", "--value-only"])
        if output:
            try:
                temp = float(output)
                if temp > 0:
                    with metrics_lock:
                        metrics["apple_cpu_temperature_celsius"] = temp
            except ValueError as e:
                logger.debug(f"Error parsing iStats output: {e}")

        time.sleep(5)


# ============================================================
# Fallback: GPU utilization via IOKit
# ============================================================
def collect_iokit_gpu():
    """
    Try to get GPU utilization via ioreg IOAccelerator (no sudo needed).

    Note:
        Runs in infinite loop. Updates global metrics dict with GPU utilization.
        Uses ioreg to read IOAccelerator device properties. Does not require root privileges.
        Fallback method when powermetrics is not available.
    """
    while True:
        output = run_command_safe(["ioreg", "-rc", "IOAccelerator"])
        if output:
            # Look for GPU utilization
            match = re.search(
                r'"Device Utilization %"\s*=\s*(\d+)',
                output,
            )
            if match:
                try:
                    gpu_util = float(match.group(1))
                    with metrics_lock:
                        if (
                            metrics.get("apple_gpu_utilization") is None
                            or metrics.get("apple_gpu_utilization", 0) == 0
                        ):
                            metrics["apple_gpu_utilization"] = gpu_util
                except ValueError as e:
                    logger.debug(f"Error parsing GPU utilization: {e}")

            # Look for performance statistics
            match = re.search(r'"GPU Activity\(%\)"\s*=\s*(\d+)', output)
            if match:
                try:
                    gpu_util = float(match.group(1))
                    with metrics_lock:
                        metrics["apple_gpu_utilization"] = gpu_util
                except ValueError as e:
                    logger.debug(f"Error parsing GPU activity: {e}")

        time.sleep(2)


# ============================================================
# Host stats collection (CPU / Memory)
# ============================================================
def collect_host_stats(interval):
    """
    Collect host-level CPU and memory metrics on macOS using top.
    """
    while True:
        output = run_command_safe(["top", "-l", "1", "-n", "0"])
        if output:
            lines = output.splitlines()
            cpu_usage = 0.0
            mem_used_mb = 0.0

            for line in lines:
                if "CPU usage:" in line:
                    match = re.search(
                        r"CPU\s+usage\s*:\s*([\d.]+)%\s*user\s*,\s*([\d.]+)%\s*sys",
                        line,
                        re.IGNORECASE,
                    )
                    if match:
                        try:
                            user = float(match.group(1))
                            sys_val = float(match.group(2))
                            cpu_usage = user + sys_val
                        except ValueError:
                            pass
                elif "PhysMem:" in line:
                    match = re.search(
                        r"PhysMem\s*:\s*([\d.]+[GM])\s+used\s*\(?.*?\)?,\s*([\d.]+[GM])\s+unused",
                        line,
                        re.IGNORECASE,
                    )
                    if match:
                        try:
                            mem_used_mb = size_to_mb(match.group(1))
                        except Exception as e:
                            logger.debug(f"Error parsing PhysMem: {e}")

            with metrics_lock:
                metrics["apple_host_cpu_usage"] = cpu_usage
                metrics["apple_host_memory_usage_mb"] = mem_used_mb

        time.sleep(interval)


# ============================================================
# HTTP Server (Prometheus text format)
# ============================================================
class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            with metrics_lock:
                output = []
                for name, value in metrics.items():
                    output.append(f"# HELP {name} Apple Silicon metric")
                    output.append(f"# TYPE {name} gauge")
                    output.append(f"{name} {value}")

                body = "\n".join(output) + "\n"

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))

        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK\n")

        elif self.path == "/debug":
            # Debug endpoint to see raw powermetrics output
            with metrics_lock:
                body = json.dumps(metrics, indent=2)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Apple Silicon metrics exporter for Prometheus")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"HTTP port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_COLLECTION_INTERVAL_SECONDS,
        help=(
            "Collection interval in seconds " f"(default: {DEFAULT_COLLECTION_INTERVAL_SECONDS})"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw powermetrics output for debugging",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (debug) logging",
    )
    args = parser.parse_args()

    # Update logging level if verbose flag is set
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    if platform.system() != "Darwin":
        logger.error("This script is only for macOS with Apple Silicon")
        print("ERROR: This script is only for macOS with Apple Silicon.")
        sys.exit(1)

    is_root = os.geteuid() == 0

    gpu_name = get_gpu_info()
    total_mem = get_total_memory_mb()
    print("Apple Silicon Host Metrics Exporter")
    print(f"  GPU: {gpu_name}")
    print(f"  Total Memory: {total_mem:.0f} MB")
    print(f"  Port: {args.port}")
    print(f"  Interval: {args.interval}s")
    print(f"  Running as root: {is_root}")

    if is_root:
        # Start powermetrics collection (primary, best data)
        pm_thread = threading.Thread(
            target=collect_powermetrics_text,
            args=(args.interval,),
            daemon=True,
        )
        pm_thread.start()
        print("  Started: powermetrics collector (sudo)")
    else:
        print("  WARNING: Not running as root. powermetrics unavailable.")
        print("           GPU/ANE utilization will use ioreg fallback (less accurate).")
        print("           Run with: sudo python3 host_metrics.py")

    # Start fallback collectors (always, as backup)
    ioreg_thread = threading.Thread(target=collect_ioreg_thermals, daemon=True)
    ioreg_thread.start()
    print("  Started: ioreg thermal collector")

    iokit_thread = threading.Thread(target=collect_iokit_gpu, daemon=True)
    iokit_thread.start()
    print("  Started: IOKit GPU collector")

    # Start host stats collector
    host_stats_thread = threading.Thread(
        target=collect_host_stats,
        args=(args.interval,),
        daemon=True,
    )
    host_stats_thread.start()
    print("  Started: host stats collector (CPU/Memory)")

    # Start HTTP server
    server = HTTPServer(("0.0.0.0", args.port), MetricsHandler)
    print(f"\nServing metrics at http://0.0.0.0:{args.port}/metrics")
    print(f"Debug endpoint at http://0.0.0.0:{args.port}/debug")
    print("Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
        server.shutdown()


if __name__ == "__main__":
    main()
