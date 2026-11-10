#!/usr/bin/env python3
"""
Dataset Creator - CSV Exporter
Queries Prometheus HTTP API and exports metrics to CSV matching the target schema.
"""

import argparse
import logging
import os
import subprocess
from datetime import datetime, timezone

import requests
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# Rate window: must be >= 2x scrape_interval for reliable results.
# With 2s scrape, we use 10s to guarantee at least 4-5 data points.
# ============================================================
RATE_WINDOW = "10s"


# ============================================================
# Prometheus Metric Queries
# ============================================================
# For container metrics, we try multiple label selectors because
# cAdvisor uses different labels across platforms:
#   - Docker Desktop (Mac): often uses "name" label
#   - Linux/WSL2: may use "name", "container_label_com_docker_compose_service", etc.
#
# Each query is a template with {container}, {service_key} and {container_id}
# placeholders; query_with_fallbacks() expands them and tries the queries in
# order until one returns data.

# Per-container label selectors, in fallback order.
NAME_SELECTOR = 'name="{container}"'
COMPOSE_SELECTOR = 'container_label_com_docker_compose_service="{service_key}"'
ID_SELECTOR = 'id=~".*{container_id}.*"'
CONTAINER_SELECTORS = [NAME_SELECTOR, COMPOSE_SELECTOR, ID_SELECTOR]


def _rate_query(metric, selector, suffix=""):
    """Build a summed PromQL rate() query template for one label selector."""
    return "sum(rate(" + metric + "{{" + selector + "}}[" + RATE_WINDOW + "]))" + suffix


def _rate_queries(metric, suffix=""):
    """The same rate query over every per-container selector, in fallback order."""
    return [_rate_query(metric, sel, suffix) for sel in CONTAINER_SELECTORS]


# Host-level fallback for network metrics: on Docker Desktop/WSL2, per-container
# network stats are unavailable. Use root cgroup (id="/") eth0 as approximation.
HOST_NETWORK_SELECTOR = 'id="/",interface="eth0"'

CONTAINER_METRICS = {
    "cpu_usage": {
        "queries": _rate_queries("container_cpu_usage_seconds_total", " * 100")
        + ["apple_host_cpu_usage"],
        "description": "CPU usage percentage",
    },
    "memory_usage": {
        "queries": [
            "container_memory_usage_bytes{{" + NAME_SELECTOR + "}} / 1048576",
            "container_memory_usage_bytes{{" + COMPOSE_SELECTOR + "}} / 1048576",
            "container_memory_working_set_bytes{{" + NAME_SELECTOR + "}} / 1048576",
            "container_memory_usage_bytes{{" + ID_SELECTOR + "}} / 1048576",
            "apple_host_memory_usage_mb",
        ],
        "description": "Container memory usage in MB",
    },
    "network_receive": {
        "queries": _rate_queries("container_network_receive_bytes_total", " / 1048576")
        + [
            _rate_query(
                "container_network_receive_bytes_total", HOST_NETWORK_SELECTOR, " / 1048576"
            )
        ],
        "description": "Network receive rate in MB/s",
    },
    "network_transmit": {
        "queries": _rate_queries("container_network_transmit_bytes_total", " / 1048576")
        + [
            _rate_query(
                "container_network_transmit_bytes_total", HOST_NETWORK_SELECTOR, " / 1048576"
            )
        ],
        "description": "Network transmit rate in MB/s",
    },
    "disk_read": {
        "queries": [
            _rate_query("container_fs_reads_bytes_total", NAME_SELECTOR, " / 1048576"),
            _rate_query("container_fs_read_seconds_total", NAME_SELECTOR, " * 100"),
            _rate_query(
                "container_blkio_device_usage_total",
                NAME_SELECTOR + ', operation="Read"',
                " / 1048576",
            ),
            _rate_query("container_fs_reads_bytes_total", ID_SELECTOR, " / 1048576"),
        ],
        "description": "Disk read rate in MB/s",
    },
    "disk_write": {
        "queries": [
            _rate_query("container_fs_writes_bytes_total", NAME_SELECTOR, " / 1048576"),
            _rate_query("container_fs_write_seconds_total", NAME_SELECTOR, " * 100"),
            _rate_query(
                "container_blkio_device_usage_total",
                NAME_SELECTOR + ', operation="Write"',
                " / 1048576",
            ),
            _rate_query("container_fs_writes_bytes_total", ID_SELECTOR, " / 1048576"),
        ],
        "description": "Disk write rate in MB/s",
    },
}

HOST_METRICS = {
    "cpu_cores": {
        "queries": [
            "count(node_cpu_seconds_total{{mode='idle'}})",
            "machine_cpu_cores",
        ],
        "description": "Number of CPU cores",
    },
    "cpu_load_avg_1s": {
        "queries": ["node_load1"],
        "description": "1-minute load average",
    },
    "memory_avail": {
        "queries": [
            "node_memory_MemTotal_bytes / 1048576",
            "machine_memory_bytes / 1048576",
        ],
        "description": "Total available memory in MB",
    },
    "memory_swap": {
        "queries": [
            "node_memory_SwapTotal_bytes / 1048576",
        ],
        "description": "Total swap in MB",
    },
    "cpu_temp": {
        "queries": [
            'node_hwmon_temp_celsius{{chip=~".*coretemp.*"}}',
            "node_hwmon_temp_celsius",
            "node_thermal_zone_temp",
        ],
        "description": "CPU temperature in Celsius",
    },
}

NVIDIA_GPU_METRICS = {
    "gpu_usage": {
        "queries": ["DCGM_FI_DEV_GPU_UTIL"],
        "description": "GPU utilization percentage",
    },
    "gpu_memory_usage": {
        "queries": ["DCGM_FI_DEV_FB_USED"],
        "description": "GPU memory used in MB",
    },
    "gpu_memory_total": {
        "queries": ["DCGM_FI_DEV_FB_FREE + DCGM_FI_DEV_FB_USED"],
        "description": "GPU total memory in MB",
    },
    "gpu_temp": {
        "queries": ["DCGM_FI_DEV_GPU_TEMP"],
        "description": "GPU temperature in Celsius",
    },
    "gpu_clock_value": {
        "queries": ["DCGM_FI_DEV_SM_CLOCK"],
        "description": "GPU SM clock in MHz",
    },
    "gpu_power_usage": {
        "queries": ["DCGM_FI_DEV_POWER_USAGE"],
        "description": "GPU power usage in W",
    },
}

APPLE_SILICON_METRICS = {
    "gpu_usage": {
        "queries": ["apple_gpu_utilization"],
        "description": "Apple GPU utilization percentage",
    },
    "gpu_memory_usage": {
        "queries": ["apple_gpu_memory_used_mb"],
        "description": "Apple GPU memory used in MB",
    },
    "gpu_memory_total": {
        "queries": ["apple_gpu_memory_total_mb"],
        "description": "Apple GPU total memory in MB",
    },
    "gpu_temp": {
        "queries": ["apple_gpu_temperature_celsius"],
        "description": "Apple GPU temperature in Celsius",
    },
    "gpu_clock_value": {
        "queries": ["apple_gpu_clock_mhz"],
        "description": "Apple GPU clock in MHz",
    },
    "npu_usage": {
        "queries": ["apple_ane_utilization"],
        "description": "Apple Neural Engine utilization percentage",
    },
    "npu_memory_usage": {
        "queries": ["apple_ane_memory_used_mb"],
        "description": "Apple ANE memory used in MB",
    },
    "npu_memory_total": {
        "queries": ["apple_ane_memory_total_mb"],
        "description": "Apple ANE total memory in MB",
    },
    "npu_temp": {
        "queries": ["apple_ane_temperature_celsius"],
        "description": "Apple ANE temperature in Celsius",
    },
    "cpu_temp": {
        "queries": ["apple_cpu_temperature_celsius"],
        "description": "Apple CPU temperature in Celsius",
    },
}


# ============================================================
# Target CSV Schema
# ============================================================
CSV_COLUMNS = [
    "timestamp",
    "cpu_usage",
    "cpu_cores",
    "cpu_load_avg_1s",
    "memory_avail",
    "memory_usage",
    "service_type",
    "service_name",
    "architecture_type",
    "gpu_usage",
    "gpu_memory_usage",
    "gpu_memory_total",
    "npu_usage",
    "npu_name",
    "npu_memory_usage",
    "npu_memory_total",
    "gpu_name",
    "cpu_name",
    "memory_swap",
    "network_receive",
    "network_transmit",
    "hostname",
    "service_size",
    "gpu_clock_value",
    "cpu_clock_value",
    "num_param",
    "workload_value",
    "workload_unit",
    "gpu_power_usage",
    "disk_read",
    "disk_write",
    "cpu_temp",
    "gpu_temp",
    "npu_temp",
    "start_time",
    "end_time",
    "execution_time",
    "elapsed_seconds",
    "remaining_seconds",
]


def query_prometheus_range(prom_url, query, start, end, step):
    """
    Query Prometheus for metric values over a time range.

    Args:
        prom_url (str): Base URL of Prometheus server (e.g., 'http://localhost:9090')
        query (str): PromQL query string
        start (int): Start timestamp (Unix epoch seconds)
        end (int): End timestamp (Unix epoch seconds)
        step (int): Query resolution in seconds

    Returns:
        list: List of (timestamp, value) tuples as floats, ordered by timestamp.
              Returns empty list on query failure.

    Raises:
        None: All exceptions are caught and logged, returns empty list on error
    """
    try:
        resp = requests.get(
            f"{prom_url}/api/v1/query_range",
            params={
                "query": query,
                "start": start,
                "end": end,
                "step": f"{step}s",
            },
            timeout=30,
        )
        resp.raise_for_status()

        # Validate JSON response
        try:
            data = resp.json()
        except ValueError as e:
            logger.error(f"Invalid JSON from Prometheus: {e}")
            return []

        if not isinstance(data, dict):
            logger.error(f"Prometheus response is not a JSON object for query: {query}")
            return []

        if data.get("status") != "success":
            error_msg = data.get("error", "unknown error")
            logger.warning(f"Prometheus query failed: {error_msg}, query: {query}")
            return []

        results = data.get("data", {}).get("result", [])
        if not results:
            logger.debug(f"No results for query: {query}")
            return []

        # Take the first result series (queries are sum()-aggregated, so at
        # most one series is expected).
        values = results[0].get("values", [])
        parsed = [(float(ts), float(val)) for ts, val in values]
        return parsed

    except requests.Timeout:
        logger.error(f"Prometheus query timeout after 30s: {query}")
        return []
    except requests.RequestException as e:
        logger.error(f"Prometheus request failed: {e}, query: {query}")
        return []
    except (KeyError, IndexError, ValueError) as e:
        logger.error(f"Error parsing Prometheus response: {e}, query: {query}")
        return []
    except Exception:
        logger.exception(f"Unexpected error querying Prometheus: {query}")
        return []


def query_with_fallbacks(
    prom_url,
    queries,
    start,
    end,
    step,
    container_name,
    service_key,
    container_id="",
):
    """
    Try multiple PromQL queries with different label selectors, return first successful result.

    This function handles platform differences in cAdvisor label naming by trying
    multiple query patterns until one returns data.

    Args:
        prom_url (str): Base URL of Prometheus server
        queries (list): List of PromQL query templates with {container},
            {service_key}, and {container_id} placeholders
        start (int): Start timestamp (Unix epoch seconds)
        end (int): End timestamp (Unix epoch seconds)
        step (int): Query resolution in seconds
        container_name (str): Container name for label matching
        service_key (str): Service key for label matching
        container_id (str): Docker container ID for id-based matching (WSL2/Docker Desktop)

    Returns:
        tuple: (values, matched_query) where values is list of (timestamp, value) tuples
               and matched_query is the successful query string. Returns ([], None) if all fail.
    """
    for query_template in queries:
        # Skip id-based queries if no container_id was provided
        if "{container_id}" in query_template and not container_id:
            continue
        query = query_template.format(
            container=container_name,
            service_key=service_key,
            container_id=container_id,
        )
        values = query_prometheus_range(prom_url, query, start, end, step)
        if values:
            return values, query
    return [], None


def get_cpu_clock():
    """
    Detect CPU clock frequency from system.

    Tries multiple methods in order:
    1. /proc/cpuinfo (Linux/WSL2) - looks for 'cpu MHz' field
    2. sysctl hw.cpufrequency (macOS) - returns Hz, converts to MHz

    Returns:
        float: CPU frequency in MHz, or None if detection fails
    """
    # Try Linux /proc/cpuinfo first
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "cpu MHz" in line:
                    return float(line.split(":")[1].strip())
    except FileNotFoundError:
        logger.debug("/proc/cpuinfo not found (not on Linux)")
    except (IOError, ValueError) as e:
        logger.warning(f"Error reading CPU clock from /proc/cpuinfo: {e}")
    except Exception as e:
        logger.error(f"Unexpected error reading /proc/cpuinfo: {e}")

    # macOS fallback
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.cpufrequency"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            freq_str = result.stdout.strip()
            # On M1/M2/M3/M4, sysctl output is in Hz (e.g., 2400000000)
            return float(freq_str) / 1000000.0
        else:
            logger.debug(
                "sysctl hw.cpufrequency returned non-zero exit code: %s",
                result.returncode,
            )
    except FileNotFoundError:
        logger.debug("sysctl command not found (not on macOS)")
    except subprocess.TimeoutExpired:
        logger.warning("Timeout getting CPU clock from sysctl")
    except ValueError as e:
        logger.warning(f"Error parsing CPU frequency from sysctl: {e}")
    except Exception as e:
        logger.error(f"Unexpected error running sysctl for CPU clock: {e}")

    logger.info("Could not detect CPU clock speed")
    return None


def collect_timeseries(
    prom_url,
    start_time,
    end_time,
    step,
    container_name,
    service_key,
    service_name,
    service_type,
    architecture,
    hostname,
    service_size,
    gpu_clock,
    cpu_clock,
    num_params,
    has_nvidia_gpu,
    has_apple_npu,
    execution_time,
    workload_value,
    workload_unit,
    gpu_name=None,
    cpu_name=None,
    npu_name=None,
    container_id="",
):
    """
    Collects all metrics for a given time range and returns a structured list of dicts.
    """
    # Convert start/end to unix timestamp if needed
    # (The script receives them as unix timestamps, so we just ensure int/float)
    start_epoch = int(float(start_time))
    end_epoch = int(float(end_time))

    # We'll use a dictionary of {timestamp: {metric: value}} to align data
    data_by_time = {}

    # Initialize timestamps based on step
    # This creates a skeleton to fill
    current = start_epoch
    while current <= end_epoch:
        elapsed_seconds = int(current - start_epoch)
        remaining_seconds = None
        if execution_time is not None:
            remaining_seconds = max(int(execution_time - elapsed_seconds), 0)
        data_by_time[current] = {
            "timestamp": datetime.fromtimestamp(current, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "service_type": service_type,
            "service_name": service_name,
            "architecture_type": architecture,
            "hostname": hostname,
            "service_size": service_size,
            "gpu_name": gpu_name,
            "cpu_name": cpu_name,
            "npu_name": npu_name,
            "gpu_clock_value": gpu_clock,
            "cpu_clock_value": cpu_clock,
            "num_param": num_params,
            "execution_time": execution_time,
            "workload_value": workload_value,
            "workload_unit": workload_unit,
            "start_time": datetime.fromtimestamp(start_epoch, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "end_time": datetime.fromtimestamp(end_epoch, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "elapsed_seconds": elapsed_seconds,
            "remaining_seconds": remaining_seconds,
        }
        current += step

    # Query each metric and merge into data_by_time
    # Combine all metric dictionaries
    ALL_METRICS = {**CONTAINER_METRICS, **HOST_METRICS}

    # Add GPU/NPU metrics if applicable
    # Convert string flags to booleans (argparse passes them as strings)
    nvidia_enabled = str(has_nvidia_gpu).lower() == "true"
    apple_enabled = str(has_apple_npu).lower() == "true"

    if nvidia_enabled:
        ALL_METRICS.update(NVIDIA_GPU_METRICS)
    if apple_enabled:
        # Check if Apple Host Metrics are available (different exporter)
        # We assume they are exposed as expected if the flag is true
        ALL_METRICS.update(APPLE_SILICON_METRICS)

    for metric_name, info in ALL_METRICS.items():
        values, matched_query = query_with_fallbacks(
            prom_url,
            info["queries"],
            start_epoch,
            end_epoch,
            step,
            container_name,
            service_key,
            container_id=container_id,
        )

        if values:
            logger.info(
                "  [OK] %s: %d values via %s",
                metric_name,
                len(values),
                matched_query[:80] if matched_query else "?",
            )
        else:
            logger.debug("  [MISS] %s: no data from any query", metric_name)

        # Merge values into our time-aligned data
        for t, v in values:
            if t in data_by_time:
                data_by_time[t][metric_name] = v

    # Convert to list and sort
    result_list = sorted(data_by_time.values(), key=lambda x: x["timestamp"])
    return result_list


def main():
    parser = argparse.ArgumentParser(description="Export Prometheus metrics to CSV")
    parser.add_argument("--prometheus-url", required=True)
    parser.add_argument(
        "--start-time",
        required=True,
        help="ISO format: YYYY-MM-DD HH:MM:SS",
    )
    parser.add_argument("--end-time", required=True)
    parser.add_argument("--service-key", required=True)
    parser.add_argument("--service-name", required=True)
    parser.add_argument("--service-type", required=True)
    parser.add_argument("--architecture-type", required=True)
    parser.add_argument("--hostname", required=True)
    parser.add_argument("--cpu-name", required=True)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--npu-name", required=True)
    parser.add_argument("--service-size", required=True)
    parser.add_argument("--num-params", required=True)
    parser.add_argument("--execution-time", required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument(
        "--container-id",
        default="",
        help="Docker container ID for id-based cAdvisor queries (WSL2/Docker Desktop)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--has-nvidia-gpu", default="false")
    parser.add_argument("--has-apple-npu", default="false")
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument(
        "--workload-value",
        type=float,
        default=0.0,
        help="Numeric value of the workload",
    )
    parser.add_argument(
        "--workload-unit",
        type=str,
        default="unknown",
        help="Unit of the workload",
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

    print(f"\nExporting metrics for: {args.service_key}")
    print(f"  Time range: {args.start_time} -> {args.end_time}")
    print(f"  Container: {args.container_name}")

    # Get CPU clock if available
    cpu_clock = get_cpu_clock()

    # Strip whitespace/newlines from string args to prevent malformed CSV
    service_name = args.service_name.strip()

    collected_data = collect_timeseries(
        prom_url=args.prometheus_url,
        start_time=args.start_time,
        end_time=args.end_time,
        step=args.step,
        container_name=args.container_name,
        service_key=args.service_key,
        service_name=service_name,
        service_type=args.service_type,
        architecture=args.architecture_type,
        hostname=args.hostname,
        service_size=float(args.service_size) if args.service_size != "0" else None,
        gpu_clock=None,  # populated from Prometheus DCGM/Apple metrics
        cpu_clock=cpu_clock,
        num_params=float(args.num_params) if args.num_params != "0" else None,
        has_nvidia_gpu=args.has_nvidia_gpu,
        has_apple_npu=args.has_apple_npu,
        execution_time=float(args.execution_time),
        workload_value=args.workload_value,
        workload_unit=args.workload_unit,
        gpu_name=args.gpu_name if args.gpu_name != "none" else None,
        cpu_name=args.cpu_name if args.cpu_name != "unknown" else None,
        npu_name=args.npu_name if args.npu_name != "none" else None,
        container_id=args.container_id,
    )

    if not collected_data:
        print("  WARNING: No metrics collected!")
        return

    # Create DataFrame from collected data
    df = pd.DataFrame(collected_data)

    # Ensure all expected columns exist and reorder, even for empty DataFrames
    if not df.empty:
        # Convert timestamps to readable format
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Ensure all expected columns exist
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Reorder columns to match expected schema
    df = df[CSV_COLUMNS]

    # One CSV per service run; the orchestrator merges them afterwards.
    output_path = os.path.join(args.output_dir, f"{args.service_key}.csv")
    df.to_csv(output_path, index=False)
    print(f"  Exported {len(df)} rows to {output_path} ({len(df.columns)} columns)")


if __name__ == "__main__":
    main()
