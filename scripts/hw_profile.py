#!/usr/bin/env python3
"""Capture a structured hardware profile for the current host.

Writes a single hw_profile.json into the target directory. The file
is consumed by build_ml_dataset.py to enrich every (run, benchmark)
sample with hardware features the prediction model can learn from
(instead of memorising hostnames).

Goals:
- Works on Linux, WSL2 and macOS.
- Never raises on missing data; unknown fields become null.
- Pure stdlib + optional shell tools (lscpu, sysctl, nvidia-smi);
  no extra Python deps beyond what the orchestrator already installs.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

# Known base / max-turbo GHz for SKUs whose OS APIs don't expose CPU freq:
#   - Intel datacenter parts inside Hyper-V/KVM (no cpufreq sysfs, no "CPU max MHz" in lscpu).
#   - Apple Silicon (sysctl hw.cpufrequency / hw.cpufrequency_max return nothing on M-series).
# Keys are substrings of cpu_name (matched with `in`). Add an entry when you onboard a new host.
KNOWN_CPU_SPECS: dict[str, dict[str, float]] = {
    # Intel Xeon (Azure)
    "Xeon(R) Platinum 8272CL": {"base_ghz": 2.6, "max_ghz": 3.7},
    "Xeon(R) Platinum 8370C": {"base_ghz": 2.8, "max_ghz": 3.5},  # Ice Lake (Esv3/Dsv-ish)
    # AMD EPYC Milan (Azure DASv4 family in swedencentral)
    "EPYC 7763": {"base_ghz": 2.45, "max_ghz": 3.5},
    # Intel Core, Raptor Lake Refresh
    "i7-14700K": {"base_ghz": 3.4, "max_ghz": 5.6},
    # desktop-i9-rtx4060ti (i9-14900KF): poorly cooled, so the all-core multiplier
    # was LOCKED in BIOS to tame thermal throttling. Once fixed, the clock stops
    # being a hidden confounder and becomes a known spec,
    # so base==max==the measured sustained all-core clock (~4.44 GHz via Windows
    # "% Processor Performance" under load; flat min 4.38 / max 4.48), NOT the
    # 3.2/6.0 GHz nominal turbo. Note: /proc/cpuinfo in WSL reports the static
    # CPUID base (3.19 GHz) and cannot see the locked ratio. Only host with this SKU.
    "i9-14900K": {"base_ghz": 4.4, "max_ghz": 4.4},
    # ARM server CPUs on Azure (fixed clock, no turbo). lscpu reports the core
    # microarchitecture as "Model name". VERIFY on first boot with
    # `python3 scripts/hw_profile.py --print`; adjust the key if it differs.
    #   Dpsv5 = Ampere Altra  -> Neoverse-N1 @ 3.0 GHz
    #   Dpsv6 = Azure Cobalt 100 -> Neoverse-N2 @ ~3.4 GHz
    "Neoverse-N1": {"base_ghz": 3.0, "max_ghz": 3.0},
    "Neoverse-N2": {"base_ghz": 3.4, "max_ghz": 3.4},
    # Apple Silicon (no usable sysctl on M-series — single value for base and max)
    "Apple M1": {"base_ghz": 3.2, "max_ghz": 3.2},
    "Apple M2": {"base_ghz": 3.49, "max_ghz": 3.49},
    "Apple M3": {"base_ghz": 4.05, "max_ghz": 4.05},
    "Apple M4": {"base_ghz": 4.4, "max_ghz": 4.4},
}


# ARM core part numbers (/proc/cpuinfo "CPU part", implementer 0x41 = ARM Ltd).
# Used to name ARM CPUs that expose no "Model name" (e.g. Azure Cobalt 100).
ARM_CPU_PARTS: dict[str, str] = {
    "0xd0c": "Neoverse-N1",  # Ampere Altra (Azure Dpsv5)
    "0xd49": "Neoverse-N2",  # Azure Cobalt 100 (Dpsv6)
    "0xd40": "Neoverse-V1",
    "0xd4f": "Neoverse-V2",  # AWS Graviton4 / Grace
    "0xd0b": "Cortex-A76",
}


def lookup_known_freqs(cpu_name: str | None) -> dict[str, float]:
    """Return {base_ghz, max_ghz} for a known SKU, or {} if unknown."""
    if not cpu_name:
        return {}
    for key, spec in KNOWN_CPU_SPECS.items():
        if key in cpu_name:
            return spec
    return {}


def _read_int_file(path: str) -> int | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _run(cmd: list[str], timeout: float = 5.0) -> str | None:
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _first_match(text: str, pattern: str, group: int = 1) -> str | None:
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(group).strip() if m else None


def detect_os() -> dict:
    system = platform.system().lower()
    release = platform.release()
    is_wsl = False
    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            is_wsl = "microsoft" in f.read().lower()
    except OSError:
        pass

    if system == "darwin":
        flavor = "macos"
    elif is_wsl:
        flavor = "wsl2"
    elif system == "linux":
        flavor = "linux"
    else:
        flavor = system or "unknown"

    return {
        "os_family": flavor,
        "os_release": release,
        "machine": platform.machine().lower(),
    }


def detect_arch(machine: str) -> str:
    m = machine.lower()
    if m in {"x86_64", "amd64"}:
        return "x86_64"
    if m in {"arm64", "aarch64"}:
        return "arm64"
    return m or "unknown"


def detect_cpu_linux() -> dict:
    info: dict = {
        "cpu_name": None,
        "cpu_vendor": None,
        "cpu_cores_physical": None,
        "cpu_cores_logical": None,
        "cpu_base_ghz": None,
        "cpu_max_ghz": None,
        "cpu_l1d_kb": None,
        "cpu_l2_kb": None,
        "cpu_l3_kb": None,
    }

    cpuinfo = ""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            cpuinfo = f.read()
    except OSError:
        pass

    info["cpu_name"] = _first_match(cpuinfo, r"^model name\s*:\s*(.+)$")
    info["cpu_vendor"] = _first_match(cpuinfo, r"^vendor_id\s*:\s*(.+)$")

    try:
        info["cpu_cores_logical"] = os.cpu_count()
    except Exception:
        info["cpu_cores_logical"] = None

    lscpu = _run(["lscpu"]) or ""
    if lscpu:
        # Physical cores = sockets * cores_per_socket
        sockets = _first_match(lscpu, r"^Socket\(s\):\s*(\d+)")
        cps = _first_match(lscpu, r"^Core\(s\) per socket:\s*(\d+)")
        if sockets and cps:
            try:
                info["cpu_cores_physical"] = int(sockets) * int(cps)
            except ValueError:
                pass

        # On ARM (aarch64), /proc/cpuinfo has no "model name"/"vendor_id"
        # fields — lscpu derives them (e.g. Model name: Neoverse-N1,
        # Vendor ID: ARM). Fill from lscpu only when /proc/cpuinfo didn't.
        # Harmless on x86, where both are already populated above.
        if info["cpu_name"] is None:
            info["cpu_name"] = _first_match(lscpu, r"^Model name:\s*(.+)$")
        if info["cpu_vendor"] is None:
            info["cpu_vendor"] = _first_match(lscpu, r"^Vendor ID:\s*(.+)$")

        # Frequencies; lscpu reports MHz, sometimes only "CPU MHz"
        max_mhz = _first_match(lscpu, r"^CPU max MHz:\s*([\d.]+)") or _first_match(
            lscpu, r"^CPU MHz:\s*([\d.]+)"
        )
        if max_mhz:
            info["cpu_max_ghz"] = round(float(max_mhz) / 1000.0, 3)

        for key, label in (
            ("cpu_l1d_kb", "L1d cache"),
            ("cpu_l2_kb", "L2 cache"),
            ("cpu_l3_kb", "L3 cache"),
        ):
            raw = _first_match(lscpu, rf"^{re.escape(label)}:\s*(.+)$")
            if raw:
                info[key] = _parse_cache_kb(raw)

    # Some ARM hosts (e.g. Azure Cobalt 100 / Dpsv6) report neither a
    # "Model name" in lscpu nor a "model name" in /proc/cpuinfo — lscpu just
    # shows "Model: 0". Derive the core name from the ARM implementer + part
    # number in /proc/cpuinfo so KNOWN_CPU_SPECS can still match.
    if info["cpu_name"] is None:
        part = _first_match(cpuinfo, r"^CPU part\s*:\s*(0x[0-9a-fA-F]+)")
        if part:
            info["cpu_name"] = ARM_CPU_PARTS.get(part.lower())

    # Base GHz often appears inside the cpu_name (e.g. "i7-14700K @ 3.40GHz")
    if info["cpu_name"]:
        base = _first_match(info["cpu_name"], r"@\s*([\d.]+)\s*GHz")
        if base:
            try:
                info["cpu_base_ghz"] = float(base)
            except ValueError:
                pass

    # cpufreq sysfs: present on bare-metal Linux with modern Intel/AMD CPUs.
    # Absent in most VMs/WSL2 — that's expected; we'll fall through to SKU lookup.
    if info["cpu_base_ghz"] is None:
        khz = _read_int_file("/sys/devices/system/cpu/cpu0/cpufreq/base_frequency")
        if khz:
            info["cpu_base_ghz"] = round(khz / 1e6, 3)
    if info["cpu_max_ghz"] is None:
        khz = _read_int_file("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
        if khz:
            info["cpu_max_ghz"] = round(khz / 1e6, 3)

    # Final fallback for known SKUs (Azure Xeon, i7-14700K under WSL2, etc.).
    known = lookup_known_freqs(info["cpu_name"])
    if info["cpu_base_ghz"] is None and "base_ghz" in known:
        info["cpu_base_ghz"] = known["base_ghz"]
    if info["cpu_max_ghz"] is None and "max_ghz" in known:
        info["cpu_max_ghz"] = known["max_ghz"]

    return info


def _parse_cache_kb(raw: str) -> int | None:
    # lscpu prints things like "32 KiB (8 instances)" or "30 MiB (1 instance)"
    m = re.match(r"([\d.]+)\s*([KMG])iB", raw)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).upper()
    mult = {"K": 1, "M": 1024, "G": 1024 * 1024}[unit]
    return int(val * mult)


def detect_cpu_macos() -> dict:
    info: dict = {
        "cpu_name": _run(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "cpu_vendor": _run(["sysctl", "-n", "machdep.cpu.vendor"]) or "Apple",
        "cpu_cores_physical": _int(_run(["sysctl", "-n", "hw.physicalcpu"])),
        "cpu_cores_logical": _int(_run(["sysctl", "-n", "hw.logicalcpu"])),
        "cpu_base_ghz": None,
        "cpu_max_ghz": None,
        "cpu_l1d_kb": _kb(_int(_run(["sysctl", "-n", "hw.l1dcachesize"]))),
        "cpu_l2_kb": _kb(_int(_run(["sysctl", "-n", "hw.l2cachesize"]))),
        "cpu_l3_kb": _kb(_int(_run(["sysctl", "-n", "hw.l3cachesize"]))),
    }
    cpu_freq = _int(_run(["sysctl", "-n", "hw.cpufrequency"]))
    if cpu_freq:
        info["cpu_base_ghz"] = round(cpu_freq / 1e9, 3)
    cpu_freq_max = _int(_run(["sysctl", "-n", "hw.cpufrequency_max"]))
    if cpu_freq_max:
        info["cpu_max_ghz"] = round(cpu_freq_max / 1e9, 3)

    # Apple Silicon: those sysctls don't exist on M-series; fall back to SKU lookup.
    known = lookup_known_freqs(info["cpu_name"])
    if info["cpu_base_ghz"] is None and "base_ghz" in known:
        info["cpu_base_ghz"] = known["base_ghz"]
    if info["cpu_max_ghz"] is None and "max_ghz" in known:
        info["cpu_max_ghz"] = known["max_ghz"]
    return info


def _int(s: str | None) -> int | None:
    try:
        return int(s) if s is not None else None
    except (TypeError, ValueError):
        return None


def _kb(byts: int | None) -> int | None:
    return byts // 1024 if isinstance(byts, int) else None


def detect_ram_gb() -> float | None:
    if shutil.which("free"):
        out = _run(["free", "-b"]) or ""
        m = re.search(r"^Mem:\s+(\d+)", out, re.MULTILINE)
        if m:
            return round(int(m.group(1)) / (1024**3), 2)
    # macOS / fallback
    memsize = _int(_run(["sysctl", "-n", "hw.memsize"]))
    if memsize:
        return round(memsize / (1024**3), 2)
    try:
        # /proc/meminfo as last resort
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            mem = _first_match(f.read(), r"^MemTotal:\s+(\d+)\s+kB")
        if mem:
            return round(int(mem) / (1024**2), 2)
    except OSError:
        pass
    return None


def detect_docker_ram_gb() -> float | None:
    """Memory of Docker's VM, i.e. what the containers can actually use.

    On macOS the containers run inside Docker Desktop's Linux VM, whose memory
    limit is usually lower than the Mac's RAM. On Linux and WSL2 `free` already
    reports the memory the containers see, so this is only needed on macOS.
    """
    total = _int((_run(["docker", "info", "--format", "{{.MemTotal}}"]) or "").strip() or None)
    return round(total / (1024**3), 2) if total else None


def detect_gpu() -> dict:
    info: dict = {
        "gpu_kind": "none",  # none | discrete | integrated
        "gpu_vendor": None,  # nvidia | apple | amd | intel
        "gpu_name": None,
        "gpu_memory_gb": None,
        "gpu_compute_capability": None,
        "gpu_count": 0,
    }

    nv = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if nv:
        lines = [ln.strip() for ln in nv.splitlines() if ln.strip()]
        if lines:
            first = [p.strip() for p in lines[0].split(",")]
            info["gpu_kind"] = "discrete"
            info["gpu_vendor"] = "nvidia"
            info["gpu_name"] = first[0] if len(first) > 0 else None
            if len(first) > 1:
                try:
                    info["gpu_memory_gb"] = round(float(first[1]) / 1024.0, 2)
                except ValueError:
                    pass
            if len(first) > 2:
                info["gpu_compute_capability"] = first[2]
            info["gpu_count"] = len(lines)
            return info

    # Apple Silicon integrated GPU
    if platform.system().lower() == "darwin":
        brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or ""
        if brand.lower().startswith("apple"):
            info["gpu_kind"] = "integrated"
            info["gpu_vendor"] = "apple"
            info["gpu_name"] = f"Apple GPU ({brand})"
            # Apple uses unified memory — gpu_memory_gb deliberately left null
            info["gpu_count"] = 1
            return info

    return info


def detect_hostname() -> str:
    return os.environ.get("HOSTNAME_LABEL") or socket.gethostname()


def build_profile() -> dict:
    os_info = detect_os()
    arch = detect_arch(os_info["machine"])

    if os_info["os_family"] in {"linux", "wsl2"}:
        cpu = detect_cpu_linux()
    elif os_info["os_family"] == "macos":
        cpu = detect_cpu_macos()
    else:
        cpu = {}

    profile = {
        "schema_version": 1,
        "hostname_label": detect_hostname(),
        "arch": arch,
        **os_info,
        **cpu,
        "ram_gb": detect_ram_gb(),
        **detect_gpu(),
    }
    profile["has_gpu"] = profile.get("gpu_kind") != "none"
    if os_info["os_family"] == "macos":
        docker_ram = detect_docker_ram_gb()
        if docker_ram:
            profile["ram_host_gb"] = profile["ram_gb"]
            profile["ram_gb"] = docker_ram
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write hw_profile.json into (default: cwd).",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Also echo the JSON to stdout.",
    )
    args = parser.parse_args()

    profile = build_profile()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hw_profile.json"
    out_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    if args.print:
        json.dump(profile, sys.stdout, indent=2)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
