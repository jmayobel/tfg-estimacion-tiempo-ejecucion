#!/usr/bin/env python3
"""Standardized machine-capability calibration micro-benchmark.

Runs a SHORT, FIXED workload (identical on every host) and reports measured
capability indices. Because the workload is the same everywhere, each result is
a pure property of the machine — a strong, collinearity-free feature for the
execution-time model (unlike nominal specs, which are highly correlated across
hosts).

Indices (best of repeats — capability, robust to slow-core scheduling):
  calib_gflops_st   single-thread matmul throughput (GFLOP/s)
  calib_gflops_mt   all-thread matmul throughput   (GFLOP/s)
  calib_mem_gbps    memory bandwidth, STREAM-like triad (GB/s)
  calib_pyloop_mops single-thread scalar loop, BLAS-independent (Mops/s)
  calib_parallel_speedup  gflops_mt / gflops_st (measured usable parallelism)

Portable: stdlib + numpy only. Thread count is controlled by re-spawning a
worker with the BLAS thread env set BEFORE numpy import (works with
OpenBLAS / MKL / Apple Accelerate).

Usage:
  python3 calibrate.py                 # print JSON to stdout
  python3 calibrate.py --output-dir D  # also write D/calibration.json
  python3 calibrate.py --merge-hw D    # merge calib_* into D/hw_profile.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

MATMUL_N = 1280  # 2*N^3 ~ 4.2 GFLOP per matmul (~1-3 s single-thread)
MATMUL_REPS = 3
MEM_ELEMS = 32_000_000  # float64 arrays of 256 MB each (triad moves ~768 MB)
MEM_REPS = 5
PYLOOP_ITERS = 4_000_000
# On hybrid CPUs (Intel P/E-cores, ARM big.LITTLE) a single ST worker process can
# be scheduled onto an efficiency core for its whole lifetime, under-reporting
# single-thread capability 2-3x (observed: i7-14700K @5.6GHz measured 30 GFLOPS,
# below a 3.5GHz Xeon). Spawning several short-lived workers and keeping the best
# makes the measurement robust even where CPU pinning is unavailable (WSL2).
ST_SPAWNS = 3

# Env vars honoured by the BLAS backends numpy may link against
# (OpenBLAS / MKL / Apple Accelerate / numexpr). Must be set BEFORE numpy import.
BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _matmul_gflops() -> float:
    import numpy as np

    a = np.random.rand(MATMUL_N, MATMUL_N)
    b = np.random.rand(MATMUL_N, MATMUL_N)
    flop = 2.0 * MATMUL_N**3
    best = float("inf")
    for _ in range(MATMUL_REPS):
        t0 = time.perf_counter()
        c = a @ b
        dt = time.perf_counter() - t0
        best = min(best, dt)
        _ = c[0, 0]  # keep result
    return flop / best / 1e9


def _mem_bandwidth_gbps() -> float:
    import numpy as np

    a = np.random.rand(MEM_ELEMS)
    b = np.random.rand(MEM_ELEMS)
    s = 3.0
    bytes_moved = 3 * MEM_ELEMS * 8  # read b, read c(=a), write a
    best = float("inf")
    for _ in range(MEM_REPS):
        t0 = time.perf_counter()
        a = b + s * a  # triad
        dt = time.perf_counter() - t0
        best = min(best, dt)
    _ = a[0]
    return bytes_moved / best / 1e9


def _pyloop_mops() -> float:
    # BLAS-independent single-thread scalar throughput.
    t0 = time.perf_counter()
    x = 0.0
    for i in range(PYLOOP_ITERS):
        x += (i * 1.000001) % 7.0
    dt = time.perf_counter() - t0
    _ = x
    return PYLOOP_ITERS / dt / 1e6


def _pin_to_fastest_cpus() -> None:
    """Pin the ST worker to the highest-max-frequency CPUs the kernel reports.

    Counters hybrid-CPU scheduling (P/E-cores, big.LITTLE) putting the
    single-thread benchmark on a slow core. Pins to the whole top-frequency
    set (not one CPU) so the scheduler can still dodge a busy core. No-op when
    unsupported (macOS) or when cpufreq is not exposed (typical under WSL2) —
    there the best-of-ST_SPAWNS strategy is the fallback.
    """
    if not hasattr(os, "sched_setaffinity"):
        return
    try:
        freqs = {}
        for cpu in os.sched_getaffinity(0):
            path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq"
            with open(path) as fh:
                freqs[cpu] = int(fh.read().strip())
        if len(set(freqs.values())) < 2:
            return  # homogeneous CPU: nothing to gain
        top = max(freqs.values())
        os.sched_setaffinity(0, {c for c, f in freqs.items() if f == top})
    except (OSError, ValueError):
        return  # missing/partial cpufreq info: leave scheduling alone


def _worker(threads: int) -> dict:
    """Run inside a process whose BLAS thread env is already set."""
    if threads == 1:
        _pin_to_fastest_cpus()

    import numpy as np  # noqa: F401  (import after env is set)

    res = {"gflops": _matmul_gflops()}
    if threads == 0:  # all-thread worker also measures memory + pyloop
        res["mem_gbps"] = _mem_bandwidth_gbps()
    if threads == 1:
        res["pyloop_mops"] = _pyloop_mops()
    return res


def _run_worker_subprocess(threads: int) -> dict:
    env = dict(os.environ)
    n = "1" if threads == 1 else str(os.cpu_count() or 1)
    for var in BLAS_THREAD_VARS:
        env[var] = n
    out = subprocess.check_output(
        [sys.executable, os.path.abspath(__file__), "--worker", "--threads", str(threads)],
        env=env,
        text=True,
    )
    return json.loads(out.strip().splitlines()[-1])


def calibrate() -> dict:
    # Best of several ST worker processes: each spawn is a fresh scheduling
    # decision, so at least one should land on a performance core even where
    # pinning is unavailable. Max is the right aggregate — we want capability,
    # and every source of error (slow core, interference) only lowers the value.
    st_runs = [_run_worker_subprocess(1) for _ in range(ST_SPAWNS)]
    st = {
        "gflops": max(r["gflops"] for r in st_runs),
        "pyloop_mops": max(r.get("pyloop_mops", 0.0) for r in st_runs),
    }
    mt = _run_worker_subprocess(0)
    gflops_st = round(st["gflops"], 3)
    gflops_mt = round(mt["gflops"], 3)
    result = {
        "calib_gflops_st": gflops_st,
        "calib_gflops_mt": gflops_mt,
        "calib_mem_gbps": round(mt.get("mem_gbps", 0.0), 3),
        "calib_pyloop_mops": round(st.get("pyloop_mops", 0.0), 3),
        "calib_parallel_speedup": round(gflops_mt / gflops_st, 3) if gflops_st else None,
        "calib_arch": platform.machine().lower(),
        # v2: ST = best of ST_SPAWNS worker processes + fastest-core pinning
        # (hybrid-CPU fix). Identical results on homogeneous CPUs, so v1 values
        # from the cloud VMs / Mac remain comparable; hybrid hosts (windows
        # i7-14700K) must be recalibrated.
        "calib_schema_version": 2,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--threads", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default=None, help="Also write calibration.json here.")
    parser.add_argument(
        "--merge-hw", default=None, help="Merge calib_* fields into <dir>/hw_profile.json."
    )
    parser.add_argument("--print", action="store_true", help="Echo JSON to stdout.")
    args = parser.parse_args()

    if args.worker:
        print(json.dumps(_worker(args.threads)))
        return 0

    result = calibrate()

    if args.output_dir:
        from pathlib import Path

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "calibration.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

    if args.merge_hw:
        from pathlib import Path

        p = Path(args.merge_hw) / "hw_profile.json"
        hw = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        hw.update(result)
        p.write_text(json.dumps(hw, indent=2), encoding="utf-8")

    json.dump(result, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
