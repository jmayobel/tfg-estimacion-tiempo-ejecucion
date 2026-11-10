#!/usr/bin/env python3
"""Inject machine-calibration features into an existing per-host ml_dataset.csv.

Calibration (scripts/calibrate.py output) is a property of the HOST, so the same
values apply to every row of that host. This lets us add the calib_* features to
already-collected data WITHOUT re-running the benchmark batches — only the ~15 s
calibration needs to run once per machine.

Usage:
  # from a calibration.json produced by calibrate.py:
  python3 scripts/backfill_calibration.py --host datasets/macbook-air-m4 \
      --calib-json /path/to/calibration.json

  # or inline (paste the JSON):
  python3 scripts/backfill_calibration.py --host datasets/macbook-air-m4 \
      --calib '{"calib_gflops_st": 366.3, ...}'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

CALIB_COLS = [
    "calib_gflops_st",
    "calib_gflops_mt",
    "calib_mem_gbps",
    "calib_pyloop_mops",
    "calib_parallel_speedup",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", required=True, help="datasets/<host> directory")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--calib-json", help="Path to calibration.json")
    g.add_argument("--calib", help="Inline JSON string")
    args = ap.parse_args()

    calib = (
        json.loads(Path(args.calib_json).read_text()) if args.calib_json else json.loads(args.calib)
    )

    csv = Path(args.host) / "ml_dataset.csv"
    if not csv.exists():
        print(f"ERROR: {csv} not found", file=sys.stderr)
        return 1
    df = pd.read_csv(csv)

    for col in CALIB_COLS:
        df[col] = calib.get(col)

    df.to_csv(csv, index=False)
    vals = {c: calib.get(c) for c in CALIB_COLS}
    print(f"✓ {csv}: injected calib into {len(df)} rows -> {vals}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
