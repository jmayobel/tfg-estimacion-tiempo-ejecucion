#!/usr/bin/env python3
"""
Dataset Creator - Merge per-service CSVs into a single all_services.csv

Supports two modes:
  1. Orchestrator mode (--run-dir + --output-dir):
     Merges CSVs from the current run AND all previous runs found under
     <output-dir>/runs/*/ into <output-dir>/all_services.csv.

  2. Legacy / standalone mode (positional directory):
     Merges all CSVs found directly in the given directory into
     <directory>/all_services.csv.
"""

import argparse
import glob
import logging
import os
import sys

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_csvs_in_dir(directory):
    """Return sorted list of CSV file paths in *directory* (non-recursive),
    excluding meta files like all_services.csv and dataset.csv."""
    csvs = []
    try:
        for f in sorted(os.listdir(directory)):
            if f.endswith(".csv") and f not in ("all_services.csv", "dataset.csv"):
                csvs.append(os.path.join(directory, f))
    except (PermissionError, OSError) as e:
        logger.error(f"Error reading directory {directory}: {e}")
    return csvs


def load_csvs(csv_files):
    """Load a list of CSV paths into a list of DataFrames."""
    frames = []
    for csv_path in csv_files:
        basename = os.path.basename(csv_path)
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"  {basename}: {len(df)} rows")
            print(f"  {basename}: {len(df)} rows")
            frames.append(df)
        except pd.errors.EmptyDataError:
            logger.warning(f"Empty CSV file: {csv_path}, skipping")
            print(f"  WARNING: {basename} is empty, skipping")
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV {csv_path}: {e}")
            print(f"  ERROR: Failed to parse {basename}")
        except Exception as e:
            logger.error(f"Unexpected error reading {csv_path}: {e}")
            print(f"  ERROR: Failed to read {basename}")
    return frames


def save_combined(frames, output_path):
    """Concatenate frames, deduplicate, and save to output_path."""
    if not frames:
        logger.error("No valid CSV files could be loaded")
        print("\nERROR: No valid CSV files to merge")
        sys.exit(1)

    try:
        combined = pd.concat(frames, ignore_index=True)
        combined.drop_duplicates(inplace=True)
        logger.info(f"Combined {len(frames)} dataframes: {len(combined)} total rows")
    except ValueError as e:
        logger.error(f"Error concatenating dataframes: {e}")
        print(f"\nERROR: Failed to merge dataframes: {e}")
        sys.exit(1)

    try:
        combined.to_csv(output_path, index=False)
        logger.info(f"Saved combined dataset to {output_path}")
        print(f"\nMaster dataset: {output_path} ({len(combined)} total rows)")
    except (PermissionError, OSError) as e:
        logger.error(f"Error writing to {output_path}: {e}")
        print(f"\nERROR: Failed to write output file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-service CSVs into a single all_services.csv"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=None,
        help="(Legacy) Directory containing CSV files to merge",
    )
    parser.add_argument(
        "--run-dir",
        help="Directory of the current run (e.g. output/runs/2026-02-20_03-21-07)",
    )
    parser.add_argument(
        "--output-dir",
        help="Top-level output directory where all_services.csv is saved",
    )
    args = parser.parse_args()

    # ── Orchestrator mode: scan ALL runs ──────────────────────
    if args.run_dir and args.output_dir:
        runs_root = os.path.join(args.output_dir, "runs")
        if not os.path.isdir(runs_root):
            logger.error(f"Runs directory not found: {runs_root}")
            print(f"ERROR: {runs_root} does not exist")
            sys.exit(1)

        # Collect CSVs from every run subfolder (sorted for determinism)
        run_dirs = sorted(glob.glob(os.path.join(runs_root, "*")))
        all_csvs = []
        for rd in run_dirs:
            if os.path.isdir(rd):
                all_csvs.extend(find_csvs_in_dir(rd))

        if not all_csvs:
            logger.info("No service CSV files found across runs")
            print("No service CSV files found in any run.")
            return

        total_runs = len(run_dirs)
        print(f"Scanning {total_runs} run(s) — found {len(all_csvs)} CSV file(s):")
        frames = load_csvs(all_csvs)
        output_path = os.path.join(args.output_dir, "all_services.csv")
        save_combined(frames, output_path)
        return

    # ── Legacy mode: single directory ─────────────────────────
    output_dir = args.directory
    if not output_dir:
        parser.print_help()
        sys.exit(1)

    if not os.path.isdir(output_dir):
        logger.error(f"Output directory does not exist: {output_dir}")
        print(f"ERROR: {output_dir} is not a valid directory")
        sys.exit(1)

    csv_files = find_csvs_in_dir(output_dir)
    if not csv_files:
        logger.info(f"No service CSV files found in {output_dir}")
        print("No service CSV files found.")
        return

    print(f"Merging {len(csv_files)} CSV files:")
    frames = load_csvs(csv_files)
    output_path = os.path.join(output_dir, "all_services.csv")
    save_combined(frames, output_path)


if __name__ == "__main__":
    main()
