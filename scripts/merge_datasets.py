#!/usr/bin/env python3
"""
Dataset Creator - Merge datasets from multiple machines
Combines CSVs collected on different computers (e.g., Windows + Mac)
into a single unified dataset.

Usage:
    python3 merge_datasets.py <dir1> <dir2> [<dir3> ...] [-o output.csv]

Example:
    python3 merge_datasets.py ./output_windows ./output_mac -o dataset.csv
"""

import argparse
import logging
import os
import sys

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_csvs(directory):
    """
    Find all CSV files in a directory.

    Args:
        directory (str): Path to directory to search

    Returns:
        list: Sorted list of CSV file paths found in directory
    """
    csvs = []
    try:
        for f in sorted(os.listdir(directory)):
            if f.endswith(".csv"):
                csvs.append(os.path.join(directory, f))
    except PermissionError as e:
        logger.error(f"Permission denied reading directory {directory}: {e}")
    except OSError as e:
        logger.error(f"Error reading directory {directory}: {e}")
    return csvs


def read_csv_safe(csv_path, label):
    """Read a CSV into a DataFrame; log/print with *label* and return None on failure."""
    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        logger.warning(f"Empty CSV file: {csv_path}")
        print(f"  WARNING: {label} is empty, skipping.")
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV {csv_path}: {e}")
        print(f"  ERROR: Failed to parse {label}")
    except Exception as e:
        logger.error(f"Unexpected error reading {csv_path}: {e}")
        print(f"  ERROR: Failed to read {label}")
    return None


def main():
    """
    Main entry point for dataset merging script.

    Loads CSV files from multiple directories (representing different machines),
    merges them into a single dataset, deduplicates rows, and sorts by hostname/service/time.
    """
    parser = argparse.ArgumentParser(description="Merge metric datasets from multiple machines")
    parser.add_argument(
        "directories",
        nargs="+",
        help="Directories containing CSV output from each machine",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="dataset.csv",
        help="Output file path (default: dataset.csv)",
    )
    parser.add_argument(
        "--prefer-combined",
        action="store_true",
        default=True,
        help="If all_services.csv exists in a dir, use it instead of individual files",
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

    all_frames = []

    for directory in args.directories:
        if not os.path.isdir(directory):
            logger.warning(f"{directory} is not a directory, skipping")
            print(f"WARNING: {directory} is not a directory, skipping.")
            continue

        print(f"\nProcessing: {directory}")
        logger.info(f"Processing directory: {directory}")

        # Check for combined file first
        combined_path = os.path.join(directory, "all_services.csv")
        if args.prefer_combined and os.path.exists(combined_path):
            df = read_csv_safe(combined_path, combined_path)
            if df is not None:
                hostname = df["hostname"].iloc[0] if "hostname" in df.columns else "unknown"
                print(f"  Using all_services.csv: {len(df)} rows (hostname: {hostname})")
                logger.info(
                    "Loaded all_services.csv from %s: %s rows, hostname: %s",
                    directory,
                    len(df),
                    hostname,
                )
                all_frames.append(df)
        else:
            # Load individual CSVs
            csvs = find_csvs(directory)
            if not csvs:
                logger.info("No CSV files found in %s", directory)
                print("  No CSV files found.")
                continue

            for csv_path in csvs:
                basename = os.path.basename(csv_path)
                if basename in ("all_services.csv", "dataset.csv"):
                    continue
                df = read_csv_safe(csv_path, basename)
                if df is not None:
                    print(f"  {basename}: {len(df)} rows")
                    logger.debug(f"Loaded {basename}: {len(df)} rows")
                    all_frames.append(df)

    if not all_frames:
        logger.error("No data found to merge")
        print("\nERROR: No data found to merge.")
        sys.exit(1)

    # Merge all
    logger.info(f"Merging {len(all_frames)} dataframes")
    try:
        merged = pd.concat(all_frames, ignore_index=True)
    except ValueError as e:
        logger.error(f"Error concatenating dataframes: {e}")
        print(f"\nERROR: Failed to merge dataframes: {e}")
        sys.exit(1)

    # Deduplicate: same hostname + timestamp + service_name
    before = len(merged)
    try:
        merged = merged.drop_duplicates(
            subset=["hostname", "timestamp", "service_name"],
            keep="last",
        )
    except KeyError as e:
        logger.error(f"Missing required column for deduplication: {e}")
        print(f"\nERROR: Missing required column: {e}")
        sys.exit(1)

    after = len(merged)
    if before != after:
        logger.info(f"Removed {before - after} duplicate rows")
        print(f"\nRemoved {before - after} duplicate rows")

    # Sort by hostname, service_name, timestamp
    try:
        merged = merged.sort_values(
            ["hostname", "service_name", "timestamp"],
            ignore_index=True,
        )
    except KeyError as e:
        logger.error(f"Missing required column for sorting: {e}")
        print(f"\nERROR: Missing required column: {e}")
        sys.exit(1)

    # Save
    try:
        merged.to_csv(args.output, index=False)
        logger.info(f"Saved merged dataset to {args.output}: {len(merged)} rows")
    except PermissionError as e:
        logger.error(f"Permission denied writing to {args.output}: {e}")
        print(f"\nERROR: Permission denied: {args.output}")
        sys.exit(1)
    except OSError as e:
        logger.error(f"Error writing to {args.output}: {e}")
        print(f"\nERROR: Failed to write output file: {e}")
        sys.exit(1)

    # Summary
    separator = "=" * 60
    print(f"\n{separator}")
    print(f"Merged Dataset: {args.output}")
    print(f"  Total rows: {len(merged)}")
    print(f"  Hostnames: {merged['hostname'].nunique()}")
    print(f"  Services: {merged['service_name'].nunique()}")
    print(f"  Columns: {len(merged.columns)}")

    print("\n  Rows per hostname:")
    for host, count in merged["hostname"].value_counts().items():
        print(f"    {host}: {count}")

    print("\n  Rows per service:")
    for svc, count in merged["service_name"].value_counts().items():
        print(f"    {svc}: {count}")

    print(f"{separator}")


if __name__ == "__main__":
    main()
