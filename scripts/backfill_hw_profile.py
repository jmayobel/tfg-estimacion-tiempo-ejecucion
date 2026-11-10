#!/usr/bin/env python3
"""Patch existing hw_profile.json files with cpu_base_ghz / cpu_max_ghz.

Re-runs every hw_profile.json under `--root` through the SKU lookup table in
hw_profile.py and fills in any null frequency fields. Idempotent: existing
non-null values are preserved. Intended as a one-shot recovery for runs
captured before hw_profile.py learned to handle VMs / WSL2 / Apple Silicon,
so the resulting ml_dataset.csv has non-null hw_cpu_base_ghz / hw_cpu_max_ghz.

After running this, re-build the per-host ml_dataset.csv with
`build_ml_dataset.py` to pick up the corrected fields.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Import the SKU table from the live detector so both stay in sync.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hw_profile import lookup_known_freqs  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("backfill_hw_profile")


def patch_profile(profile: dict) -> list[str]:
    """Fill missing frequency fields in-place. Return list of fields modified."""
    known = lookup_known_freqs(profile.get("cpu_name"))
    changed: list[str] = []
    for field, spec_key in (("cpu_base_ghz", "base_ghz"), ("cpu_max_ghz", "max_ghz")):
        if profile.get(field) is None and spec_key in known:
            profile[field] = known[spec_key]
            changed.append(field)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="datasets",
        help="Directory to scan recursively for hw_profile.json (default: datasets).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files.",
    )
    args = parser.parse_args()

    files = sorted(Path(args.root).rglob("hw_profile.json"))
    if not files:
        log.warning("No hw_profile.json under %s", args.root)
        return 1

    n_patched = 0
    n_already_ok = 0
    n_unknown_sku = 0
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Skip %s: %s", f, exc)
            continue

        # Track state before patching.
        had_nulls = data.get("cpu_base_ghz") is None or data.get("cpu_max_ghz") is None
        changed = patch_profile(data)

        if changed:
            n_patched += 1
            log.info(
                "%s %s -> base=%s max=%s (%s)",
                "DRY-RUN" if args.dry_run else "PATCH",
                f,
                data.get("cpu_base_ghz"),
                data.get("cpu_max_ghz"),
                ",".join(changed),
            )
            if not args.dry_run:
                f.write_text(json.dumps(data, indent=2), encoding="utf-8")
        elif had_nulls:
            n_unknown_sku += 1
            log.warning(
                "No SKU lookup for cpu_name=%r in %s — add it to KNOWN_CPU_SPECS",
                data.get("cpu_name"),
                f,
            )
        else:
            n_already_ok += 1

    log.info(
        "Done. patched=%d already-ok=%d unknown-sku=%d total=%d",
        n_patched,
        n_already_ok,
        n_unknown_sku,
        len(files),
    )
    return 0 if n_unknown_sku == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
