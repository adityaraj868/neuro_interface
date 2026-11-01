#!/usr/bin/env python3
"""Download a small, explicit PhysioNet EEGBCI subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from muse.data import download_eegbci  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "physionet")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--runs", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument(
        "--minimum-free-mb",
        type=int,
        default=100,
        help="Abort unless this much disk space is free before downloading.",
    )
    args = parser.parse_args()
    try:
        paths = download_eegbci(
            args.data_dir,
            args.subjects,
            args.runs,
            minimum_free_bytes=args.minimum_free_mb * 1024**2,
        )
    except Exception as exc:
        parser.exit(
            1,
            "MUSE dataset download failed. No EEG result was produced.\n"
            f"Reason: {exc}\n"
            "If this is a network/DNS error, run the same command on a machine "
            "with PhysioNet access and copy the EDF files under data/physionet/.\n",
        )
    print(f"Downloaded or found {len(paths)} EDF file(s) under {args.data_dir}:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

