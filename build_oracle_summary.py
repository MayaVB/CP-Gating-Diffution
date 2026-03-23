"""Build oracle_summary.csv from per-try _results.csv files.

Reads:
    {per_try_dir}/try_0/_results.csv
    {per_try_dir}/try_1/_results.csv
    ...
    {per_try_dir}/try_{K-1}/_results.csv

Produces:
    {per_try_dir}/oracle_summary.csv
    columns: filename, try_0, try_1, ..., try_{K-1}   (sisdr_enh per try)

Usage:
    python build_oracle_summary.py \
        --per_try_dir voicebank/test_weiner_ablation_per_try \
        --kmax 10
"""

import argparse
import csv
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_try_dir", required=True,
                        help="Base directory containing try_0/, try_1/, ... subdirs")
    parser.add_argument("--kmax", type=int, default=10,
                        help="Number of tries (default: 10)")
    parser.add_argument("--out_csv", default=None,
                        help="Output path (default: {per_try_dir}/oracle_summary.csv)")
    args = parser.parse_args()

    try_cols = [f"try_{t}" for t in range(args.kmax)]

    per_try = {}
    for t in range(args.kmax):
        path = os.path.join(args.per_try_dir, f"try_{t}", "_results.csv")
        if not os.path.isfile(path):
            sys.exit(f"ERROR: Missing {path} — run calc_metrics.py for try_{t} first.")
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                fname = row["filename"].lstrip("./").lstrip("/")
                per_try.setdefault(fname, {})[t] = float(row["sisdr_enh"])
        print(f"  try_{t}: loaded {sum(1 for v in per_try.values() if t in v)} files")

    filenames = sorted(per_try)
    missing = [f for f in filenames if len(per_try[f]) < args.kmax]
    if missing:
        print(f"WARNING: {len(missing)} files missing some tries — skipping them:")
        for f in missing[:10]:
            print(f"  {f}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        filenames = [f for f in filenames if f not in missing]

    out_path = args.out_csv or os.path.join(args.per_try_dir, "oracle_summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename"] + try_cols)
        writer.writeheader()
        for fname in filenames:
            row = {"filename": fname}
            for t in range(args.kmax):
                row[f"try_{t}"] = per_try[fname][t]
            writer.writerow(row)

    print(f"\noracle_summary.csv written: {len(filenames)} files  →  {out_path}")


if __name__ == "__main__":
    main()
