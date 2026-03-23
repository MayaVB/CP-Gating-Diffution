#!/usr/bin/env python3
"""Gating-location ablation table from per-location tau-sweep CSVs.

For each gate location (N=30, N=28, N=22, N=20), reads the tau-sweep CSV
produced by sweep_mid_wiener_tau.py, finds the row whose escalation_rate is
closest to a target value, and prints a compact ablation table with:

    gate_location | avg_K | mean_sisdr | p10_sisdr | mean_pesq

Also prints a LaTeX table row block.

Usage
-----
python summarize_midstep_ablation.py \\
    --runs  "end N=30:voicebank/mid_gate_N30/mid_wiener_tau_sweep.csv" \\
            "mid N=28:voicebank/mid_gate_N28/mid_wiener_tau_sweep.csv" \\
            "mid N=22:voicebank/mid_gate_N22/mid_wiener_tau_sweep.csv" \\
            "mid N=20:voicebank/mid_gate_N20/mid_wiener_tau_sweep.csv" \\
    --target_escalation 0.22 \\
    --out_csv voicebank/midstep_ablation_table.csv

Notes
-----
- Each tau-sweep CSV must have been produced by sweep_mid_wiener_tau.py
  (columns: escalation_rate, avg_K, mean_sisdr, p10_sisdr, mean_pesq).
- p10_sisdr requires re-running sweep_mid_wiener_tau.py after the patch
  that added that column (rows produced before the patch will show n/a).
- Baseline rows (tau == "") are excluded from the target-escalation search.
- For the N=30 end-gating row: run the mid_wiener pipeline with
  --mid_wiener_step 29 (last diffusion step, 0-indexed) to stay consistent.
"""

import sys
import csv
import argparse

import numpy as np


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load_sweep(csv_path: str) -> list[dict]:
    """Load a tau_sweep CSV; return only numeric-tau rows (skip baselines)."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("tau", "").strip() == "":
                continue   # baseline row
            try:
                row["_tau_f"]            = float(row["tau"])
                row["_escalation_f"]     = float(row["escalation_rate"])
                row["_avg_K_f"]          = float(row["avg_K"])
                row["_mean_sisdr_f"]     = float(row["mean_sisdr"])
                row["_p10_sisdr_f"]      = float(row["p10_sisdr"])   if row.get("p10_sisdr",  "").strip() else None
                row["_mean_pesq_f"]      = float(row["mean_pesq"])   if row.get("mean_pesq",  "").strip() else None
            except (ValueError, KeyError):
                continue
            rows.append(row)
    return rows


def _pick_row(rows: list[dict], target_esc: float) -> dict:
    """Return the row whose escalation_rate is closest to target_esc."""
    return min(rows, key=lambda r: abs(r["_escalation_f"] - target_esc))


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _f(v, fmt) -> str:
    return "n/a" if v is None else format(v, fmt)


def _latex_row(label: str, r: dict) -> str:
    sisdr  = _f(r["_mean_sisdr_f"], ".2f")
    p10    = _f(r["_p10_sisdr_f"],  ".2f")
    pesq   = _f(r["_mean_pesq_f"],  ".3f")
    avg_k  = _f(r["_avg_K_f"],      ".2f")
    esc    = _f(r["_escalation_f"], ".1%")
    return f"  {label} & {avg_k} & {esc} & {sisdr} & {p10} & {pesq} \\\\"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gating-location ablation table at matched escalation rate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--runs", nargs="+", required=True,
        metavar="LABEL:CSV",
        help="One or more 'label:tau_sweep_csv' pairs (use quotes if label has spaces)",
    )
    parser.add_argument(
        "--target_escalation", type=float, default=0.22,
        help="Target escalation rate (default: 0.22 = 22%%)",
    )
    parser.add_argument(
        "--out_csv", default=None,
        help="Optional: save table to this CSV path",
    )
    args = parser.parse_args()

    results = []
    for spec in args.runs:
        # Split on first ":" only
        if ":" not in spec:
            print(f"ERROR: expected LABEL:CSV, got '{spec}'", file=sys.stderr)
            sys.exit(1)
        label, csv_path = spec.split(":", 1)
        label = label.strip()

        try:
            rows = _load_sweep(csv_path)
        except FileNotFoundError:
            print(f"  [warn] {label}: file not found: {csv_path}", file=sys.stderr)
            results.append({"label": label, "csv": csv_path, "row": None})
            continue

        if not rows:
            print(f"  [warn] {label}: no numeric-tau rows in {csv_path}", file=sys.stderr)
            results.append({"label": label, "csv": csv_path, "row": None})
            continue

        chosen = _pick_row(rows, args.target_escalation)
        results.append({"label": label, "csv": csv_path, "row": chosen})
        actual_esc = chosen["_escalation_f"]
        if abs(actual_esc - args.target_escalation) > 0.05:
            print(
                f"  [warn] {label}: closest escalation_rate={actual_esc:.3f} "
                f"is >5pp from target {args.target_escalation:.3f}",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------ #
    # Print table                                                         #
    # ------------------------------------------------------------------ #
    col_w = max(len(r["label"]) for r in results) + 2
    header = (
        f"{'gate_location':<{col_w}}  {'avg_K':>7}  {'esc_rate':>9}  "
        f"{'mean_sisdr':>11}  {'p10_sisdr':>10}  {'mean_pesq':>10}"
    )
    sep = "-" * len(header)
    print(f"\nTarget escalation rate: {args.target_escalation:.1%}")
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        row = r["row"]
        if row is None:
            print(f"{r['label']:<{col_w}}  (no data)")
            continue
        print(
            f"{r['label']:<{col_w}}  "
            f"{_f(row['_avg_K_f'],      '.2f'):>7}  "
            f"{_f(row['_escalation_f'], '.3f'):>9}  "
            f"{_f(row['_mean_sisdr_f'], '.2f'):>11}  "
            f"{_f(row['_p10_sisdr_f'],  '.2f'):>10}  "
            f"{_f(row['_mean_pesq_f'],  '.3f'):>10}"
        )
    print(sep)

    # ------------------------------------------------------------------ #
    # LaTeX block                                                         #
    # ------------------------------------------------------------------ #
    print("\n--- LaTeX rows ---")
    print("  % gate_location & avg_K & esc_rate & mean_sisdr & p10_sisdr & mean_pesq")
    for r in results:
        row = r["row"]
        if row is None:
            print(f"  % {r['label']}: no data")
        else:
            print(_latex_row(r["label"], row))

    # ------------------------------------------------------------------ #
    # Optional CSV save                                                   #
    # ------------------------------------------------------------------ #
    if args.out_csv:
        fields = ["gate_location", "avg_K", "escalation_rate",
                  "mean_sisdr", "p10_sisdr", "mean_pesq", "tau", "source_csv"]
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in results:
                row = r["row"]
                writer.writerow({
                    "gate_location":   r["label"],
                    "avg_K":           _f(row["_avg_K_f"],      ".4f") if row else "",
                    "escalation_rate": _f(row["_escalation_f"], ".4f") if row else "",
                    "mean_sisdr":      _f(row["_mean_sisdr_f"], ".4f") if row else "",
                    "p10_sisdr":       _f(row["_p10_sisdr_f"],  ".4f") if row else "",
                    "mean_pesq":       _f(row["_mean_pesq_f"],  ".4f") if row else "",
                    "tau":             _f(row["_tau_f"],         ".6g") if row else "",
                    "source_csv":      r["csv"],
                })
        print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
