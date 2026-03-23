#!/usr/bin/env python3
"""Side-by-side comparison of multiple gate_log CSV runs for trajectory gates.

Reads gate_log CSVs produced by enhancement.py and prints a compact table
comparing restart-separation across runs, to help decide which gate to
calibrate first.

Usage:
    python compare_traj_gate_runs.py \\
        --csv run_jump/gate_log.csv run_curv/gate_log.csv run_both/gate_log.csv

    # Override auto-labels (parent dir name):
    python compare_traj_gate_runs.py \\
        --csv run_jump/gate_log.csv run_curv/gate_log.csv \\
        --labels traj_jump traj_curvature
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCORE_COLS = ["traj_jump_max", "traj_curvature_max", "pred_jump_max"]

_COL_WIDTH = 22   # label column width


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _valid_mean(series: pd.Series) -> float:
    v = _to_numeric(series).dropna()
    return float(v.mean()) if len(v) > 0 else float("nan")


def _load_run(csv_path: Path, label: str) -> dict:
    """Load one CSV and compute per-gate stats.  Returns a run-info dict."""
    df = pd.read_csv(csv_path, dtype=str)
    nr = _to_numeric(df.get("num_restarts", pd.Series([""] * len(df))))
    mask_clean     = (nr == 0)
    mask_restarted = (nr > 0)

    n_total     = len(df)
    n_clean     = int(mask_clean.sum())
    n_restarted = int(mask_restarted.sum())

    gates = {}
    for col in _SCORE_COLS:
        if col not in df.columns:
            continue
        mu_c = _valid_mean(df.loc[mask_clean,     col])
        mu_r = _valid_mean(df.loc[mask_restarted, col])
        delta = mu_r - mu_c if (not np.isnan(mu_c) and not np.isnan(mu_r)) else float("nan")
        ratio = (mu_r / mu_c) if (not np.isnan(mu_c) and mu_c > 0 and not np.isnan(mu_r)) else float("nan")
        gates[col] = {
            "mean_clean":     mu_c,
            "mean_restarted": mu_r,
            "delta":          delta,
            "ratio":          ratio,
        }

    # Best gate in this run: highest ratio among available gates.
    best_gate = None
    best_ratio = float("nan")
    for col, g in gates.items():
        r = g["ratio"]
        if not np.isnan(r) and (np.isnan(best_ratio) or r > best_ratio):
            best_gate  = col
            best_ratio = r

    return {
        "label":       label,
        "csv":         csv_path,
        "n_total":     n_total,
        "n_clean":     n_clean,
        "n_restarted": n_restarted,
        "gates":       gates,
        "best_gate":   best_gate,
        "best_ratio":  best_ratio,
        "best_delta":  gates[best_gate]["delta"] if best_gate else float("nan"),
    }


def _fmt(v: float, fmt: str = ".4f") -> str:
    return format(v, fmt) if not np.isnan(v) else "n/a"


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _print_run_block(run: dict) -> None:
    lbl = run["label"]
    print(f"\n  Run: {lbl}  ({run['csv']})")
    print(f"    rows: {run['n_total']}  clean: {run['n_clean']}  restarted: {run['n_restarted']}")

    if not run["gates"]:
        print("    (no traj gate columns found)")
        return

    for col, g in run["gates"].items():
        print(f"    {col}:")
        print(f"      mean_clean={_fmt(g['mean_clean'], '.6g')}  "
              f"mean_restarted={_fmt(g['mean_restarted'], '.6g')}  "
              f"ratio={_fmt(g['ratio'], '.4f')}x  "
              f"delta={_fmt(g['delta'], '+.6g')}")

    if run["best_gate"]:
        print(f"    → best gate in this run: {run['best_gate']}  "
              f"(ratio={_fmt(run['best_ratio'], '.4f')}x)")


def _print_comparison_table(runs: list) -> None:
    """Print side-by-side compact table for all gate columns across runs."""
    all_cols = []
    for run in runs:
        for col in run["gates"]:
            if col not in all_cols:
                all_cols.append(col)

    if not all_cols:
        return

    print(f"\n{'='*70}")
    print("  SIDE-BY-SIDE TABLE")
    print(f"{'='*70}")

    for col in all_cols:
        print(f"\n  {col}")
        hdr = f"  {'run':{_COL_WIDTH}s}  {'mean_clean':>12}  {'mean_rest':>12}  {'ratio':>8}  {'delta':>10}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for run in runs:
            g = run["gates"].get(col)
            if g is None:
                print(f"  {run['label']:{_COL_WIDTH}s}  {'—':>12}  {'—':>12}  {'—':>8}  {'—':>10}")
            else:
                print(
                    f"  {run['label']:{_COL_WIDTH}s}"
                    f"  {_fmt(g['mean_clean'], '.6g'):>12}"
                    f"  {_fmt(g['mean_restarted'], '.6g'):>12}"
                    f"  {_fmt(g['ratio'], '.4f') + 'x':>8}"
                    f"  {_fmt(g['delta'], '+.6g'):>10}"
                )


def _print_final_recommendation(runs: list) -> None:
    """Rank runs by best available ratio and delta."""
    # Only runs that have at least one gate with valid ratio/delta
    valid = [r for r in runs if not np.isnan(r["best_ratio"])]

    print(f"\n{'='*70}")
    print("  FINAL RECOMMENDATION  (descriptive only, no statistical testing)")
    print(f"{'='*70}")

    if not valid:
        print("  No valid gate data found in any run.")
        return

    print(f"\n  Ranking by best restarted/clean ratio  (higher = better separation):")
    for rank, run in enumerate(
        sorted(valid, key=lambda r: r["best_ratio"], reverse=True), 1
    ):
        print(
            f"    #{rank}  {run['label']:{_COL_WIDTH}s}"
            f"  {run['best_gate']}  ratio={_fmt(run['best_ratio'], '.4f')}x"
        )

    print(f"\n  Ranking by best restarted/clean delta  (higher = bigger absolute gap):")
    for rank, run in enumerate(
        sorted(valid, key=lambda r: r["best_delta"], reverse=True), 1
    ):
        print(
            f"    #{rank}  {run['label']:{_COL_WIDTH}s}"
            f"  {run['best_gate']}  delta={_fmt(run['best_delta'], '+.6g')}"
        )

    top = sorted(valid, key=lambda r: r["best_ratio"], reverse=True)[0]
    print(f"\n  Suggested first calibration target: {top['label']} "
          f"({top['best_gate']}, ratio={_fmt(top['best_ratio'], '.4f')}x)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare trajectory-gate runs from multiple gate_log CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv", nargs="+", required=True, metavar="PATH",
        help="One or more gate_log CSV paths produced by enhancement.py",
    )
    parser.add_argument(
        "--labels", nargs="+", metavar="LABEL", default=None,
        help="Short labels for each CSV (default: parent directory name)",
    )
    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.csv]

    if args.labels is not None and len(args.labels) != len(csv_paths):
        print(
            f"ERROR: --labels has {len(args.labels)} entries but --csv has {len(csv_paths)}",
            file=sys.stderr,
        )
        sys.exit(1)

    labels = args.labels if args.labels else [p.parent.name or p.stem for p in csv_paths]

    runs = []
    for path, label in zip(csv_paths, labels):
        if not path.is_file():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        runs.append(_load_run(path, label))

    print(f"\nLoaded {len(runs)} run(s)")

    print(f"\n{'='*70}")
    print("  PER-RUN SUMMARIES")
    print(f"{'='*70}")
    for run in runs:
        _print_run_block(run)

    _print_comparison_table(runs)
    _print_final_recommendation(runs)
    print()


if __name__ == "__main__":
    main()
