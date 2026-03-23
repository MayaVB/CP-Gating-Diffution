#!/usr/bin/env python3
"""Compact summary of trajectory-gate statistics from a gate_log CSV.

Reads the gate_log CSV written by enhancement.py and prints per-gate
descriptive statistics for the per-step trajectory gates:

    traj_jump_max / traj_jump_max_step
    traj_curvature_max / traj_curvature_max_step

Statistics are printed for three subsets:
    ALL       — every row in the CSV
    clean     — rows where num_restarts == 0 (accepted on first try)
    restarted — rows where num_restarts >  0 (at least one abort)

For each gate, if both the max-score and max-step columns are present,
the Pearson correlation between them is also printed.

Usage:
    python analyze_traj_gate_log.py --csv path/to/gate_log.csv
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GATE_PAIRS = [
    ("traj_jump",      "traj_jump_max",      "traj_jump_max_step"),
    ("traj_curvature", "traj_curvature_max",  "traj_curvature_max_step"),
    ("pred_jump",      "pred_jump_max",       "pred_jump_max_step"),
]


def _to_numeric(series: pd.Series) -> pd.Series:
    """Coerce a column to float, treating empty strings and non-numeric as NaN."""
    return pd.to_numeric(series, errors="coerce")


def _describe(arr: np.ndarray, label: str) -> None:
    """Print descriptive statistics for a 1-D float array (NaNs excluded)."""
    valid = arr[~np.isnan(arr)]
    n = len(valid)
    if n == 0:
        print(f"    {label}: no valid rows")
        return
    p50, p90, p95 = np.percentile(valid, [50, 90, 95])
    print(
        f"    {label}: n={n}  mean={valid.mean():.6g}  std={valid.std():.6g}"
        f"  min={valid.min():.6g}  p50={p50:.6g}  p90={p90:.6g}"
        f"  p95={p95:.6g}  max={valid.max():.6g}"
    )


def _subset_stats(df: pd.DataFrame, col: str, label: str) -> None:
    """Print stats for one column over the given (already-filtered) DataFrame."""
    arr = _to_numeric(df[col]).to_numpy(dtype=float)
    _describe(arr, label)


def _valid_mean(series: pd.Series) -> float:
    """Mean of non-NaN values; returns NaN if none available."""
    v = _to_numeric(series).dropna()
    return float(v.mean()) if len(v) > 0 else float("nan")


def _correlation(df: pd.DataFrame, col_score: str, col_step: str) -> None:
    """Print Pearson correlation between max score and max step."""
    s = _to_numeric(df[col_score])
    t = _to_numeric(df[col_step])
    mask = (~s.isna()) & (~t.isna())
    n = mask.sum()
    if n < 3:
        print(f"    correlation({col_score}, {col_step}): n={n} — too few rows")
        return
    r = np.corrcoef(s[mask].to_numpy(float), t[mask].to_numpy(float))[0, 1]
    print(f"    corr(max_score, max_step): r={r:.4f}  (n={n})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise trajectory-gate statistics from a gate_log CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", required=True, help="Path to gate_log.csv produced by enhancement.py")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"ERROR: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path, dtype=str)   # read everything as str → handle empty cells
    print(f"\nLoaded {len(df)} rows from {csv_path}")

    # Build subset masks.  num_restarts may be empty string for some paths.
    nr = _to_numeric(df.get("num_restarts", pd.Series([""] * len(df))))
    mask_clean     = (nr == 0)
    mask_restarted = (nr > 0)
    subsets = [
        ("ALL",       df),
        ("clean",     df[mask_clean]),
        ("restarted", df[mask_restarted]),
    ]

    for gate_name, col_score, col_step in _GATE_PAIRS:
        has_score = col_score in df.columns
        has_step  = col_step  in df.columns

        if not has_score and not has_step:
            print(f"\n[{gate_name}] — columns not present in CSV (gate was not active)")
            continue

        print(f"\n{'='*60}")
        print(f"  Gate: {gate_name}")
        print(f"{'='*60}")

        if has_score:
            print(f"\n  Column: {col_score}")
            for label, sub in subsets:
                _subset_stats(sub, col_score, label)

        if has_step:
            print(f"\n  Column: {col_step}")
            for label, sub in subsets:
                _subset_stats(sub, col_step, label)

        if has_score and has_step:
            print(f"\n  Correlation:")
            _correlation(df, col_score, col_step)

    # -----------------------------------------------------------------------
    # Decision summary
    # -----------------------------------------------------------------------
    _print_decision_summary(df, mask_clean, mask_restarted)

    print()


def _print_decision_summary(
    df: pd.DataFrame,
    mask_clean: pd.Series,
    mask_restarted: pd.Series,
) -> None:
    """Print a compact restart-separation summary and ranking across gates."""

    print(f"\n{'='*60}")
    print("  DECISION SUMMARY")
    print(f"{'='*60}")

    n_clean     = int(mask_clean.sum())
    n_restarted = int(mask_restarted.sum())
    print(f"\n  Subsets: clean={n_clean}  restarted={n_restarted}")

    # Collect per-gate stats for ranking.
    ranking_rows = []   # list of (gate_name, ratio, delta)

    for gate_name, col_score, col_step in _GATE_PAIRS:
        if col_score not in df.columns:
            print(f"\n  [{gate_name}] not in CSV — skipped")
            continue

        mu_clean     = _valid_mean(df.loc[mask_clean,     col_score])
        mu_restarted = _valid_mean(df.loc[mask_restarted, col_score])

        print(f"\n  [{gate_name}]")

        if np.isnan(mu_clean):
            print("    mean_clean:     n/a (no valid rows)")
        else:
            print(f"    mean_clean:     {mu_clean:.6g}")

        if np.isnan(mu_restarted):
            print("    mean_restarted: n/a (no valid rows)")
        else:
            print(f"    mean_restarted: {mu_restarted:.6g}")

        if not np.isnan(mu_clean) and not np.isnan(mu_restarted):
            delta = mu_restarted - mu_clean
            ratio = (mu_restarted / mu_clean) if mu_clean > 0 else float("nan")
            print(f"    delta  (rest - clean): {delta:+.6g}")
            if not np.isnan(ratio):
                print(f"    ratio  (rest / clean): {ratio:.4f}x")
            ranking_rows.append((gate_name, ratio, delta))

        # Timing: do restarted trajectories peak earlier or later?
        if col_step in df.columns:
            ms_clean     = _valid_mean(df.loc[mask_clean,     col_step])
            ms_restarted = _valid_mean(df.loc[mask_restarted, col_step])
            if not np.isnan(ms_clean) and not np.isnan(ms_restarted):
                direction = "earlier" if ms_restarted < ms_clean else "later"
                print(
                    f"    peak step — clean: {ms_clean:.1f}  restarted: {ms_restarted:.1f}"
                    f"  → restarted peak {direction}"
                )

    # Ranking (only if both gates contributed).
    if len(ranking_rows) < 2:
        return

    print(f"\n  Ranking by ratio  (higher ratio = better separation):")
    for rank, (gname, ratio, _) in enumerate(
        sorted(ranking_rows, key=lambda r: r[1] if not np.isnan(r[1]) else -1, reverse=True), 1
    ):
        tag = f"{ratio:.4f}x" if not np.isnan(ratio) else "n/a"
        print(f"    #{rank}  {gname:<20s}  ratio={tag}")

    print(f"\n  Ranking by delta  (higher delta = better separation):")
    for rank, (gname, _, delta) in enumerate(
        sorted(ranking_rows, key=lambda r: r[2], reverse=True), 1
    ):
        print(f"    #{rank}  {gname:<20s}  delta={delta:+.6g}")


if __name__ == "__main__":
    main()
