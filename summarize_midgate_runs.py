#!/usr/bin/env python3
"""Collect ablation-table numbers across multiple mid-gate / end-gate run dirs.

For each enhanced_dir, reads:
  - tail_sisdr_stats.json  (mean SI-SDR, P10 SI-SDR)
  - _results.csv           (per-utterance sisdr_enh, pesq)
  - mid_wiener_log.csv     (effective_k, mid_wiener_step, mid_wiener_kmax)
     OR gate_log.csv        (num_restarts — for legacy end-gate runs)

Outputs one row per run:
  label | gate_step | kmax | avg_NFE | avg_K | mean_sisdr | p10_sisdr | mean_pesq

avg_NFE note
------------
For mid_wiener runs: effective_k == kmax always (all tries run to completion),
so avg_NFE = kmax * N_diffusion_steps.  avg_K = kmax (constant).
For end-gate (legacy) runs: avg_K = mean(num_restarts + 1); avg_NFE = avg_K * N.
N_diffusion_steps defaults to 30; override with --N if different.

Usage
-----
python summarize_midgate_runs.py \\
    --runs  end_N30:voicebank/test_end_gate \\
            mid_N28:voicebank/test_mid_N28 \\
            mid_N22:voicebank/test_mid_N22 \\
            mid_N20:voicebank/test_mid_N20 \\
    --N 30
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_run(label: str, enhanced_dir: Path, N: int) -> dict:
    row = {"label": label, "dir": str(enhanced_dir)}

    # ------------------------------------------------------------------ #
    # SI-SDR stats from tail_sisdr_stats.json                             #
    # ------------------------------------------------------------------ #
    json_path = enhanced_dir / "tail_sisdr_stats.json"
    if json_path.is_file():
        with open(json_path) as f:
            stats = json.load(f)
        row["mean_sisdr"] = stats.get("sisdr_enh_mean", float("nan"))
        row["p10_sisdr"]  = stats.get("sisdr_enh_p10",  float("nan"))
    else:
        row["mean_sisdr"] = float("nan")
        row["p10_sisdr"]  = float("nan")
        print(f"  [warn] {label}: tail_sisdr_stats.json not found — run calc_metrics.py first", file=sys.stderr)

    # ------------------------------------------------------------------ #
    # PESQ from _results.csv                                              #
    # ------------------------------------------------------------------ #
    results_path = enhanced_dir / "_results.csv"
    if results_path.is_file():
        df = pd.read_csv(results_path)
        if "pesq" in df.columns:
            row["mean_pesq"] = float(df["pesq"].dropna().mean())
        else:
            row["mean_pesq"] = float("nan")
            print(f"  [warn] {label}: no 'pesq' column in _results.csv", file=sys.stderr)
    else:
        row["mean_pesq"] = float("nan")
        print(f"  [warn] {label}: _results.csv not found — run calc_metrics.py first", file=sys.stderr)

    # ------------------------------------------------------------------ #
    # avg_K / avg_NFE                                                     #
    # mid_wiener_log.csv  → effective_k always == kmax (constant)        #
    # gate_log.csv        → num_restarts varies per utterance             #
    # ------------------------------------------------------------------ #
    mw_path   = enhanced_dir / "mid_wiener_log.csv"
    gate_flat = enhanced_dir / "gate_log.csv"
    gate_path = enhanced_dir / "calib" / "gate_log.csv"

    if mw_path.is_file():
        mw = pd.read_csv(mw_path)
        kmax      = int(mw["mid_wiener_kmax"].iloc[0]) if "mid_wiener_kmax" in mw.columns else None
        gate_step = int(mw["mid_wiener_step"].iloc[0]) if "mid_wiener_step" in mw.columns else None
        eff_k     = pd.to_numeric(mw["effective_k"], errors="coerce").dropna()
        avg_k     = float(eff_k.mean()) if len(eff_k) > 0 else float("nan")
        row["gate_step"] = gate_step
        row["kmax"]      = kmax
        row["avg_K"]     = avg_k
        row["avg_NFE"]   = avg_k * N if not np.isnan(avg_k) else float("nan")
        row["source"]    = "mid_wiener_log"

    elif gate_flat.is_file() or gate_path.is_file():
        gp = gate_flat if gate_flat.is_file() else gate_path
        gl = pd.read_csv(gp)
        row["gate_step"] = N   # end-gate: full N steps
        row["kmax"]      = None
        if "num_restarts" in gl.columns:
            nr    = pd.to_numeric(gl["num_restarts"], errors="coerce").dropna()
            avg_k = float((nr + 1).mean()) if len(nr) > 0 else float("nan")
        else:
            avg_k = float("nan")
            print(f"  [warn] {label}: gate_log.csv has no num_restarts column", file=sys.stderr)
        row["avg_K"]   = avg_k
        row["avg_NFE"] = avg_k * N if not np.isnan(avg_k) else float("nan")
        row["source"]  = "gate_log"

    else:
        row["gate_step"] = None
        row["kmax"]      = None
        row["avg_K"]     = float("nan")
        row["avg_NFE"]   = float("nan")
        row["source"]    = "none"
        print(f"  [warn] {label}: no mid_wiener_log.csv or gate_log.csv found", file=sys.stderr)

    return row


def _fmt(v, fmt=".2f"):
    if isinstance(v, float) and np.isnan(v):
        return "n/a"
    if v is None:
        return "—"
    return format(v, fmt)


def main():
    parser = argparse.ArgumentParser(
        description="Print ablation table across mid-gate / end-gate runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--runs", nargs="+", required=True,
        metavar="LABEL:DIR",
        help="One or more label:enhanced_dir pairs",
    )
    parser.add_argument(
        "--N", type=int, default=30,
        help="Total diffusion steps used in all runs (default: 30)",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Optional: save table to this CSV path",
    )
    args = parser.parse_args()

    rows = []
    for spec in args.runs:
        if ":" not in spec:
            print(f"ERROR: expected LABEL:DIR, got '{spec}'", file=sys.stderr)
            sys.exit(1)
        label, dpath = spec.split(":", 1)
        enhanced_dir = Path(dpath)
        if not enhanced_dir.is_dir():
            print(f"  [warn] {label}: directory not found: {enhanced_dir}", file=sys.stderr)
        rows.append(_load_run(label, enhanced_dir, args.N))

    # ------------------------------------------------------------------ #
    # Print table                                                         #
    # ------------------------------------------------------------------ #
    col_w = max(len(r["label"]) for r in rows) + 2
    header = (
        f"{'run':<{col_w}}  {'gate_step':>10}  {'kmax':>5}  "
        f"{'avg_K':>7}  {'avg_NFE':>8}  {'mean_sisdr':>11}  "
        f"{'p10_sisdr':>10}  {'mean_pesq':>10}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['label']:<{col_w}}  "
            f"{_fmt(r['gate_step'], 'd') if r['gate_step'] is not None else '—':>10}  "
            f"{_fmt(r['kmax'], 'd') if r['kmax'] is not None else '—':>5}  "
            f"{_fmt(r['avg_K'], '.2f'):>7}  "
            f"{_fmt(r['avg_NFE'], '.1f'):>8}  "
            f"{_fmt(r['mean_sisdr'], '.2f'):>11}  "
            f"{_fmt(r['p10_sisdr'], '.2f'):>10}  "
            f"{_fmt(r['mean_pesq'], '.3f'):>10}"
        )
    print(sep)

    # avg_NFE note for mid_wiener runs
    mw_rows = [r for r in rows if r["source"] == "mid_wiener_log"]
    if mw_rows:
        print(
            "\n  Note: mid_wiener runs always execute all kmax tries "
            "(effective_k == kmax), so avg_K == kmax and avg_NFE == kmax*N "
            "for every utterance. These columns are constant within each run."
        )

    if args.csv:
        df = pd.DataFrame(rows)[
            ["label", "gate_step", "kmax", "avg_K", "avg_NFE",
             "mean_sisdr", "p10_sisdr", "mean_pesq"]
        ]
        df.to_csv(args.csv, index=False)
        print(f"\n  Saved to {args.csv}")


if __name__ == "__main__":
    main()
