"""Offline tau sweep for mid-step Wiener gating.

Produces the same column structure as tau_sweep_wiener_residual.csv:
    method, tau, escalation_rate, avg_K, mean_sisdr, worst10_mean_sisdr,
    harm_rate_vs_try0, mean_pesq, mean_estoi, mean_dnsmos, n_files

Required inputs
---------------
--log_csv            mid_wiener_log.csv from enhancement.py --mid_wiener_enable
                     Columns needed: filename, score_try0, scores_all (JSON),
                                     mid_wiener_step, mid_wiener_kmax

--oracle_summary_csv oracle_summary.csv from oracle_resample_eval.py
                     Columns needed: filename, try_0, try_1, ..., try_{K-1}
                                     (absolute SI-SDR, i.e. sisdr_enh per try)

Optional inputs
---------------
--per_try_results_dir  Base directory for per-try _results.csv files.
                       Pattern: {dir}/try_{t}/_results.csv
                       Columns used: filename, pesq, estoi, dnsmos_ovrl
                       Provides mean_pesq / mean_estoi / mean_dnsmos.
                       If omitted those columns are blank.

Decision rule (applied independently per utterance, per tau)
------------------------------------------------------------
    d0 = score_try0   (mid-step Wiener residual of try 0)
    if d0 <= tau:  select try 0
    else:          select argmin(scores_all)   (lowest mid-step Wiener score)

Baselines (appended to output CSV, tau column left blank)
---------------------------------------------------------
    always_try0      select try 0 for every utterance  (tau → +inf)
    always_best_of_K select argmin(scores_all) for every utterance (tau → -inf)

Usage
-----
    python sweep_mid_wiener_tau.py \\
        --log_csv path/to/mid_wiener_log.csv \\
        --oracle_summary_csv path/to/oracle_summary.csv \\
        [--per_try_results_dir path/to/enhanced_dir_base] \\
        [--tau_min 0.0 --tau_max 2.0 --n_tau 200] \\
        [--out_csv path/to/mid_wiener_tau_sweep.csv] \\
        [--no_plot]
"""

import argparse
import csv
import json
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(fname: str) -> str:
    """Strip leading ./ or / so filenames from different CSVs compare equal."""
    return fname.lstrip("./").lstrip("/")


def _safe_float(s) -> float | None:
    """Parse float; return None for blank, NaN, or non-finite values."""
    if s is None or str(s).strip() in ("", "nan", "NaN", "None", "inf", "-inf"):
        return None
    try:
        v = float(s)
        return v if (v == v and abs(v) != float("inf")) else None  # NaN / inf guard
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_log(log_csv: str) -> tuple[list[dict], int, int | None]:
    """Load mid_wiener_log.csv.

    Returns
    -------
    records  : list of per-utterance dicts with keys:
                 filename, d0 (float), scores_all (list[float])
    kmax     : mid_wiener_kmax (consistent across all rows)
    step     : mid_wiener_step (for method label; None if absent)

    Exits with a clear message if the file is missing or columns are absent.
    """
    if not os.path.isfile(log_csv):
        sys.exit(f"ERROR: --log_csv not found:\n  {log_csv}")

    required = {"filename", "score_try0", "scores_all", "mid_wiener_kmax"}
    records = []
    kmax_set = set()
    step = None

    with open(log_csv, newline="") as f:
        reader = csv.DictReader(f)
        missing_cols = required - set(reader.fieldnames or [])
        if missing_cols:
            sys.exit(
                f"ERROR: --log_csv is missing required columns: {sorted(missing_cols)}\n"
                f"  Found columns: {reader.fieldnames}\n"
                f"  Re-run enhancement.py --mid_wiener_enable to regenerate."
            )
        for row in reader:
            d0_raw = row["score_try0"]
            d0 = float(d0_raw) if d0_raw not in ("", "None") else float("inf")
            scores_all = json.loads(row["scores_all"])
            scores_all = [float(s) if s is not None else float("inf") for s in scores_all]
            kmax_declared = int(row["mid_wiener_kmax"])
            keff = len(scores_all)
            if keff != kmax_declared:
                sys.exit(
                    f"ERROR: Row for '{row['filename']}' has len(scores_all)={keff} "
                    f"but mid_wiener_kmax={kmax_declared}.  "
                    f"log_csv may be from a partial or mismatched run."
                )
            kmax_set.add(kmax_declared)
            if step is None and row.get("mid_wiener_step", "") not in ("", "None"):
                step = int(row["mid_wiener_step"])
            records.append({
                "filename":   _norm(row["filename"]),
                "d0":         d0,
                "scores_all": scores_all,
                "keff":       keff,
            })

    if not records:
        sys.exit(f"ERROR: --log_csv contains no data rows:\n  {log_csv}")
    if len(kmax_set) > 1:
        sys.exit(
            f"ERROR: Inconsistent mid_wiener_kmax values in log_csv: {kmax_set}\n"
            f"  All rows must have the same kmax."
        )
    return records, kmax_set.pop(), step


def _load_oracle_summary(oracle_summary_csv: str, kmax: int) -> dict[str, list[float]]:
    """Load oracle_summary.csv; return {filename: [sisdr_try0, ..., sisdr_try_{K-1}]}.

    Exits with a clear message if the file is missing or try columns are absent.
    """
    if not os.path.isfile(oracle_summary_csv):
        sys.exit(
            f"ERROR: --oracle_summary_csv not found:\n  {oracle_summary_csv}\n\n"
            f"  Per-try SI-SDR is required to compute mean_sisdr, worst10_mean_sisdr,\n"
            f"  and harm_rate_vs_try0.  Produce this file by running:\n\n"
            f"    python oracle_resample_eval.py --K_list {kmax} ...\n\n"
            f"  The output oracle_summary.csv must use the same test set and seed\n"
            f"  convention as the mid_wiener_log (gate_seed + 1000 * utt_idx)."
        )

    try_cols = [f"try_{t}" for t in range(kmax)]
    records: dict[str, list[float]] = {}

    with open(oracle_summary_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        if "filename" not in fieldnames:
            sys.exit(
                f"ERROR: --oracle_summary_csv has no 'filename' column.\n"
                f"  Found: {fieldnames}"
            )
        missing_try_cols = [c for c in try_cols if c not in fieldnames]
        if missing_try_cols:
            found_try = sorted(c for c in fieldnames if c.startswith("try_"))
            sys.exit(
                f"ERROR: --oracle_summary_csv is missing SI-SDR columns for kmax={kmax}:\n"
                f"  Missing: {missing_try_cols}\n"
                f"  Found try columns: {found_try}\n"
                f"  Re-run oracle_resample_eval.py with K >= {kmax}."
            )
        for row in reader:
            fname = _norm(row["filename"])
            if fname in records:
                continue  # skip duplicates (multiple subsets); first row wins
            try:
                sisdr_vals = [float(row[c]) for c in try_cols]
            except (ValueError, KeyError) as exc:
                sys.exit(
                    f"ERROR: Could not parse SI-SDR value in oracle_summary_csv "
                    f"for '{row.get('filename', '?')}': {exc}"
                )
            records[fname] = sisdr_vals

    if not records:
        sys.exit(f"ERROR: --oracle_summary_csv contains no data rows:\n  {oracle_summary_csv}")
    return records


def _load_per_try_metrics(per_try_results_dir: str, kmax: int
                          ) -> tuple[dict[int, dict[str, dict]], list[str]]:
    """Load optional per-try _results.csv files.

    Pattern: {per_try_results_dir}/try_{t}/_results.csv
    Returns ({try_idx: {filename: {pesq, estoi, dnsmos}}}, list_of_missing_paths).
    """
    per_try: dict[int, dict[str, dict]] = {}
    missing: list[str] = []
    for t in range(kmax):
        path = os.path.join(per_try_results_dir, f"try_{t}", "_results.csv")
        if not os.path.isfile(path):
            missing.append(path)
            continue
        recs: dict[str, dict] = {}
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = _norm(row.get("filename", ""))
                recs[fname] = {
                    "pesq":  _safe_float(row.get("pesq")),
                    "estoi": _safe_float(row.get("estoi")),
                    "dnsmos": _safe_float(row.get("dnsmos_ovrl")),
                }
        per_try[t] = recs
    return per_try, missing


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def _align(log_records: list[dict],
           oracle_sisdr: dict[str, list[float]],
           per_try_metrics: dict[int, dict[str, dict]],
           kmax: int) -> tuple[list[dict], list[str]]:
    """Join log_records with oracle_sisdr and per_try_metrics on filename.

    Returns (aligned_records, filenames_not_in_oracle).
    aligned_records have keys: filename, d0, scores_all, sisdr[k], pesq[k],
    estoi[k], dnsmos[k] for k in 0..kmax-1.
    """
    aligned: list[dict] = []
    not_in_oracle: list[str] = []

    for rec in log_records:
        fname = rec["filename"]
        if fname not in oracle_sisdr:
            not_in_oracle.append(fname)
            continue

        sisdr_vals = oracle_sisdr[fname]      # list[float], length kmax
        pesq_vals  = [None] * kmax
        estoi_vals = [None] * kmax
        dnsmos_vals = [None] * kmax
        for t in range(kmax):
            if t in per_try_metrics and fname in per_try_metrics[t]:
                m = per_try_metrics[t][fname]
                pesq_vals[t]   = m["pesq"]
                estoi_vals[t]  = m["estoi"]
                dnsmos_vals[t] = m["dnsmos"]

        aligned.append({
            "filename":   fname,
            "d0":         rec["d0"],
            "scores_all": rec["scores_all"],
            "keff":       rec["keff"],
            "sisdr":      sisdr_vals,
            "pesq":       pesq_vals,
            "estoi":      estoi_vals,
            "dnsmos":     dnsmos_vals,
        })

    return aligned, not_in_oracle


# ---------------------------------------------------------------------------
# Selection and metrics
# ---------------------------------------------------------------------------

def _select_all(aligned: list[dict], tau: float) -> list[dict]:
    """Apply decision rule for one tau; return per-utterance selection dicts."""
    result = []
    for rec in aligned:
        if rec["d0"] <= tau:
            t_sel = 0
        else:
            t_sel = int(np.argmin(rec["scores_all"]))
        result.append({
            "d0":              rec["d0"],
            "keff":            rec["keff"],
            "t_sel":           t_sel,
            "sisdr_selected":  rec["sisdr"][t_sel],
            "sisdr_try0":      rec["sisdr"][0],
            "pesq_selected":   rec["pesq"][t_sel],
            "estoi_selected":  rec["estoi"][t_sel],
            "dnsmos_selected": rec["dnsmos"][t_sel],
        })
    return result


def _compute_metrics(selected: list[dict], tau: float, kmax: int, method: str) -> dict:
    """Aggregate per-utterance selections into one sweep-table row."""
    n = len(selected)
    sisdr_arr  = np.array([r["sisdr_selected"] for r in selected], dtype=float)
    sisdr_try0 = np.array([r["sisdr_try0"]     for r in selected], dtype=float)

    escalation_rate = float(np.mean([r["d0"] > tau for r in selected]))
    # avg_K computed row-by-row from actual len(scores_all): 1 try if accepted,
    # keff tries if escalated.  This is exact when all runs complete unconditionally.
    avg_K = float(np.mean([
        1 if r["d0"] <= tau else r["keff"]
        for r in selected
    ]))

    mean_sisdr = float(np.mean(sisdr_arr))
    p10_sisdr  = float(np.percentile(sisdr_arr, 10))

    # worst10: mean of the bottom floor(10%) utterances by SI-SDR
    k10 = max(1, int(n * 0.1))
    worst10_mean_sisdr = float(np.mean(np.sort(sisdr_arr)[:k10]))

    harm_rate_vs_try0 = float(np.mean(sisdr_arr < sisdr_try0))

    def _opt_mean(vals):
        finite = [v for v in vals if v is not None]
        return float(np.mean(finite)) if finite else None

    mean_pesq  = _opt_mean([r["pesq_selected"]   for r in selected])
    mean_estoi = _opt_mean([r["estoi_selected"]   for r in selected])
    mean_dnsmos = _opt_mean([r["dnsmos_selected"] for r in selected])

    return {
        "method":             method,
        "tau":                tau,
        "escalation_rate":    escalation_rate,
        "avg_K":              avg_K,
        "mean_sisdr":         mean_sisdr,
        "p10_sisdr":          p10_sisdr,
        "worst10_mean_sisdr": worst10_mean_sisdr,
        "harm_rate_vs_try0":  harm_rate_vs_try0,
        "mean_pesq":          mean_pesq  if mean_pesq  is not None else "",
        "mean_estoi":         mean_estoi if mean_estoi is not None else "",
        "mean_dnsmos":        mean_dnsmos if mean_dnsmos is not None else "",
        "n_files":            n,
    }


def _baseline(aligned: list[dict], tau_sentinel: float, kmax: int, method: str) -> dict:
    """Compute a baseline row; tau column is left blank in the CSV."""
    sel = _select_all(aligned, tau_sentinel)
    row = _compute_metrics(sel, tau=tau_sentinel, kmax=kmax, method=method)
    row["method"] = method
    row["tau"] = ""   # baselines have no meaningful tau value
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_OUT_FIELDS = [
    "method", "tau", "escalation_rate", "avg_K",
    "mean_sisdr", "p10_sisdr", "worst10_mean_sisdr", "harm_rate_vs_try0",
    "mean_pesq", "mean_estoi", "mean_dnsmos", "n_files",
]


def main():
    parser = argparse.ArgumentParser(
        description="Offline tau sweep for mid-step Wiener gating — SI-SDR evaluation."
    )
    parser.add_argument("--log_csv", required=True,
                        help="mid_wiener_log.csv from enhancement.py --mid_wiener_enable")
    parser.add_argument("--oracle_summary_csv", required=True,
                        help="oracle_summary.csv from oracle_resample_eval.py; "
                             "must contain filename and try_0..try_{K-1} SI-SDR columns")
    parser.add_argument("--per_try_results_dir", default=None,
                        help="Directory containing try_0/_results.csv, try_1/_results.csv, "
                             "etc. for per-try PESQ/ESTOI/DNSMOS (optional)")
    parser.add_argument("--tau_min", type=float, default=None,
                        help="Lower bound of tau grid (default: min(score_try0))")
    parser.add_argument("--tau_max", type=float, default=None,
                        help="Upper bound of tau grid (default: max(score_try0))")
    parser.add_argument("--n_tau", type=int, default=200,
                        help="Number of tau grid points (default: 200)")
    parser.add_argument("--out_csv", default=None,
                        help="Output CSV path (default: alongside log_csv as "
                             "mid_wiener_tau_sweep.csv)")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip the diagnostic plot")
    args = parser.parse_args()

    # ---- Load log -------------------------------------------------------
    log_records, kmax, step = _load_log(args.log_csv)
    method_name = f"mid_wiener_step{step}" if step is not None else "mid_wiener"
    print(f"log_csv          : {len(log_records)} utterances  "
          f"(kmax={kmax}, method={method_name})")

    # ---- Load oracle SI-SDR ---------------------------------------------
    oracle_sisdr = _load_oracle_summary(args.oracle_summary_csv, kmax)
    print(f"oracle_summary   : {len(oracle_sisdr)} utterances with SI-SDR")

    # ---- Load optional per-try PESQ/ESTOI/DNSMOS ------------------------
    per_try_metrics: dict = {}
    if args.per_try_results_dir is not None:
        per_try_metrics, missing_paths = _load_per_try_metrics(
            args.per_try_results_dir, kmax
        )
        if missing_paths:
            print(f"\nWARNING: {len(missing_paths)} per-try _results.csv not found "
                  f"(PESQ/ESTOI/DNSMOS will be blank for those tries):")
            for p in missing_paths:
                print(f"  {p}")
        loaded = sorted(per_try_metrics)
        print(f"per_try_metrics  : loaded tries {loaded}")
    else:
        print("per_try_results_dir not provided — mean_pesq/mean_estoi/mean_dnsmos "
              "will be blank.")

    # ---- Align ----------------------------------------------------------
    aligned, not_in_oracle = _align(log_records, oracle_sisdr, per_try_metrics, kmax)

    if not_in_oracle:
        n_miss = len(not_in_oracle)
        print(f"\nERROR: {n_miss} filename(s) from log_csv were not found in "
              f"oracle_summary_csv:")
        for fname in not_in_oracle[:20]:
            print(f"  {fname}")
        if n_miss > 20:
            print(f"  ... and {n_miss - 20} more")
        print(
            "\nMake sure oracle_summary_csv was produced from the same test set "
            "with matching filenames.\nExiting."
        )
        sys.exit(1)

    if not aligned:
        sys.exit("ERROR: No utterances remain after alignment. Exiting.")
    print(f"Aligned          : {len(aligned)} utterances for sweep")

    # ---- Tau grid -------------------------------------------------------
    d0_finite = [r["d0"] for r in aligned if r["d0"] != float("inf")]
    if not d0_finite:
        sys.exit(
            "ERROR: All score_try0 values are inf.  "
            "The target step (mid_wiener_step) was never reached — "
            "check that mid_wiener_step < N."
        )
    tau_min = args.tau_min if args.tau_min is not None else float(min(d0_finite))
    tau_max = args.tau_max if args.tau_max is not None else float(max(d0_finite))
    if tau_min >= tau_max:
        sys.exit(
            f"ERROR: tau_min ({tau_min:.6g}) >= tau_max ({tau_max:.6g}).  "
            "Provide explicit --tau_min / --tau_max."
        )
    tau_grid = np.linspace(tau_min, tau_max, args.n_tau)
    print(f"Tau grid         : [{tau_min:.6g}, {tau_max:.6g}]  ×  {args.n_tau} steps")

    # ---- Sweep ----------------------------------------------------------
    sweep_rows = []
    for tau in tau_grid:
        sel = _select_all(aligned, float(tau))
        row = _compute_metrics(sel, tau=float(tau), kmax=kmax, method=method_name)
        sweep_rows.append(row)

    # ---- Baselines ------------------------------------------------------
    bl_try0   = _baseline(aligned, float("inf"),  kmax, f"{method_name}_always_try0")
    bl_argmin = _baseline(aligned, float("-inf"), kmax, f"{method_name}_always_best_of_K")

    # ---- Save CSV -------------------------------------------------------
    if args.out_csv is None:
        args.out_csv = os.path.join(
            os.path.dirname(os.path.abspath(args.log_csv)),
            "mid_wiener_tau_sweep.csv"
        )
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    all_rows = sweep_rows + [bl_try0, bl_argmin]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_OUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSweep CSV saved  : {args.out_csv}")
    print(f"  {len(sweep_rows)} tau rows + 2 baseline rows")

    # ---- Summary --------------------------------------------------------
    best = max(sweep_rows, key=lambda r: r["mean_sisdr"])
    print("\n--- Baselines ---")
    print(f"  {bl_try0['method']}:   "
          f"mean_sisdr={bl_try0['mean_sisdr']:.4f}  "
          f"worst10={bl_try0['worst10_mean_sisdr']:.4f}  "
          f"harm={bl_try0['harm_rate_vs_try0']:.4f}")
    print(f"  {bl_argmin['method']}: "
          f"mean_sisdr={bl_argmin['mean_sisdr']:.4f}  "
          f"worst10={bl_argmin['worst10_mean_sisdr']:.4f}  "
          f"harm={bl_argmin['harm_rate_vs_try0']:.4f}")
    print("\n--- Best tau (by mean_sisdr) ---")
    print(f"  tau               = {best['tau']:.6g}")
    print(f"  mean_sisdr        = {best['mean_sisdr']:.4f}")
    print(f"  worst10_mean_sisdr= {best['worst10_mean_sisdr']:.4f}")
    print(f"  escalation_rate   = {best['escalation_rate']:.4f}")
    print(f"  avg_K             = {best['avg_K']:.3f}")
    print(f"  harm_rate_vs_try0 = {best['harm_rate_vs_try0']:.4f}")
    if best["mean_pesq"] != "":
        print(f"  mean_pesq         = {best['mean_pesq']:.4f}")
    if best["mean_estoi"] != "":
        print(f"  mean_estoi        = {best['mean_estoi']:.4f}")
    if best["mean_dnsmos"] != "":
        print(f"  mean_dnsmos       = {best['mean_dnsmos']:.4f}")

    # ---- Plot -----------------------------------------------------------
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            taus = [r["tau"]                for r in sweep_rows]
            ms   = [r["mean_sisdr"]         for r in sweep_rows]
            w10  = [r["worst10_mean_sisdr"]  for r in sweep_rows]
            hr   = [r["harm_rate_vs_try0"]   for r in sweep_rows]
            er   = [r["escalation_rate"]     for r in sweep_rows]

            fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

            ax = axes[0]
            ax.plot(taus, ms,  label="mean SI-SDR (selected)")
            ax.plot(taus, w10, label="worst-10% mean SI-SDR", linestyle="--")
            ax.axhline(bl_try0["mean_sisdr"],   color="tab:orange", linestyle=":",
                       label=f"{bl_try0['method']} ({bl_try0['mean_sisdr']:.3f})")
            ax.axhline(bl_argmin["mean_sisdr"], color="tab:green",  linestyle=":",
                       label=f"{bl_argmin['method']} ({bl_argmin['mean_sisdr']:.3f})")
            ax.axvline(best["tau"], color="red", linestyle="--",
                       label=f"best tau={best['tau']:.4g}")
            ax.set_ylabel("SI-SDR (dB)")
            ax.legend(fontsize=8)
            ax.grid(True)
            ax.set_title(f"{method_name} — tau sweep ({len(aligned)} utterances)")

            axes[1].plot(taus, hr)
            axes[1].axhline(bl_try0["harm_rate_vs_try0"],   color="tab:orange",
                            linestyle=":", label="always_try0")
            axes[1].axhline(bl_argmin["harm_rate_vs_try0"], color="tab:green",
                            linestyle=":", label="always_argmin")
            axes[1].axvline(best["tau"], color="red", linestyle="--")
            axes[1].set_ylabel("harm_rate_vs_try0")
            axes[1].set_ylim(0, 1)
            axes[1].legend(fontsize=8)
            axes[1].grid(True)

            axes[2].plot(taus, er)
            axes[2].axvline(best["tau"], color="red", linestyle="--")
            axes[2].set_xlabel("tau")
            axes[2].set_ylabel("escalation_rate")
            axes[2].set_ylim(0, 1)
            axes[2].grid(True)

            plt.tight_layout()
            plot_path = args.out_csv.replace(".csv", ".png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Plot saved       : {plot_path}")
        except ImportError:
            print("matplotlib not available — skipping plot.")


if __name__ == "__main__":
    main()
