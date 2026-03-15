"""Oracle resampling experiment — PESQ edition.

For each subset (worst-q% and random), run K independent enhancement attempts
and compute oracle (best-of-K) and mean-of-K absolute PESQ-WB.

Does NOT modify enhancement.py or calc_metrics.py — calls them via subprocess.
PESQ is always computed by calc_metrics.py (no extra flag needed).

Ranking  : worst-q% files by lowest baseline PESQ-WB
Selection: best try per file = argmax(PESQ) across K tries
Secondary: SI-SDR of the PESQ-oracle-selected output is also reported

Usage example:
    python oracle_resample_eval_pesq.py \\
        --test_dir voicebank/test_noisy \\
        --clean_dir voicebank/test_clean \\
        --baseline_enhanced_dir voicebank/test_enhanced_baseline \\
        --work_dir voicebank/oracle_exp_pesq \\
        --q_percent 20 \\
        --K_list 1 2 5 10 \\
        --seed_base 100 \\
        --enhancement_cmd "python enhancement.py --test_dir voicebank/test_noisy --ckpt checkpoints_updated/train_vb_29nqe0uh_epoch=115.ckpt" \\
        --metrics_cmd "python calc_metrics.py --clean_dir voicebank/test_clean --noisy_dir voicebank/test_noisy"

Outputs (all under --work_dir):
  oracle_summary_pesq.csv     – per-file best/mean PESQ + SI-SDR-of-best for K_max tries
  oracle_curve_pesq.csv       – (subset, K, n, mean_best_pesq, mean_mean_pesq, ci_low, ci_high)
  oracle_absolute_summary_pesq.csv – per-subset mean stats
  fit_report_pesq.txt         – OLS fits of mean_best_pesq vs log(K) / sqrt(log(K))
  plots/
    best_of_K_vs_K.png              – mean best-of-K PESQ vs K
    mean_of_K_pesq_vs_K.png         – mean mean-of-K PESQ vs K
    sisdr_of_pesq_oracle.png        – SI-SDR of PESQ-oracle-selected output vs K
    pesq_hist_worst.png             – pooled per-try PESQ distribution (worst subset)
    pesq_hist_random.png            – pooled per-try PESQ distribution (random subset)
    worst_sweep_curve_worst.png     – best/mean/baseline PESQ vs worst-x% cutoff
    worst_sweep_curve_worst.txt     – text table of the above
"""

import argparse
import random
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_results_csv(directory: Path) -> "Path | None":
    matches = list(directory.glob("*_results.csv"))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"Warning: multiple *_results.csv in {directory}; using {matches[0].name}")
    return matches[0]


def normalize_fname(fname: str, test_dir: Path) -> str:
    fname = fname.strip().replace("\\", "/")
    prefix = str(test_dir).replace("\\", "/").rstrip("/") + "/"
    if fname.startswith(prefix):
        fname = fname[len(prefix):]
    return fname.lstrip("/")


def run(cmd: str) -> bool:
    print(f"\n$ {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Warning: command failed (exit {e.returncode}), skipping this try.")
        return False


def load_metric(results_csv: Path, col: str) -> pd.Series:
    """Load filename → metric Series from a _results.csv. Returns empty Series if column absent."""
    df = pd.read_csv(results_csv)
    if col not in df.columns:
        print(f"  Warning: column '{col}' not in {results_csv.name} — returning empty Series.")
        return pd.Series(dtype=float, name=col)
    return df.set_index("filename")[col]


def _linregress(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    ss_yy = np.sum((y - y_mean) ** 2)
    slope = ss_xy / ss_xx if ss_xx != 0 else 0.0
    intercept = y_mean - slope * x_mean
    r2 = (ss_xy ** 2 / (ss_xx * ss_yy)) if (ss_xx * ss_yy) > 0 else 0.0
    return slope, intercept, r2


def _bootstrap_ci(arr: np.ndarray, n_resamples: int = 100,
                  rng: "np.random.Generator | None" = None) -> "tuple[float, float]":
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(arr)
    if n == 1:
        return float(arr[0]), float(arr[0])
    boot_means = rng.choice(arr, size=(n_resamples, n), replace=True).mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


# ---------------------------------------------------------------------------
# Curve aggregation
# ---------------------------------------------------------------------------

def compute_curve_stats(
    pesq_results: dict,
    sisdr_results: dict,
    baseline_pesq: pd.Series,
    K_list: list,
) -> pd.DataFrame:
    """Per-(subset, K) PESQ curve stats.  Also computes SI-SDR of PESQ-oracle-selected output.

    Columns:
        subset, K, n,
        mean_best_pesq, mean_mean_pesq, mean_improvement_best_pesq,
        ci_low, ci_high,
        mean_sisdr_of_pesq_oracle   – SI-SDR of the argmax-PESQ-selected try
    """
    _subset_seed = {"worst": 1, "random": 2}
    rows = []
    for subset_name, file_pesq in pesq_results.items():
        file_sdr = sisdr_results.get(subset_name, {})
        for K in sorted(K_list):
            abs_best_pesq, abs_mean_pesq, baselines_pesq = [], [], []
            sisdr_of_oracle = []

            for fname, try_pesq in file_pesq.items():
                vals = [try_pesq[t] for t in range(K)
                        if t in try_pesq and not np.isnan(try_pesq[t])]
                if not vals:
                    continue
                base = float(baseline_pesq.get(fname, float("nan")))
                if np.isnan(base):
                    continue
                abs_best_pesq.append(float(np.max(vals)))
                abs_mean_pesq.append(float(np.mean(vals)))
                baselines_pesq.append(base)

                # SI-SDR at the PESQ-oracle-selected try
                valid_pesq = {t: try_pesq[t] for t in range(K)
                              if t in try_pesq and not np.isnan(try_pesq[t])}
                k_star = max(valid_pesq, key=valid_pesq.__getitem__)
                sdr_map = file_sdr.get(fname, {})
                if k_star in sdr_map and not np.isnan(sdr_map[k_star]):
                    sisdr_of_oracle.append(float(sdr_map[k_star]))

            n = len(abs_best_pesq)
            if n == 0:
                rows.append({
                    "subset": subset_name, "K": K, "n": 0,
                    "mean_best_pesq": float("nan"), "mean_mean_pesq": float("nan"),
                    "mean_improvement_best_pesq": float("nan"),
                    "ci_low": float("nan"), "ci_high": float("nan"),
                    "mean_sisdr_of_pesq_oracle": float("nan"),
                })
                continue

            arr = np.array(abs_best_pesq)
            base_arr = np.array(baselines_pesq)
            local_rng = np.random.default_rng(
                42 + 1000 * _subset_seed.get(subset_name, 0) + K
            )
            ci_low, ci_high = _bootstrap_ci(arr, rng=local_rng)
            rows.append({
                "subset": subset_name, "K": K, "n": n,
                "mean_best_pesq": float(arr.mean()),
                "mean_mean_pesq": float(np.mean(abs_mean_pesq)),
                "mean_improvement_best_pesq": float((arr - base_arr).mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "mean_sisdr_of_pesq_oracle": float(np.mean(sisdr_of_oracle)) if sisdr_of_oracle else float("nan"),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fit diagnostics
# ---------------------------------------------------------------------------

def fit_diagnostics(curve_df: pd.DataFrame) -> str:
    lines = ["=== Best-of-K Curve Fit (PESQ) ===", ""]
    for subset_name in sorted(curve_df["subset"].unique()):
        sub = (curve_df[curve_df["subset"] == subset_name]
               .dropna(subset=["mean_best_pesq"])
               .query("K >= 1")
               .sort_values("K"))
        if len(sub) < 3:
            lines.append(f"Subset: {subset_name} — too few K points ({len(sub)}) for fitting.")
            lines.append("")
            continue
        K_vals = sub["K"].values.astype(float)
        y = sub["mean_best_pesq"].values
        lines.append(f"Subset: {subset_name}  (n_K={len(sub)}, K={list(K_vals.astype(int))})")
        for label, x_feat in [
            ("log(K)",       np.log(K_vals)),
            ("sqrt(log(K))", np.sqrt(np.log(K_vals))),
        ]:
            slope, intercept, r2 = _linregress(x_feat, y)
            lines.append(f"  PESQ ~ a·{label} + b  →  a={slope:.4f}, b={intercept:.4f}, R²={r2:.4f}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def log_curve_sanity(curve_df: pd.DataFrame) -> None:
    print("\n=== Oracle Curve Sanity Check (PESQ) ===")
    hdr = f"  {'subset':<8}  {'K':>4}  {'n':>6}  {'mean_best_pesq':>16}  {'ci_low':>8}  {'ci_high':>8}"
    sep = "  " + "-" * (len(hdr) - 2)
    for subset_name in sorted(curve_df["subset"].unique()):
        sub = curve_df[curve_df["subset"] == subset_name].sort_values("K")
        print(hdr)
        print(sep)
        for _, row in sub.iterrows():
            def _fmt(v):
                return f"{v:.4f}" if not np.isnan(v) else "       nan"
            print(f"  {subset_name:<8}  {int(row['K']):>4}  {int(row['n']):>6}"
                  f"  {_fmt(row['mean_best_pesq']):>16}"
                  f"  {_fmt(row['ci_low']):>8}"
                  f"  {_fmt(row['ci_high']):>8}")
        print()

        # Monotonicity check
        valid = sub.dropna(subset=["mean_best_pesq"]).sort_values("K")
        mb_vals = valid["mean_best_pesq"].values
        K_vals  = valid["K"].values
        for i in range(1, len(mb_vals)):
            if mb_vals[i] < mb_vals[i - 1]:
                print(f"  WARNING [{subset_name}]: mean_best_pesq not monotone: "
                      f"K={int(K_vals[i-1])}→K={int(K_vals[i])}: "
                      f"{mb_vals[i-1]:.4f}→{mb_vals[i]:.4f}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _use_logx(K_list: list) -> bool:
    return len(K_list) >= 3 and max(K_list) / max(1, min(K_list)) >= 10


def plot_curve(curve_df, metric_col, ylabel, title, save_path, K_list,
               extra_hlines=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    _colors = {}
    for subset_name in sorted(curve_df["subset"].unique()):
        sub = curve_df[curve_df["subset"] == subset_name].sort_values("K")
        (line,) = ax.plot(sub["K"], sub[metric_col], marker="o", label=subset_name)
        _colors[subset_name] = line.get_color()
    if extra_hlines:
        for label, y_val in extra_hlines:
            if not np.isnan(float(y_val)):
                col = _colors.get(label.split(" ")[0])
                ax.axhline(float(y_val), linestyle="--", color=col, label=label, alpha=0.8)
    if _use_logx(K_list):
        ax.set_xscale("log")
    ax.set_xlabel("K (number of tries)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_hist(vals, xlabel, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(vals, bins=40, edgecolor="black", alpha=0.75)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_worst_sweep_curve_pesq(
    summary_df: pd.DataFrame,
    subset_name: str,
    X_pct: list,
    save_path: "Path",
    K: "int | None" = None,
    baseline_all_pesq: "pd.Series | None" = None,
) -> None:
    """Plot mean best_of_K_pesq, mean_of_K_pesq, and baseline_pesq vs worst-x% cutoff.

    baseline_all_pesq: PESQ scores for the FULL test set (not just the oracle subset).
    When provided, x% is a true percentage of the full test set and the x-axis
    naturally stops when the ranked group contains files outside the oracle subset.
    When omitted, ranking and n_total fall back to the oracle subset only.

    Saves both a PNG and a .txt table alongside it.
    """
    sub = summary_df[summary_df["subset"] == subset_name].copy()
    if sub.empty:
        print(f"Warning: no data for subset '{subset_name}', skipping sweep plot.")
        return

    # Compute per-file best / mean for the requested K from try columns
    if K is not None:
        try_cols = sorted(
            [c for c in sub.columns if c.startswith("try_") and c.endswith("_pesq")],
            key=lambda c: int(c.split("_")[1]),
        )
        K_actual = min(K, len(try_cols))
        use_cols = try_cols[:K_actual]
        sub["_best"] = sub[use_cols].max(axis=1)
        sub["_mean"] = sub[use_cols].mean(axis=1)
    else:
        K_actual = None
        sub["_best"] = sub["best_of_K_pesq"]
        sub["_mean"] = sub["mean_of_K_pesq"]

    sub_indexed = sub.set_index("filename")

    # Rank by baseline PESQ ascending (worst = lowest).  When the full-test-set
    # series is supplied, n_total covers all files so x% is relative to the full
    # test set; the continue-check below skips any x% whose top-k group extends
    # beyond the oracle subset.
    if baseline_all_pesq is not None:
        all_ranked = baseline_all_pesq.sort_values().index.tolist()
        n_total = len(all_ranked)
    else:
        all_ranked = sub_indexed["baseline_pesq"].sort_values().index.tolist()
        n_total = len(all_ranked)

    x_vals, n_vals, y_best, y_mean, y_base = [], [], [], [], []
    for x in X_pct:
        k = max(1, int(np.ceil(n_total * x / 100)))
        if k > n_total:
            continue
        group_fnames = all_ranked[:k]
        if any(f not in sub_indexed.index for f in group_fnames):
            continue
        x_vals.append(x)
        n_vals.append(k)
        y_best.append(float(sub_indexed.loc[group_fnames, "_best"].mean()))
        y_mean.append(float(sub_indexed.loc[group_fnames, "_mean"].mean()))
        y_base.append(float(sub_indexed.loc[group_fnames, "baseline_pesq"].mean()))

    if not x_vals:
        print(f"Warning: no valid x% points for subset '{subset_name}', skipping sweep plot.")
        return

    K_label = f"K={K_actual}" if K_actual is not None else "K=max"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_vals, y_best, marker="o", label=f"best_of_K ({K_label})")
    ax.plot(x_vals, y_mean, marker="s", label=f"mean_of_K ({K_label})")
    ax.plot(x_vals, y_base, marker="^", linestyle="--", label="baseline")
    ax.set_xlabel("Worst subset size (% of full test set)"
                  if baseline_all_pesq is not None
                  else "Worst subset size (% of oracle subset)")
    ax.set_ylabel("PESQ-WB")
    ax.set_title(f"Oracle resampling — worst-x% by PESQ ({K_label})")
    ax.set_xticks(x_vals)
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")

    # Text table
    txt_path = save_path.with_suffix(".txt")
    col_w = 12
    header = (f"{'worst_%':>{col_w}}  {'n_files':>{col_w}}"
              f"  {'baseline':>{col_w}}  {'best_of_K':>{col_w}}  {'mean_of_K':>{col_w}}")
    sep = "  ".join(["-" * col_w] * 5)
    lines = [f"# worst_sweep_curve (PESQ) — {subset_name}  ({K_label})", header, sep]
    for x, n, yb, ym, ybase in zip(x_vals, n_vals, y_best, y_mean, y_base):
        lines.append(
            f"{x:>{col_w}.1f}  {n:>{col_w}d}"
            f"  {ybase:>{col_w}.4f}  {yb:>{col_w}.4f}  {ym:>{col_w}.4f}"
        )
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {txt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Oracle resampling — PESQ oracle (select best try per file by max PESQ)."
    )
    parser.add_argument("--test_dir",              required=True)
    parser.add_argument("--clean_dir",             required=True)
    parser.add_argument("--baseline_enhanced_dir", required=True)
    parser.add_argument("--work_dir",              required=True)
    parser.add_argument("--q_percent",  type=float, default=20,
                        help="Bottom percentile of baseline PESQ to treat as 'worst' (default: 20)")
    parser.add_argument("--K",          type=int,   default=5,
                        help="Number of tries when --K_list not provided (default: 5)")
    parser.add_argument("--K_list",     type=int,   nargs="+", default=None,
                        help="List of K values, e.g. --K_list 1 2 5 10.  K_max=max(K_list) tries run.")
    parser.add_argument("--seed_base",  type=int,   default=100)
    parser.add_argument("--enhancement_cmd", required=True)
    parser.add_argument("--metrics_cmd",     required=True)
    args = parser.parse_args()

    K_list: list = args.K_list if args.K_list is not None else [args.K]
    K_max = max(K_list)

    work_dir  = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = work_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    test_dir  = Path(args.test_dir)

    print(f"K_list={K_list}, K_max={K_max}")

    # ------------------------------------------------------------------
    # 1. Load baseline PESQ and SI-SDR
    # ------------------------------------------------------------------
    baseline_csv = find_results_csv(Path(args.baseline_enhanced_dir))
    if baseline_csv is None:
        sys.exit(f"Error: no *_results.csv in {args.baseline_enhanced_dir}")

    baseline_df = pd.read_csv(baseline_csv)
    baseline_df["filename"] = baseline_df["filename"].apply(
        lambda f: normalize_fname(f, test_dir))

    if "pesq" not in baseline_df.columns:
        sys.exit("Error: baseline _results.csv has no 'pesq' column. "
                 "Re-run calc_metrics.py on the baseline dir first.")

    baseline_pesq  = baseline_df.set_index("filename")["pesq"]
    baseline_sisdr = baseline_df.set_index("filename")["sisdr_enh"] if "sisdr_enh" in baseline_df.columns \
                     else pd.Series(dtype=float)
    print(f"Loaded baseline: {len(baseline_pesq)} files   "
          f"PESQ mean={baseline_pesq.mean():.3f}   "
          f"SI-SDR mean={baseline_sisdr.mean():.2f}")

    # ------------------------------------------------------------------
    # 2. Worst-q% subset by PESQ (lower PESQ = harder)
    # ------------------------------------------------------------------
    n_select   = max(1, int(np.ceil(len(baseline_pesq) * args.q_percent / 100)))
    worst_files = baseline_pesq.dropna().sort_values().head(n_select).index.tolist()
    worst_txt  = work_dir / "worst.txt"
    worst_txt.write_text("\n".join(worst_files) + "\n")
    print(f"Worst {args.q_percent}% by PESQ: {n_select} files → {worst_txt}")

    # ------------------------------------------------------------------
    # 3. Random subset of equal size
    # ------------------------------------------------------------------
    rng = random.Random(1234)
    random_files = rng.sample(baseline_pesq.dropna().index.tolist(), n_select)
    random_txt   = work_dir / "random.txt"
    random_txt.write_text("\n".join(random_files) + "\n")
    print(f"Random subset:  {n_select} files → {random_txt}")

    # ------------------------------------------------------------------
    # 4. Run enhancement + metrics for each subset × K_max tries
    # ------------------------------------------------------------------
    # pesq_results[subset][fname][try] = pesq_val
    # sisdr_results[subset][fname][try] = sisdr_val
    pesq_results: dict  = {
        "worst":  {f: {} for f in worst_files},
        "random": {f: {} for f in random_files},
    }
    sisdr_results: dict = {
        "worst":  {f: {} for f in worst_files},
        "random": {f: {} for f in random_files},
    }
    seed_offsets = {"worst": 0, "random": 10_000}
    file_lists   = {"worst": worst_txt, "random": random_txt}

    for subset_name, file_list_path in file_lists.items():
        for t in range(K_max):
            enh_dir = work_dir / f"{subset_name}_try_{t}"
            enh_dir.mkdir(parents=True, exist_ok=True)
            seed = args.seed_base + seed_offsets[subset_name] + t

            if not run(f"{args.enhancement_cmd} "
                       f"--enhanced_dir {enh_dir} "
                       f"--file_list {file_list_path} "
                       f"--gate_seed {seed}"):
                continue

            if not run(f"{args.metrics_cmd} --enhanced_dir {enh_dir}"):
                continue

            csv_path = find_results_csv(enh_dir)
            if csv_path is None:
                print(f"Warning: no results CSV in {enh_dir}, skipping try {t}")
                continue

            attempt_pesq  = load_metric(csv_path, "pesq")
            attempt_sisdr = load_metric(csv_path, "sisdr_enh")

            for fname in pesq_results[subset_name]:
                nfname = normalize_fname(fname, test_dir)
                if nfname in attempt_pesq.index:
                    pesq_results[subset_name][fname][t]  = float(attempt_pesq[nfname])
                if nfname in attempt_sisdr.index:
                    sisdr_results[subset_name][fname][t] = float(attempt_sisdr[nfname])

    # ------------------------------------------------------------------
    # 5. Per-file oracle summary CSV
    # ------------------------------------------------------------------
    rows = []
    for subset_name, file_pesq in pesq_results.items():
        file_sdr = sisdr_results[subset_name]
        for fname, try_pesq in file_pesq.items():
            pesq_vals = [v for v in try_pesq.values() if not np.isnan(v)]
            if not pesq_vals:
                continue
            # PESQ-oracle selected try
            valid_pesq = {t: v for t, v in try_pesq.items() if not np.isnan(v)}
            k_star = max(valid_pesq, key=valid_pesq.__getitem__)
            sdr_map  = file_sdr.get(fname, {})
            row = {
                "filename":           fname,
                "subset":             subset_name,
                "baseline_pesq":      float(baseline_pesq.get(fname, float("nan"))),
                "baseline_sisdr_enh": float(baseline_sisdr.get(fname, float("nan"))),
                "best_of_K_pesq":     float(np.max(pesq_vals)),
                "mean_of_K_pesq":     float(np.mean(pesq_vals)),
                "pesq_oracle_k":      k_star,
                "sisdr_of_pesq_oracle": float(sdr_map.get(k_star, float("nan"))),
            }
            for t in range(K_max):
                row[f"try_{t}_pesq"]  = try_pesq.get(t, float("nan"))
                row[f"try_{t}_sisdr"] = sdr_map.get(t, float("nan"))
            rows.append(row)

    summary_df   = pd.DataFrame(rows)
    summary_path = work_dir / "oracle_summary_pesq.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nOracle summary (PESQ) → {summary_path}")

    # ------------------------------------------------------------------
    # 6. Absolute summary table
    # ------------------------------------------------------------------
    abs_rows = []
    for subset_name in ["worst", "random"]:
        sub = summary_df[summary_df["subset"] == subset_name]
        if sub.empty:
            continue
        abs_rows.append({
            "subset":                      subset_name,
            "n_files":                     sub["filename"].nunique(),
            "K":                           K_max,
            "mean_baseline_pesq":          round(float(sub["baseline_pesq"].mean()),   4),
            "mean_best_of_K_pesq":         round(float(sub["best_of_K_pesq"].mean()),  4),
            "mean_mean_of_K_pesq":         round(float(sub["mean_of_K_pesq"].mean()),  4),
            "mean_baseline_sisdr_enh":     round(float(sub["baseline_sisdr_enh"].mean(skipna=True)), 4),
            "mean_sisdr_of_pesq_oracle":   round(float(sub["sisdr_of_pesq_oracle"].mean(skipna=True)), 4),
        })

    abs_summary_path = work_dir / "oracle_absolute_summary_pesq.csv"
    pd.DataFrame(abs_rows).to_csv(abs_summary_path, index=False)
    print(f"Absolute summary (PESQ) → {abs_summary_path}")

    print("\n=== Oracle Resampling Results (PESQ-WB, absolute) ===\n")
    hdr = (f"  {'subset':<8}  {'n':>6}  {'K':>4}"
           f"  {'base_PESQ':>10}  {'best_of_K':>10}  {'mean_of_K':>10}"
           f"  {'base_SI-SDR':>12}  {'SI-SDR@oracle':>14}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in abs_rows:
        print(
            f"  {r['subset']:<8}  {r['n_files']:>6}  {r['K']:>4}"
            f"  {r['mean_baseline_pesq']:>10.3f}"
            f"  {r['mean_best_of_K_pesq']:>10.3f}"
            f"  {r['mean_mean_of_K_pesq']:>10.3f}"
            f"  {r['mean_baseline_sisdr_enh']:>12.2f}"
            f"  {r['mean_sisdr_of_pesq_oracle']:>14.2f}"
        )

    # ------------------------------------------------------------------
    # 7. Curve stats (one row per subset × K)
    # ------------------------------------------------------------------
    curve_df   = compute_curve_stats(pesq_results, sisdr_results, baseline_pesq, K_list)
    curve_path = work_dir / "oracle_curve_pesq.csv"
    curve_df.to_csv(curve_path, index=False)
    print(f"\nOracle curve (PESQ) → {curve_path}")
    log_curve_sanity(curve_df)

    # ------------------------------------------------------------------
    # 8. Fit diagnostics
    # ------------------------------------------------------------------
    fit_report = fit_diagnostics(curve_df)
    print("\n" + fit_report)
    fit_path = work_dir / "fit_report_pesq.txt"
    fit_path.write_text(fit_report)
    print(f"Fit report → {fit_path}")

    # ------------------------------------------------------------------
    # 9. Baseline hlines for plots
    # ------------------------------------------------------------------
    hlines_pesq  = []
    hlines_sisdr = []
    for subset_name in ["worst", "random"]:
        sub = summary_df[summary_df["subset"] == subset_name]
        if not sub.empty:
            hlines_pesq.append(
                (f"{subset_name} baseline", float(sub["baseline_pesq"].mean(skipna=True)))
            )
            hlines_sisdr.append(
                (f"{subset_name} baseline", float(sub["baseline_sisdr_enh"].mean(skipna=True)))
            )

    # ------------------------------------------------------------------
    # 10. Plots
    # ------------------------------------------------------------------
    plot_curve(
        curve_df, "mean_best_pesq", "Mean Best-of-K PESQ-WB",
        "Best-of-K PESQ Oracle",
        plots_dir / "best_of_K_vs_K.png",
        K_list, extra_hlines=hlines_pesq,
    )
    plot_curve(
        curve_df, "mean_mean_pesq", "Mean Mean-of-K PESQ-WB",
        "Mean-of-K PESQ",
        plots_dir / "mean_of_K_pesq_vs_K.png",
        K_list, extra_hlines=hlines_pesq,
    )
    if curve_df["mean_sisdr_of_pesq_oracle"].notna().any():
        plot_curve(
            curve_df, "mean_sisdr_of_pesq_oracle", "Mean SI-SDR (dB)",
            "SI-SDR of PESQ-Oracle-Selected Output",
            plots_dir / "sisdr_of_pesq_oracle.png",
            K_list, extra_hlines=hlines_sisdr,
        )

    # Per-try PESQ histograms
    for subset_name in ["worst", "random"]:
        pooled = [v for try_map in pesq_results.get(subset_name, {}).values()
                  for v in try_map.values() if not np.isnan(v)]
        if pooled:
            plot_hist(
                pooled, "PESQ-WB",
                f"Pooled per-try PESQ ({subset_name}, n={len(pooled)})",
                plots_dir / f"pesq_hist_{subset_name}.png",
            )

    # Worst-sweep curve (PESQ vs worst-x% of the FULL test set)
    if not summary_df.empty:
        X_pct = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        plot_worst_sweep_curve_pesq(
            summary_df, "worst", X_pct,
            plots_dir / "worst_sweep_curve_worst.png",
            K=K_max,
            baseline_all_pesq=baseline_pesq,
        )

    print("\nDone.")
    print(f"  Summary      → {summary_path}")
    print(f"  Curve        → {curve_path}")
    print(f"  Abs summary  → {abs_summary_path}")
    print(f"  Plots        → {plots_dir}/")


if __name__ == "__main__":
    main()
