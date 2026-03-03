"""Oracle resampling experiment.

For each subset (worst-q% and random), run K independent enhancement attempts
and compute oracle (best-of-K) and mean-of-K absolute SI-SDR (enhanced vs clean).

Does NOT modify enhancement.py or calc_metrics.py — calls them via subprocess.

Usage example:
    python oracle_resample_eval.py \
        --test_dir voicebank/test_noisy \
        --clean_dir voicebank/test_clean \
        --baseline_enhanced_dir voicebank/test_enhanced_baseline \
        --work_dir voicebank/oracle_exp \
        --q_percent 20 \
        --K_list 1 2 3 5 10 \
        --seed_base 100 \
        --enhancement_cmd "python enhancement.py --test_dir voicebank/test_noisy --ckpt checkpoints_updated/train_vb_29nqe0uh_epoch=115.ckpt" \
        --metrics_cmd "python calc_metrics.py --clean_dir voicebank/test_clean --noisy_dir voicebank/test_noisy"

Outputs (all under --work_dir):
  oracle_summary.csv           – per-file best/mean SI-SDR for K_max tries
  oracle_absolute_summary.csv  – per-subset mean stats (subset, n_files, K, mean_baseline_sisdr_enh, mean_best_of_K, mean_mean_of_K)
  oracle_curve.csv             – (subset, K, n, mean_best_sisdr, mean_mean_sisdr, ci_low, ci_high)
  fit_report.txt               – linear regression of mean_best_sisdr vs log(K)/sqrt(log(K))
  plots/
    best_of_K_vs_K.png        – mean(best_of_K − baseline) vs K (improvement over single-sample run)
    sisdr_hist_{subset}.png           – pooled per-try SI-SDR (n_files × K_max points)
    worst_sweep_curve_worst.png       – best/mean/baseline SI-SDR vs worst-x% of FULL test set (K=K_max)
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


# ── helpers ───────────────────────────────────────────────────────────────────

def find_results_csv(directory: Path) -> Path:
    matches = list(directory.glob("*_results.csv"))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"Warning: multiple *_results.csv in {directory}; using {matches[0].name}")
    return matches[0]


def normalize_fname(fname: str, test_dir: Path) -> str:
    """Return fname as a clean relative path with respect to test_dir."""
    fname = fname.strip().replace("\\", "/")
    prefix = str(test_dir).replace("\\", "/").rstrip("/") + "/"
    if fname.startswith(prefix):
        fname = fname[len(prefix):]
    fname = fname.lstrip("/")
    return fname


def run(cmd: str) -> bool:
    """Run shell command; return True on success, False on CalledProcessError."""
    print(f"\n$ {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Warning: command failed (exit {e.returncode}), skipping this try.")
        return False


def load_sisdr(results_csv: Path) -> pd.Series:
    """Return a filename→si_sdr Series from a _results.csv."""
    df = pd.read_csv(results_csv)
    return df.set_index("filename")["sisdr_enh"]


def _linregress(x, y):
    """OLS regression; returns (slope, intercept, r_squared)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    ss_yy = np.sum((y - y_mean) ** 2)
    slope = ss_xy / ss_xx if ss_xx != 0 else 0.0
    intercept = y_mean - slope * x_mean
    r2 = (ss_xy ** 2 / (ss_xx * ss_yy)) if (ss_xx * ss_yy) > 0 else 0.0
    return slope, intercept, r2


def _bootstrap_ci(
    arr: np.ndarray,
    n_resamples: int = 100,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """95% bootstrap CI for the mean of arr via the percentile method (numpy only)."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(arr)
    if n == 1:
        return float(arr[0]), float(arr[0])
    boot_means = rng.choice(arr, size=(n_resamples, n), replace=True).mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


# ── curve aggregation ─────────────────────────────────────────────────────────

def compute_curve_stats(
    results: dict,
    baseline_sisdr: pd.Series,
    K_list: list[int],
) -> pd.DataFrame:
    """For each (subset, K) pair, compute aggregate absolute SI-SDR stats.

    Only tries 0..K-1 are considered for a given K.  Files with no valid try
    or no baseline value are excluded (counted in the returned n).
    Bootstrap 95% CI (100 resamples, seed 42) is added as ci_low / ci_high.

    Columns: subset, K, n, mean_best_sisdr, mean_mean_sisdr, mean_improvement_best, ci_low, ci_high
      mean_best_sisdr       — mean over files of max(sisdr_enh across K tries)
      mean_mean_sisdr       — mean over files of mean(sisdr_enh across K tries)
      mean_improvement_best — mean over files of (best_of_K − baseline_sisdr_enh)
    """
    # Fixed mapping avoids Python's randomised hash() across processes/versions.
    _subset_seed = {"worst": 1, "random": 2}
    rows = []
    for subset_name, file_scores in results.items():
        for K in sorted(K_list):
            abs_best, abs_mean, baselines = [], [], []
            for fname, try_map in file_scores.items():
                vals = [
                    try_map[t] for t in range(K)
                    if t in try_map and not np.isnan(try_map[t])
                ]
                if not vals:
                    continue
                base = float(baseline_sisdr.get(fname, float("nan")))
                if np.isnan(base):
                    continue
                abs_best.append(float(np.max(vals)))
                abs_mean.append(float(np.mean(vals)))
                baselines.append(base)

            n = len(abs_best)
            if n == 0:
                rows.append({
                    "subset": subset_name, "K": K, "n": 0,
                    "mean_best_sisdr": float("nan"),
                    "mean_mean_sisdr": float("nan"),
                    "mean_improvement_best": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                })
            else:
                arr = np.array(abs_best)
                base_arr = np.array(baselines)
                local_rng = np.random.default_rng(
                    42 + 1000 * _subset_seed.get(subset_name, 0) + K
                )
                ci_low, ci_high = _bootstrap_ci(arr, rng=local_rng)
                rows.append({
                    "subset": subset_name, "K": K, "n": n,
                    "mean_best_sisdr": float(arr.mean()),
                    "mean_mean_sisdr": float(np.mean(abs_mean)),
                    "mean_improvement_best": float((arr - base_arr).mean()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                })
    return pd.DataFrame(rows)


def collect_sisdr(results: dict) -> dict[str, list[float]]:
    """Pool all per-try absolute SI-SDR (enhanced vs clean) for each subset."""
    sisdr_vals: dict[str, list[float]] = {}
    for subset_name, file_scores in results.items():
        d = []
        for try_map in file_scores.values():
            for val in try_map.values():
                if not np.isnan(val):
                    d.append(val)
        sisdr_vals[subset_name] = d
    return sisdr_vals


# ── sanity logging ────────────────────────────────────────────────────────────

def log_curve_sanity(curve_df: pd.DataFrame) -> None:
    """Print a per-(subset, K) stats table and run monotonicity / n-consistency checks."""
    print("\n=== Oracle Curve Sanity Check ===")

    hdr = f"  {'subset':<8}  {'K':>4}  {'n':>6}  {'mean_best':>10}  {'mean_mean':>10}  {'ci_low':>8}  {'ci_high':>8}"
    sep = "  " + "-" * (len(hdr) - 2)

    for subset_name in sorted(curve_df["subset"].unique()):
        sub = curve_df[curve_df["subset"] == subset_name].sort_values("K")
        print(hdr)
        print(sep)
        for _, row in sub.iterrows():
            def _fmt(v):
                return f"{v:+.3f}" if not np.isnan(v) else "      nan"
            print(
                f"  {subset_name:<8}  {int(row['K']):>4}  {int(row['n']):>6}"
                f"  {_fmt(row['mean_best_sisdr']):>10}"
                f"  {_fmt(row['mean_mean_sisdr']):>10}"
                f"  {_fmt(row['ci_low']):>8}"
                f"  {_fmt(row['ci_high']):>8}"
            )
        print()

        # ── monotonicity check ────────────────────────────────────────────────
        valid = sub.dropna(subset=["mean_best_sisdr"]).sort_values("K")
        mb_vals = valid["mean_best_sisdr"].values
        K_vals  = valid["K"].values
        for i in range(1, len(mb_vals)):
            if mb_vals[i] < mb_vals[i - 1]:
                print(
                    f"  WARNING [{subset_name}]: mean_best_sisdr not monotone: "
                    f"K={int(K_vals[i-1])}→K={int(K_vals[i])}: "
                    f"{mb_vals[i-1]:+.3f}→{mb_vals[i]:+.3f} dB"
                )

        # ── n monotonicity check ──────────────────────────────────────────────
        # n(K) must be non-decreasing: more tries means more files can have ≥1 valid try.
        # A decrease is unexpected and signals inconsistent results; an increase is normal.
        n_vals = sub["n"].values
        K_sorted = sub["K"].values
        for i in range(1, len(n_vals)):
            k_prev, n_prev = int(K_sorted[i - 1]), int(n_vals[i - 1])
            k_curr, n_curr = int(K_sorted[i]),     int(n_vals[i])
            if n_curr < n_prev:
                print(
                    f"  WARNING [{subset_name}]: n decreased from "
                    f"K={k_prev} (n={n_prev}) to K={k_curr} (n={n_curr}). "
                    f"Some tries may have produced inconsistent results."
                )
            elif n_curr > n_prev:
                print(
                    f"  INFO [{subset_name}]: n increased from "
                    f"K={k_prev} (n={n_prev}) to K={k_curr} (n={n_curr}); "
                    f"earlier tries had NaNs for some files."
                )


# ── fit diagnostics ───────────────────────────────────────────────────────────

def fit_diagnostics(curve_df: pd.DataFrame) -> str:
    """Fit y(K) = mean_best_sisdr vs log(K) and sqrt(log(K)) bases.

    Requires at least 3 distinct K values with valid data per subset.
    Returns a formatted report string.
    """
    lines = ["=== Best-of-K Curve Fit Diagnostics ===", ""]
    for subset_name in sorted(curve_df["subset"].unique()):
        sub = (
            curve_df[curve_df["subset"] == subset_name]
            .dropna(subset=["mean_best_sisdr"])
            .query("K >= 1")
            .sort_values("K")
        )
        if len(sub) < 3:
            lines.append(f"Subset: {subset_name} — too few valid K points ({len(sub)}) for fitting.")
            lines.append("")
            continue

        K_vals = sub["K"].values.astype(float)
        y = sub["mean_best_sisdr"].values
        lines.append(f"Subset: {subset_name}  (n_K={len(sub)}, K={list(K_vals.astype(int))})")

        for label, x_feat in [
            ("log(K)",       np.log(K_vals)),
            ("sqrt(log(K))", np.sqrt(np.log(K_vals))),
        ]:
            slope, intercept, r2 = _linregress(x_feat, y)
            lines.append(f"  y ~ a·{label} + b  →  a={slope:.4f}, b={intercept:.4f}, R²={r2:.4f}")
        lines.append("")

    return "\n".join(lines)


# ── plotting ──────────────────────────────────────────────────────────────────

def _use_logx(K_list: list[int]) -> bool:
    """Use log x-scale when K values span more than one decade."""
    return len(K_list) >= 3 and max(K_list) / max(1, min(K_list)) >= 10


def plot_curve(
    curve_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    save_path: Path,
    K_list: list[int],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for subset_name in sorted(curve_df["subset"].unique()):
        sub = curve_df[curve_df["subset"] == subset_name].sort_values("K")
        ax.plot(sub["K"], sub[metric_col], marker="o", label=subset_name)
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


def plot_sisdr_hist(sisdr_vals: list[float], subset_name: str, save_path: Path) -> None:
    """Histogram of per-try absolute SI-SDR (enhanced vs clean) for one subset.

    Each enhanced file contributes K points (one per try).  Pooled over all
    files in the subset → n_files × K_max data points total.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(sisdr_vals, bins=50, edgecolor="black", alpha=0.75)
    ax.set_xlabel("SI-SDR (enhanced vs clean, dB)")
    ax.set_ylabel("Count")
    ax.set_title(f"Pooled per-try SI-SDR distribution ({subset_name}, n={len(sisdr_vals)})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")



def plot_worst_sweep_curve(
    summary_df: pd.DataFrame,
    subset_name: str,
    X_pct: list[float],
    save_path: Path,
    baseline_all_sisdr: pd.Series | None = None,
    K: int | None = None,
) -> None:
    """Plot mean best_of_K, mean_of_K, and baseline SI-SDR vs worst-x% cutoff.

    If baseline_all_sisdr is provided (full test-set baseline), x% is taken
    relative to ALL test files ranked by baseline SI-SDR ascending.  Only x
    values for which every selected file has resampling data are plotted, so
    the curve naturally extends up to q_percent.

    If K is provided, best_of_K and mean_of_K are computed from try_0..try_{K-1}
    columns (capped at available tries).  Otherwise the precomputed best_of_K /
    mean_of_K columns (= K_max) are used.
    """
    sub = summary_df[summary_df["subset"] == subset_name].copy()
    if sub.empty:
        print(f"Warning: no data for subset '{subset_name}', skipping sweep plot.")
        return

    # Compute per-file best / mean for the requested K
    if K is not None:
        try_cols = sorted([c for c in sub.columns if c.startswith("try_")],
                          key=lambda c: int(c.split("_")[1]))
        K_actual = min(K, len(try_cols))
        if K_actual < K:
            print(f"Warning: only {K_actual} tries available, using K={K_actual}")
        use_cols = try_cols[:K_actual]
        sub["_best"] = sub[use_cols].max(axis=1)
        sub["_mean"] = sub[use_cols].mean(axis=1)
    else:
        K_actual = None
        sub["_best"] = sub["best_of_K"]
        sub["_mean"] = sub["mean_of_K"]

    sub_indexed = sub.set_index("filename")

    # Determine ranking pool and total N
    if baseline_all_sisdr is not None:
        all_ranked = baseline_all_sisdr.sort_values().index.tolist()
        n_total = len(all_ranked)
    else:
        all_ranked = sub_indexed["baseline_sisdr_enh"].sort_values().index.tolist()
        n_total = len(all_ranked)

    x_vals, n_vals, y_best, y_mean, y_base = [], [], [], [], []
    for x in X_pct:
        k = max(1, int(np.ceil(n_total * x / 100)))
        if k > n_total:
            continue
        group_fnames = all_ranked[:k]
        # Skip this x if any file in the group lacks resampling data
        if any(f not in sub_indexed.index for f in group_fnames):
            continue
        group_best = sub_indexed.loc[group_fnames, "_best"]
        group_mean = sub_indexed.loc[group_fnames, "_mean"]
        if baseline_all_sisdr is not None:
            group_base = baseline_all_sisdr[group_fnames]
        else:
            group_base = sub_indexed.loc[group_fnames, "baseline_sisdr_enh"]
        x_vals.append(x)
        n_vals.append(k)
        y_best.append(float(group_best.mean()))
        y_mean.append(float(group_mean.mean()))
        y_base.append(float(group_base.mean()))

    if not x_vals:
        print(f"Warning: no valid x% points for subset '{subset_name}', skipping sweep plot.")
        return

    K_label = f"K={K_actual}" if K_actual is not None else "K=max"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_vals, y_best, marker="o", label=f"best_of_K ({K_label})")
    ax.plot(x_vals, y_mean, marker="s", label=f"mean_of_K ({K_label})")
    ax.plot(x_vals, y_base, marker="^", linestyle="--", label="baseline")
    ax.set_xlabel("Worst subset size (% of full test set)")
    ax.set_ylabel("SI-SDR (enhanced vs clean, dB)")
    ax.set_title(f"Oracle resampling — worst-x% of test set ({K_label})")
    ax.set_xticks(x_vals)
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")

    # Save the same data as a text table
    txt_path = save_path.with_suffix(".txt")
    col_w = 12
    header = (f"{'worst_%':>{col_w}}  {'n_files':>{col_w}}"
              f"  {'baseline':>{col_w}}  {'best_of_K':>{col_w}}  {'mean_of_K':>{col_w}}")
    sep = "  ".join(["-" * col_w] * 5)
    lines = [f"# worst_sweep_curve — {subset_name}  ({K_label})", header, sep]
    for x, n, yb, ym, ybase in zip(x_vals, n_vals, y_best, y_mean, y_base):
        lines.append(
            f"{x:>{col_w}.1f}  {n:>{col_w}d}"
            f"  {ybase:>{col_w}.4f}  {yb:>{col_w}.4f}  {ym:>{col_w}.4f}"
        )
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {txt_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Oracle resampling experiment")
    parser.add_argument("--test_dir",              required=True)
    parser.add_argument("--clean_dir",             required=True)
    parser.add_argument("--baseline_enhanced_dir", required=True)
    parser.add_argument("--work_dir",              required=True)
    parser.add_argument("--q_percent",  type=float, default=20,
                        help="Bottom percentile of baseline SI-SDR to treat as 'worst' (default: 20)")
    parser.add_argument("--K",          type=int,   default=5,
                        help="Number of tries when --K_list is not provided (default: 5)")
    parser.add_argument("--K_list",     type=int,   nargs="+", default=None,
                        help="List of K values to evaluate, e.g. --K_list 1 2 3 5 10. "
                             "K_max = max(K_list) tries are run. If omitted, uses [--K].")
    parser.add_argument("--seed_base",  type=int,   default=100,
                        help="Base random seed; try t uses seed_base + t (default: 100)")
    parser.add_argument("--enhancement_cmd", required=True,
                        help="Full enhancement.py command (without --enhanced_dir / --file_list / --gate_seed)")
    parser.add_argument("--metrics_cmd", required=True,
                        help="Full calc_metrics.py command (without --enhanced_dir)")
    args = parser.parse_args()

    # Resolve K_list and K_max
    K_list: list[int] = args.K_list if args.K_list is not None else [args.K]
    K_max = max(K_list)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = work_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    test_dir     = Path(args.test_dir)
    baseline_dir = Path(args.baseline_enhanced_dir)

    print(f"K_list={K_list}, K_max={K_max}")

    # ── 1. Load baseline CSV ──────────────────────────────────────────────────
    baseline_csv = find_results_csv(baseline_dir)
    if baseline_csv is None:
        sys.exit(f"Error: no *_results.csv found in {baseline_dir}")
    baseline_df = pd.read_csv(baseline_csv)
    if "filename" not in baseline_df.columns or "si_sdr" not in baseline_df.columns:
        sys.exit("Error: baseline CSV must contain 'filename' and 'si_sdr' columns")
    baseline_df["filename"] = baseline_df["filename"].apply(
        lambda f: normalize_fname(f, test_dir))
    baseline_sisdr = baseline_df.set_index("filename")["si_sdr"]
    print(f"Loaded baseline: {len(baseline_sisdr)} files from {baseline_csv}")

    # ── 2. Worst-q% subset ───────────────────────────────────────────────────
    n_select = max(1, int(np.ceil(len(baseline_sisdr) * args.q_percent / 100)))
    worst_files = baseline_sisdr.sort_values().head(n_select).index.tolist()
    worst_txt = work_dir / "worst.txt"
    worst_txt.write_text("\n".join(worst_files) + "\n")
    print(f"Worst {args.q_percent}%: {n_select} files → {worst_txt}")

    # ── 3. Random subset of equal size ───────────────────────────────────────
    rng = random.Random(1234)
    random_files = rng.sample(baseline_sisdr.index.tolist(), n_select)
    random_txt = work_dir / "random.txt"
    random_txt.write_text("\n".join(random_files) + "\n")
    print(f"Random subset:  {n_select} files → {random_txt}")

    # ── 4. Enhancement + metrics for each subset × K_max tries ───────────────
    # results[subset][fname][try_idx] = si_sdr  (absent when try failed)
    results: dict[str, dict[str, dict[int, float]]] = {
        "worst":  {f: {} for f in worst_files},
        "random": {f: {} for f in random_files},
    }
    # Separate seed offsets so worst/random seed streams never collide
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

            attempt_sisdr = load_sisdr(csv_path)
            for fname in results[subset_name]:
                if fname in attempt_sisdr.index:
                    results[subset_name][fname][t] = float(attempt_sisdr[fname])
                else:
                    print(f"Warning: {fname} missing from {enh_dir} results")

    # ── 5. Oracle summary CSV (one row per file × subset) ────────────────────
    # Columns: filename, subset, baseline_sisdr_enh, best_of_K, mean_of_K, try_0..K-1
    rows = []
    for subset_name, file_scores in results.items():
        for fname, try_map in file_scores.items():
            vals = [v for v in try_map.values() if not np.isnan(v)]
            if not vals:
                continue
            base = float(baseline_sisdr.get(fname, float("nan")))
            row = {
                "filename":           fname,
                "subset":             subset_name,
                "baseline_sisdr_enh": base,
                "best_of_K":          float(np.max(vals)),
                "mean_of_K":          float(np.mean(vals)),
            }
            for t in range(K_max):
                row[f"try_{t}"] = try_map.get(t, float("nan"))
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_path = work_dir / "oracle_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nOracle summary saved to {summary_path}")

    # ── 6. Absolute SI-SDR summary table (console + CSV) ─────────────────────
    _K_inferred = sum(1 for c in summary_df.columns if c.startswith("try_"))
    abs_summary_rows = []
    for subset_name in ["worst", "random"]:
        sub = summary_df[summary_df["subset"] == subset_name]
        if sub.empty:
            continue
        abs_summary_rows.append({
            "subset":                  subset_name,
            "n_files":                 sub["filename"].nunique(),
            "K":                       _K_inferred,
            "mean_baseline_sisdr_enh": round(float(sub["baseline_sisdr_enh"].mean()), 4),
            "mean_best_of_K":          round(float(sub["best_of_K"].mean()), 4),
            "mean_mean_of_K":          round(float(sub["mean_of_K"].mean()), 4),
        })
    abs_summary_path = work_dir / "oracle_absolute_summary.csv"
    pd.DataFrame(abs_summary_rows).to_csv(abs_summary_path, index=False)
    print(f"\nAbsolute summary saved to {abs_summary_path}")

    # pretty-print as aligned table
    print("\n=== Oracle Resampling Results (Absolute SI-SDR) ===\n")
    _hdr = f"  {'subset':<8}  {'n':>6}  {'K':>4}  {'baseline (dB)':>14}  {'best_of_K (dB)':>15}  {'mean_of_K (dB)':>15}"
    print(_hdr)
    print("  " + "-" * (len(_hdr) - 2))
    for r in abs_summary_rows:
        print(
            f"  {r['subset']:<8}  {r['n_files']:>6}  {r['K']:>4}"
            f"  {r['mean_baseline_sisdr_enh']:>14.2f}"
            f"  {r['mean_best_of_K']:>15.2f}"
            f"  {r['mean_mean_of_K']:>15.2f}"
        )

    # ── 7. Oracle curve stats (one row per subset × K) ────────────────────────
    curve_df = compute_curve_stats(results, baseline_sisdr, K_list)
    curve_path = work_dir / "oracle_curve.csv"
    curve_df.to_csv(curve_path, index=False)
    print(f"\nOracle curve saved to {curve_path}")
    log_curve_sanity(curve_df)

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    if len(K_list) >= 2:
        plot_curve(
            curve_df, "mean_improvement_best",
            "Mean best-of-K improvement over single-sample baseline (dB)",
            "Oracle gain: best-of-K SI-SDR(enh,clean) − SI-SDR(baseline,clean)",
            plots_dir / "best_of_K_vs_K.png", K_list,
        )
    else:
        print("Skipping curve plots (need at least 2 K values).")

    all_sisdr = collect_sisdr(results)
    for subset_name, sisdr_vals in all_sisdr.items():
        if not sisdr_vals:
            print(f"Warning: no sisdr values for {subset_name}, skipping histogram.")
            continue
        plot_sisdr_hist(sisdr_vals, subset_name, plots_dir / f"sisdr_hist_{subset_name}.png")

    _X_pct = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    plot_worst_sweep_curve(
        summary_df, "worst", _X_pct,
        plots_dir / "worst_sweep_curve_worst.png",
        baseline_all_sisdr=baseline_sisdr,
        K=K_max,
    )

    # ── 9. Fit diagnostics ────────────────────────────────────────────────────
    if len(K_list) >= 3:
        report = fit_diagnostics(curve_df)
        print(f"\n{report}")
        fit_path = work_dir / "fit_report.txt"
        fit_path.write_text(report)
        print(f"Fit report saved to {fit_path}")
    else:
        print("Skipping fit diagnostics (need at least 3 K values).")


if __name__ == "__main__":
    main()
