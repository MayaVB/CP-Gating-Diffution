"""Oracle resampling experiment.

For each subset (worst-q% and random), run K independent enhancement attempts
and compute oracle (best-of-K) and mean-of-K SI-SDR improvement over baseline.

Does NOT modify enhancement.py or calc_metrics.py — calls them via subprocess.

Usage example:
    python oracle_resample_eval.py \
        --test_dir voicebank/test_noisy \
        --clean_dir voicebank/test_clean \
        --baseline_enhanced_dir voicebank/test_enhanced_baseline \
        --work_dir voicebank/oracle_exp \
        --q_percent 20 \
        --K 5 \
        --seed_base 100 \
        --enhancement_cmd "python enhancement.py --test_dir voicebank/test_noisy --ckpt checkpoints_updated/train_vb_29nqe0uh_epoch=115.ckpt" \
        --metrics_cmd "python calc_metrics.py --clean_dir voicebank/test_clean --noisy_dir voicebank/test_noisy"
"""

import argparse
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def find_results_csv(directory: Path) -> Path:
    matches = list(directory.glob("*_results.csv"))
    if not matches:
        return None  # caller handles missing CSV
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
    return df.set_index("filename")["si_sdr"]


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Oracle resampling experiment")
    parser.add_argument("--test_dir",              required=True)
    parser.add_argument("--clean_dir",             required=True)
    parser.add_argument("--baseline_enhanced_dir", required=True)
    parser.add_argument("--work_dir",              required=True)
    parser.add_argument("--q_percent",  type=float, default=20,
                        help="Bottom percentile of baseline SI-SDR to treat as 'worst' (default: 20)")
    parser.add_argument("--K",          type=int,   default=5,
                        help="Number of resampling tries per file (default: 5)")
    parser.add_argument("--seed_base",  type=int,   default=100,
                        help="Base random seed; try t uses seed_base + t (default: 100)")
    parser.add_argument("--enhancement_cmd", required=True,
                        help="Full enhancement.py command except --enhanced_dir / --file_list / --gate_seed")
    parser.add_argument("--metrics_cmd", required=True,
                        help="Full calc_metrics.py command except --enhanced_dir")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    test_dir     = Path(args.test_dir)
    baseline_dir = Path(args.baseline_enhanced_dir)

    # ── 1. Load baseline CSV ──────────────────────────────────────────────────
    baseline_csv = find_results_csv(baseline_dir)
    if baseline_csv is None:
        sys.exit(f"Error: no *_results.csv found in {baseline_dir}")
    baseline_df  = pd.read_csv(baseline_csv)
    if "filename" not in baseline_df.columns or "si_sdr" not in baseline_df.columns:
        sys.exit("Error: baseline CSV must contain 'filename' and 'si_sdr' columns")
    # Normalize filenames so they match file_list entries (relative to test_dir)
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

    # ── 4. Enhancement + metrics for each subset × K tries ───────────────────
    # results[subset][fname][try_idx] = si_sdr  (missing try → absent key)
    results: dict[str, dict[str, dict[int, float]]] = {
        "worst":  {f: {} for f in worst_files},
        "random": {f: {} for f in random_files},
    }
    # Separate seed streams so worst and random draws never collide
    seed_offsets = {"worst": 0, "random": 10000}
    file_lists   = {"worst": worst_txt, "random": random_txt}

    for subset_name, file_list_path in file_lists.items():
        for t in range(args.K):
            enh_dir = work_dir / f"{subset_name}_try_{t}"
            enh_dir.mkdir(parents=True, exist_ok=True)
            seed = args.seed_base + seed_offsets[subset_name] + t

            # Run enhancement (non-fatal: skip this try on failure)
            if not run(f"{args.enhancement_cmd} "
                       f"--enhanced_dir {enh_dir} "
                       f"--file_list {file_list_path} "
                       f"--gate_seed {seed}"):
                continue

            # Run metrics (non-fatal)
            if not run(f"{args.metrics_cmd} --enhanced_dir {enh_dir}"):
                continue

            # Load per-utterance SI-SDR from this attempt
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

    # ── 5. Aggregate ─────────────────────────────────────────────────────────
    rows = []
    for subset_name, file_scores in results.items():
        for fname, try_map in file_scores.items():
            score_vals = [v for v in try_map.values()]
            if not score_vals:
                continue
            base = float(baseline_sisdr.get(fname, float("nan")))
            best = float(np.max(score_vals))
            mean = float(np.mean(score_vals))
            row = {
                "filename":         fname,
                "subset":           subset_name,
                "baseline_si_sdr":  base,
                "best_of_K":        best,
                "mean_of_K":        mean,
                "improvement_best": best - base,
                "improvement_mean": mean - base,
            }
            # Per-try SI-SDR columns (NaN when that try failed/was skipped)
            for t in range(args.K):
                row[f"try_{t}"] = try_map.get(t, float("nan"))
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_path = work_dir / "oracle_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nOracle summary saved to {summary_path}")

    # ── 6. Print summary stats ────────────────────────────────────────────────
    print("\n=== Oracle Resampling Results ===")
    for subset_name in ["worst", "random"]:
        sub = summary_df[summary_df["subset"] == subset_name]
        if sub.empty:
            continue
        imp  = sub["improvement_best"]
        p_gt0 = (imp > 0).mean()
        print(f"\n  Subset: {subset_name}  (n={len(sub)}, K={args.K})")
        print(f"  mean improvement_best : {imp.mean():.2f} dB")
        print(f"  mean improvement_mean : {sub['improvement_mean'].mean():.2f} dB")
        print(f"  P(improvement_best>0) : {p_gt0:.1%}")


if __name__ == "__main__":
    main()
