#!/usr/bin/env python3
"""Evaluate whether early diffusion-step DNSMOS predicts worst final-step files.

Reads a directory produced by enhancement.py --save_steps_all, which contains:
    enhanced_dir/
        step_000/*.wav
        step_003/*.wav
        ...
        step_029/*.wav

For each step directory the script computes DNSMOS per file, builds a wide
table (dnsmos_by_step.csv), then computes recall/precision/F1/Spearman of the
worst-q% early-step identifications against the worst-q% final-step files.

Usage:
    python dnsmos_early_overlap_eval.py \\
        --enhanced_dir OUT \\
        --work_dir results/dnsmos_overlap \\
        [--prefix step] \\
        [--q_percent 10] \\
        [--resume]
"""

import re
import sys
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

# Add repo root to path so utils/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.dnsmos_helper import compute_dnsmos, is_available as dnsmos_available


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _strip_step_suffix(stem: str, k: int) -> str:
    """Remove _step{k:03d} suffix from a wav stem to recover the original name.

    enhancement.py saves files as {orig_stem}_step{k:03d}.wav, so:
        'p232_001_step003'  →  'p232_001'
    """
    suffix = f"_step{k:03d}"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _compute_dnsmos_for_step(
    step_dir: Path,
    k: int,
    already_scored: set,
    sr: int = 16000,
) -> dict:
    """Return {orig_stem + '.wav': dnsmos_score} for all new wav/flac files in step_dir.

    The canonical key is orig_stem + '.wav' so it is compatible with other
    per-file CSVs in the repo (e.g. baseline_results.csv).
    Files whose canonical key is in already_scored are skipped (resumable).
    """
    wav_files = sorted(step_dir.glob("*.wav")) + sorted(step_dir.glob("*.flac"))
    scores = {}
    for wav_path in tqdm(wav_files, desc=f"  step {k:03d}", leave=False):
        orig_stem = _strip_step_suffix(wav_path.stem, k)
        filename_key = orig_stem + ".wav"
        if filename_key in already_scored:
            continue
        audio, file_sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # mix to mono if needed
        if file_sr != sr:
            from librosa import resample as lr_resample
            audio = lr_resample(audio, orig_sr=file_sr, target_sr=sr)
        score = compute_dnsmos(audio, sr)
        scores[filename_key] = score
    return scores


def _merge_scores_into_df(
    df: pd.DataFrame,
    scores: dict,
    col: str,
) -> pd.DataFrame:
    """Merge a {stem: score} dict into the wide DataFrame."""
    new_df = pd.DataFrame(
        {"filename": list(scores.keys()), col: list(scores.values())}
    )
    if df.empty:
        return new_df

    if col not in df.columns:
        return df.merge(new_df, on="filename", how="outer")

    # Update existing rows
    update_mask = df["filename"].isin(scores)
    df.loc[update_mask, col] = df.loc[update_mask, "filename"].map(scores)

    # Append completely new filenames
    new_filenames = set(scores) - set(df["filename"])
    if new_filenames:
        extra = new_df[new_df["filename"].isin(new_filenames)]
        df = pd.concat([df, extra], ignore_index=True)

    return df


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DNSMOS early-step overlap evaluation."
    )
    parser.add_argument(
        "--enhanced_dir", required=True, type=Path,
        help="Parent directory containing step_### subdirectories",
    )
    parser.add_argument(
        "--work_dir", required=True, type=Path,
        help="Output directory for CSVs",
    )
    parser.add_argument(
        "--prefix", default="step",
        help="Step directory prefix (default: 'step')",
    )
    parser.add_argument(
        "--q_percent", type=float, default=10.0,
        help="Percentile of worst files to define W_t and W_final (default: 10)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Load existing dnsmos_by_step.csv and compute only missing entries",
    )
    args = parser.parse_args()

    # Check DNSMOS availability early
    if not dnsmos_available():
        print("ERROR: speechmos not available. Install with: pip install speechmos")
        sys.exit(1)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    csv_path     = args.work_dir / "dnsmos_by_step.csv"
    overlap_path = args.work_dir / "overlap_curve.csv"

    # ------------------------------------------------------------------
    # 1. Discover step directories
    # ------------------------------------------------------------------
    pattern = re.compile(rf"^{re.escape(args.prefix)}_(\d{{3}})$")
    step_dirs: dict[int, Path] = {}
    for d in sorted(args.enhanced_dir.iterdir()):
        if d.is_dir():
            m = pattern.match(d.name)
            if m:
                step_dirs[int(m.group(1))] = d

    if not step_dirs:
        print(
            f"No directories matching '{args.prefix}_###' found in "
            f"{args.enhanced_dir}"
        )
        sys.exit(1)

    steps = sorted(step_dirs.keys())
    final_step = steps[-1]
    print(f"Found {len(steps)} step dirs: {steps}")
    print(f"Final step: {final_step:03d}")

    # ------------------------------------------------------------------
    # 2. Load or initialise wide table
    # ------------------------------------------------------------------
    if args.resume and csv_path.exists():
        df = pd.read_csv(csv_path, dtype={"filename": str})
        print(f"Loaded existing table: {len(df)} rows, columns: {list(df.columns)}")
    else:
        df = pd.DataFrame(columns=["filename"])

    # ------------------------------------------------------------------
    # 3. Compute DNSMOS per step (resumable)
    # ------------------------------------------------------------------
    for k in steps:
        col = f"dnsmos_step_{k:03d}"
        step_dir = step_dirs[k]

        # Determine which stems already have scores
        already_scored: set = set()
        if col in df.columns:
            already_scored = set(df.loc[df[col].notna(), "filename"].tolist())
            n_total = len(
                list(step_dir.glob("*.wav")) + list(step_dir.glob("*.flac"))
            )
            if len(already_scored) >= n_total:
                print(f"Step {k:03d}: all {n_total} files present — skipping.")
                continue
            print(
                f"Step {k:03d}: {len(already_scored)}/{n_total} done, "
                f"computing rest …"
            )
        else:
            print(f"Step {k:03d}: computing DNSMOS in {step_dir} …")

        new_scores = _compute_dnsmos_for_step(step_dir, k, already_scored)

        if not new_scores:
            print(f"Step {k:03d}: nothing new to compute.")
            continue

        df = _merge_scores_into_df(df, new_scores, col)
        df.to_csv(csv_path, index=False)
        print(f"  Saved {len(df)} rows → {csv_path}")

    if df.empty:
        print("No data collected. Exiting.")
        sys.exit(1)

    # Ensure every step column exists (fill missing with NaN)
    for k in steps:
        col = f"dnsmos_step_{k:03d}"
        if col not in df.columns:
            df[col] = float("nan")
    df.to_csv(csv_path, index=False)

    # ------------------------------------------------------------------
    # 4 & 5. Overlap metrics and Spearman
    # ------------------------------------------------------------------
    from scipy.stats import spearmanr

    final_col = f"dnsmos_step_{final_step:03d}"
    valid_final = df[["filename", final_col]].dropna()

    print(
        f"\nFinal step {final_step:03d}: {len(valid_final)} files with scores. "
        f"W_final and W_t computed on per-step common set at q={args.q_percent}%."
    )
    print(
        f"\n{'step':>6}  {'n':>5}  {'recall':>7}  {'prec':>7}  "
        f"{'f1':>7}  {'spearman':>9}"
    )
    print("-" * 57)

    records = []
    for k in steps:
        col = f"dnsmos_step_{k:03d}"

        # Intersection of files with non-NaN scores at both step k and final
        valid_k   = df[["filename", col]].dropna()
        common    = pd.merge(valid_k, valid_final, on="filename")
        n_common  = len(common)

        if n_common == 0:
            records.append(
                dict(step=k, n=0, recall=float("nan"),
                     precision=float("nan"), f1=float("nan"),
                     spearman=float("nan"))
            )
            print(f"  {k:03d}   {0:5d}  {'—':>7}  {'—':>7}  {'—':>7}  {'—':>9}")
            continue

        # Both W_t and W_final are defined within common (equal-population comparison).
        # ceil ensures at least 1 file even for tiny common sets.
        n_worst_common   = max(1, int(np.ceil(n_common * args.q_percent / 100)))
        worst_k          = set(common.nsmallest(n_worst_common, col)["filename"])
        worst_final_here = set(common.nsmallest(n_worst_common, final_col)["filename"])

        intersect = len(worst_k & worst_final_here)
        recall    = intersect / len(worst_final_here) if worst_final_here else float("nan")
        precision = intersect / len(worst_k)          if worst_k         else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else float("nan")
        )

        sp_r, _ = spearmanr(common[col].values, common[final_col].values)

        records.append(
            dict(
                step=k, n=n_common,
                recall=round(recall, 4),
                precision=round(precision, 4),
                f1=round(f1, 4),
                spearman=round(float(sp_r), 4),
            )
        )
        print(
            f"  {k:03d}   {n_common:5d}  {recall:7.3f}  {precision:7.3f}  "
            f"{f1:7.3f}  {sp_r:9.4f}"
        )

    overlap_df = pd.DataFrame(records)
    overlap_df.to_csv(overlap_path, index=False)

    print(f"\nOverlap curve  → {overlap_path}")
    print(f"DNSMOS table   → {csv_path}")


if __name__ == "__main__":
    main()
