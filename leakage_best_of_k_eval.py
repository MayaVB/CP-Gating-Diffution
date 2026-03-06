#!/usr/bin/env python3
"""Best-of-K diffusion sample selection using the leakage gate score.

Uses utils/speech_gate.py compute_speech_gate_score:
    score = 0.6*log(non-speech leakage) + 0.4*speech distortion  ← lower better

For each utterance, picks the try with the LOWEST score,
copies it to work_dir/best_of_k_leak/, then evaluates full metrics
(PESQ / ESTOI / SI-SDR / DNSMOS) comparing try_0 vs selected best-of-K.

Also prints Spearman(score, SI-SDR) and Spearman(score, DNSMOS).
Negative correlation means the gate is predictive of quality.

Expected folder layout:
    tries_root/
        try_0/  *.wav
        try_1/  *.wav
        ...

NOTE: all metrics comparisons are done on the SAME file subset
      (joined on filename) to avoid the cross-dataset mixing bug.

Usage:
    python leakage_best_of_k_eval.py \\
        --tries_root voicebank/oracle_tries \\
        --clean_dir  voicebank/test_clean \\
        --noisy_dir  voicebank/test_noisy \\
        --work_dir   voicebank/exp \\
        --dnsmos
"""

import re
import shutil
import subprocess
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

# repo root on sys.path so utils/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.speech_gate import compute_speech_gate_score
from utils.dnsmos_helper import compute_dnsmos, is_available as dnsmos_available, to_16k
from utils.nisqa_helper import compute_nisqa, is_available as nisqa_available


# ---------------------------------------------------------------------------
# Shared VAD helper  (identical to enhancement.py lines 434-437)
# ---------------------------------------------------------------------------

_VAD_WINDOW = 400  # samples


def _speech_mask(y: np.ndarray, percentile: float = 80.0) -> np.ndarray:
    """Energy-based VAD mask on the noisy signal."""
    energy = np.convolve(y ** 2, np.ones(_VAD_WINDOW) / _VAD_WINDOW, mode="same")
    return energy > np.percentile(energy, percentile)


# ---------------------------------------------------------------------------
# Gate 1: leakage  (reuses utils/speech_gate.py)
# ---------------------------------------------------------------------------

def _leakage_score(y: np.ndarray, x_hat: np.ndarray) -> float:
    """Post-hoc leakage gate score.  Lower = better."""
    mask = _speech_mask(y)
    return compute_speech_gate_score(y, x_hat, mask)

# ---------------------------------------------------------------------------
# Gate 2: stft_leakage
# ---------------------------------------------------------------------------


def _stft_leakage_score(y: np.ndarray, x_hat: np.ndarray) -> float:
    """Post-hoc STFT frame-power leakage gate.  Lower = better.

    Computes STFT of the enhanced signal, sums power over frequency bins to get
    a per-frame scalar, then returns mean(non-speech) / mean(speech).
    Identical formula to gate_step_stft_leakage in utils/speech_gate.py.
    """
    import librosa

    _EPS  = 1e-8
    N_FFT = 512
    HOP   = 128

    T = min(len(y), len(x_hat))
    y     = np.asarray(y[:T],     dtype=np.float32)
    x_hat = np.asarray(x_hat[:T], dtype=np.float32)

    speech_mask_samples = _speech_mask(y)
    win = np.hanning(N_FFT).astype(np.float32)
    S   = np.abs(librosa.stft(x_hat, n_fft=N_FFT, hop_length=HOP,
                               win_length=N_FFT, window=win)) ** 2  # [F, T_frames]

    # Per-frame power: sum over frequency bins
    fp = S.sum(axis=0)  # [T_frames]

    # Convert sample-level mask → frame-level
    n_frames   = fp.shape[0]
    frame_mask = np.zeros(n_frames, dtype=bool)
    for t in range(n_frames):
        start = t * HOP
        end   = min(start + N_FFT, len(speech_mask_samples))
        if start < len(speech_mask_samples):
            frame_mask[t] = np.mean(speech_mask_samples[start:end]) > 0.5

    non_mask = ~frame_mask
    if not frame_mask.any() or not non_mask.any():
        return 0.0

    return float(np.mean(fp[non_mask])) / (float(np.mean(fp[frame_mask])) + _EPS)


def _wiener_residual_score(y: np.ndarray, x_hat: np.ndarray) -> float:
    """Post-hoc Wiener residual gate.  Lower = better.

    Estimates noise PSD from non-speech frames (median across time), then
    computes the ratio of residual (noise-like) energy to speech-excess energy
    over speech frames.  High values = more noise leaking through = worse.
    """
    import librosa

    _EPS   = 1e-8
    N_FFT  = 512
    HOP    = 128

    T = min(len(y), len(x_hat))
    y     = np.asarray(y[:T],     dtype=np.float32)
    x_hat = np.asarray(x_hat[:T], dtype=np.float32)

    speech_mask_samples = _speech_mask(y)
    win = np.hanning(N_FFT).astype(np.float32)
    S   = np.abs(librosa.stft(x_hat, n_fft=N_FFT, hop_length=HOP,
                               win_length=N_FFT, window=win)) ** 2  # [F, T_frames]

    # Convert sample-level mask → frame-level
    n_frames   = S.shape[1]
    frame_mask = np.zeros(n_frames, dtype=bool)
    for t in range(n_frames):
        start = t * HOP
        end   = min(start + N_FFT, len(speech_mask_samples))
        if start < len(speech_mask_samples):
            frame_mask[t] = np.mean(speech_mask_samples[start:end]) > 0.5

    non_mask = ~frame_mask
    if non_mask.sum() < 5 or frame_mask.sum() < 10:
        return 0.0

    # Noise PSD estimate from non-speech frames
    N_f = np.median(S[:, non_mask], axis=1)  # [F]
    N_f = np.maximum(N_f, _EPS)

    S_speech   = S[:, frame_mask]                                   # [F, N_speech]
    speech_est = np.maximum(S_speech - N_f[:, None], 0.0)
    resid_est  = np.minimum(S_speech, N_f[:, None])

    E_speech = float(speech_est.sum())
    E_resid  = float(resid_est.sum())
    return min(E_resid / (E_speech + _EPS), 1e6)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _gate_score(method: str, y: np.ndarray, x_hat: np.ndarray, **_kw) -> float:
    if method == "leakage":
        return _leakage_score(y, x_hat)
    if method == "wiener_residual":
        return _wiener_residual_score(y, x_hat)
    if method == "stft_leakage":
        return _stft_leakage_score(y, x_hat)
    if method == "nisqa":
        return _nisqa_score(x_hat, _kw["sr"])
    raise ValueError(f"Unknown --select_by method: {method!r}")

# Quality metric helpers
# ---------------------------------------------------------------------------

def _sisdr(est: np.ndarray, ref: np.ndarray) -> float:
    ref = np.asarray(ref, dtype=float)
    est = np.asarray(est, dtype=float)
    alpha    = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    s_target = alpha * ref
    e_noise  = est - s_target
    denom = np.dot(e_noise, e_noise)
    if denom < 1e-12:
        return float("nan")
    return float(10.0 * np.log10(np.dot(s_target, s_target) / denom))


def _dnsmos_score(audio_raw: np.ndarray, file_sr: int) -> float:
    audio_16k = to_16k(audio_raw, file_sr)
    raw = compute_dnsmos(audio_16k, 16000)
    return float(raw) if raw is not None else float("nan")


def _nisqa_score(audio_raw: np.ndarray, file_sr: int) -> float:
    """NISQA overall MOS.  Higher = better."""
    raw = compute_nisqa(audio_raw, file_sr)
    return float(raw) if raw is not None else float("nan")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman r (no scipy dependency)."""
    if len(x) < 3:
        return float("nan")
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x, y = x[mask], y[mask]
    n  = len(x)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d2 = np.sum((rx - ry) ** 2)
    return float(1.0 - 6.0 * d2 / (n * (n ** 2 - 1)))


# ---------------------------------------------------------------------------
# Natural sort key
# ---------------------------------------------------------------------------

def _nat_key(p: Path) -> list:
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Best-of-K leakage gate selection: pick the try with the "
                    "lowest leakage gate score per utterance."
    )
    parser.add_argument("--tries_root", required=True, type=Path,
                        help="Directory containing try_0/, try_1/, … subdirs")
    parser.add_argument("--clean_dir",  required=True, type=Path)
    parser.add_argument("--noisy_dir",  required=True, type=Path)
    parser.add_argument("--work_dir",   required=True, type=Path)
    parser.add_argument("--K", type=int, default=None,
                        help="Expected number of tries (sanity check; auto-detected if omitted)")
    parser.add_argument("--dir_pattern", type=str, default="*try*",
                        help="Glob pattern for try subdirectories (default: '*try*')")
    parser.add_argument("--baseline_enhanced_dir", type=Path, default=None,
                        help="Optional baseline enhanced dir whose _results.csv is included "
                             "as an extra comparison column.  Filtered to the same file subset.")
    parser.add_argument("--dnsmos", action="store_true", default=False,
                        help="Compute DNSMOS per try for Spearman correlation and summary table")
    parser.add_argument(
        "--file_list", type=Path, default=None,
        help="Optional text file (one filename per line) to restrict processing to a "
             "specific subset of files.  Files must exist in the try_0 directory. "
             "Use this to evaluate on e.g. worst-30%% SI-SDR without re-running enhancement.",
    )
    parser.add_argument(
        "--select_by", choices=["leakage", "wiener_residual", "stft_leakage", "nisqa"],
        default="leakage",
        help="Gate score used for best-of-K selection (default: leakage)",
    )
    args = parser.parse_args()

    method    = args.select_by
    score_col = {"leakage": "leak_score", "wiener_residual": "wiener_score", "stft_leakage": "stft_leak_score", "nisqa": "nisqa_score"}[method]
    log_tag   = {"leakage": "[LEAK_SELECT]", "wiener_residual": "[WIENER_SELECT]", "stft_leakage": "[STFT_LEAK_SELECT]", "nisqa": "[NISQA_SELECT]"}[method]
    corr_tag  = {"leakage": "[LEAK_CORR]",   "wiener_residual": "[WIENER_CORR]",   "stft_leakage": "[STFT_LEAK_CORR]",   "nisqa": "[NISQA_CORR]"}[method]
    short     = {"leakage": "leak",           "wiener_residual": "wiener",          "stft_leakage": "stft_leak",           "nisqa": "nisqa"}[method]
    higher_is_better = (method == "nisqa")

    # ------------------------------------------------------------------
    # 0. Setup
    # ------------------------------------------------------------------
    args.work_dir.mkdir(parents=True, exist_ok=True)
    best_dir    = args.work_dir / f"best_of_k_{short}"
    results_dir = args.work_dir / f"{short}_best_of_K"
    best_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    compute_dnsmos_flag = args.dnsmos
    if compute_dnsmos_flag and not dnsmos_available():
        print("WARNING: --dnsmos requested but speechmos not installed; "
              "DNSMOS will be NaN.  Install with: pip install speechmos")
        compute_dnsmos_flag = False

    if method == "nisqa" and not nisqa_available():
        print("ERROR: --select_by nisqa requires the nisqa package. "
              "Install with: pip install nisqa  and set NISQA_CKPT_PATH.")
        sys.exit(1)

    print(f"Selection method : {method}")

    # ------------------------------------------------------------------
    # 1. Locate try folders
    # ------------------------------------------------------------------
    _discovered = sorted(
        [d for d in args.tries_root.glob(args.dir_pattern) if d.is_dir()],
        key=_nat_key,
    )
    if not _discovered:
        print(f"ERROR: no directories matching '{args.dir_pattern}' found under {args.tries_root}")
        print(f"  Contents: {[d.name for d in sorted(args.tries_root.iterdir())]}")
        sys.exit(1)

    if args.K is not None and len(_discovered) != args.K:
        print(f"WARNING: --K={args.K} but found {len(_discovered)} matching dirs; "
              f"proceeding with all {len(_discovered)}.")

    try_dirs: list[tuple[int, Path]] = list(enumerate(_discovered))
    try_ks = [k for k, _ in try_dirs]
    try_dir_map: dict[int, Path] = {k: d for k, d in try_dirs}
    K = len(try_dirs)
    print(f"Found {K} try folder(s):")
    for k, d in try_dirs:
        print(f"  [{k}] {d.name}")

    # ------------------------------------------------------------------
    # 2. Collect file list (from try_0; all tries must have the same files)
    # ------------------------------------------------------------------
    try0_dir = try_dir_map[try_ks[0]]
    wav_files = sorted(
        list(try0_dir.glob("*.wav")) + list(try0_dir.glob("*.flac")),
        key=lambda p: p.name,
    )
    if not wav_files:
        print(f"ERROR: no wav/flac files found in {try0_dir}")
        sys.exit(1)
    filenames = [p.name for p in wav_files]

    # Optional: restrict to a specific file subset (e.g. worst-30% SI-SDR)
    if args.file_list is not None:
        if not args.file_list.exists():
            print(f"ERROR: --file_list {args.file_list} not found")
            sys.exit(1)
        wanted = {ln.strip() for ln in args.file_list.read_text().splitlines() if ln.strip()}
        missing_from_tries = wanted - set(filenames)
        if missing_from_tries:
            print(f"  WARNING: {len(missing_from_tries)} files in --file_list not found in "
                  f"try_0 dir: {sorted(missing_from_tries)[:5]} …")
        filenames = [f for f in filenames if f in wanted]
        print(f"  --file_list filter: {len(filenames)} files kept "
              f"(from {len(wanted)} requested, {len(missing_from_tries)} not found in tries)")

    print(f"\n{len(filenames)} files in try_0 ({try0_dir.name})")

    # ------------------------------------------------------------------
    # 3. Per-file per-try: gate score + optional DNSMOS + SI-SDR
    # ------------------------------------------------------------------
    print(f"\nStep 3 — Computing {method} scores (and optional quality metrics) for all tries …")

    per_row_records = []  # one row per (file, try)

    # Pre-load noisy files once
    print("  Loading noisy files …")
    noisy_cache: dict[str, tuple[np.ndarray, int]] = {}
    for fname in tqdm(filenames, desc="  noisy", leave=False):
        noisy_path = args.noisy_dir / fname
        if not noisy_path.exists():
            candidates = list(args.noisy_dir.rglob(fname))
            noisy_path = candidates[0] if candidates else noisy_path
        if noisy_path.exists():
            y, sr_y = sf.read(str(noisy_path), dtype="float32")
            if y.ndim == 2:
                y = y.mean(axis=1)
            noisy_cache[fname] = (y, int(sr_y))
        else:
            print(f"  WARNING: noisy file not found: {fname} — gate score will be NaN")

    # Pre-load clean files for SI-SDR
    print("  Loading clean files …")
    clean_cache: dict[str, tuple[np.ndarray, int]] = {}
    for fname in tqdm(filenames, desc="  clean", leave=False):
        clean_fname = fname.split("_")[0] + ".wav" if "dB" in fname else fname
        clean_path  = args.clean_dir / clean_fname
        if clean_path.exists():
            x, sr_x = sf.read(str(clean_path), dtype="float32")
            if x.ndim == 2:
                x = x.mean(axis=1)
            clean_cache[fname] = (x, int(sr_x))
        else:
            print(f"  WARNING: clean file not found: {clean_fname}")

    # Iterate over all tries
    for k, try_dir in try_dirs:
        print(f"  Try {k}: {try_dir.name}")
        for fname in tqdm(filenames, desc=f"    try_{k}", leave=False):
            enh_path = try_dir / fname
            if not enh_path.exists():
                print(f"    WARNING: {enh_path} not found — skipping.")
                per_row_records.append({
                    "file_id":   fname,
                    "try_index": k,
                    score_col:   float("nan"),
                    "dnsmos":    float("nan"),
                    "sisdr":     float("nan"),
                })
                continue

            x_hat, sr_h = sf.read(str(enh_path), dtype="float32")
            if x_hat.ndim == 2:
                x_hat = x_hat.mean(axis=1)

            # --- gate score ---
            if fname in noisy_cache:
                y, _ = noisy_cache[fname]
                T    = min(len(y), len(x_hat))
                gate = _gate_score(method, y[:T], x_hat[:T], sr=sr_h)
            else:
                gate = float("nan")

            # --- SI-SDR ---
            if fname in clean_cache:
                x, _ = clean_cache[fname]
                T    = min(len(x), len(x_hat))
                sdr  = _sisdr(x_hat[:T], x[:T])
            else:
                sdr = float("nan")

            # --- DNSMOS (optional) ---
            dnsmos_val = _dnsmos_score(x_hat, sr_h) if compute_dnsmos_flag else float("nan")

            per_row_records.append({
                "file_id":   fname,
                "try_index": k,
                score_col:   gate,
                "dnsmos":    dnsmos_val,
                "sisdr":     sdr,
            })

    per_try_df = pd.DataFrame(per_row_records)

    # ------------------------------------------------------------------
    # 4. Select best try per file (argmin gate score)
    # ------------------------------------------------------------------
    print(f"\nStep 4 — Selecting best try per file (argmin {score_col}) …")

    selection_records = []
    for fname in filenames:
        file_df = per_try_df[per_try_df["file_id"] == fname].copy()
        scores  = file_df[score_col].to_numpy()

        assert len(scores) == K, f"Expected {K} scores for {fname}, got {len(scores)}"

        finite_mask = np.isfinite(scores)
        if not finite_mask.any():
            # All tries failed — cannot select; fall back to try_0
            print(f"  WARNING: all {score_col} scores non-finite for {fname} "
                  f"— falling back to try_0")
            k_star     = try_ks[0]
            score_best = float("nan")
        else:
            # Map failed tries to worst possible value so they are never selected
            if higher_is_better:
                safe_scores = np.where(finite_mask, scores, -np.inf)
                best_local  = int(np.argmax(safe_scores))
            else:
                safe_scores = np.where(finite_mask, scores, np.inf)
                best_local  = int(np.argmin(safe_scores))
            if not finite_mask.all():
                print(f"  WARNING: {(~finite_mask).sum()} non-finite {score_col} score(s) for "
                      f"{fname} — selecting from {finite_mask.sum()} valid tries")
            k_star     = int(try_ks[best_local])
            score_best = float(scores[best_local])

        score_try0 = float(scores[0]) if np.isfinite(scores[0]) else float("nan")

        print(
            f"\n{log_tag}\n"
            f"  file={fname}\n"
            f"  scores={[round(float(v), 6) for v in scores]}\n"
            f"  k_star={k_star}\n"
            f"  score_best={score_best:.6f}\n"
            f"  score_try0={score_try0:.6f}\n"
            f"  score_min={float(np.nanmin(scores)):.6f}  "
            f"score_max={float(np.nanmax(scores)):.6f}"
        )

        selection_records.append({
            "file_id":            fname,
            "selected_try_index": k_star,
            f"{short}_try0":      score_try0,
            f"{short}_best":      score_best,
        })

    sel_df = pd.DataFrame(selection_records)

    # ------------------------------------------------------------------
    # 5. Copy selected wav files to best_dir
    # ------------------------------------------------------------------
    print(f"\nStep 5 — Copying best-of-{K} ({method}) wavs to {best_dir} …")
    n_copied = n_missing = 0
    for _, row in sel_df.iterrows():
        fname  = row["file_id"]
        best_k = int(row["selected_try_index"])
        src    = try_dir_map[best_k] / fname
        if not src.exists():
            print(f"  WARNING: {src} not found — skipping.")
            n_missing += 1
            continue
        shutil.copy2(src, best_dir / fname)
        n_copied += 1
    print(f"  Copied {n_copied} files  ({n_missing} missing).")

    # ------------------------------------------------------------------
    # 6. Run calc_metrics on try_0 and best_of_k_{method}
    # ------------------------------------------------------------------
    def _run_metrics(enhanced_dir: Path):
        cmd = [
            sys.executable, "calc_metrics.py",
            "--clean_dir",    str(args.clean_dir),
            "--noisy_dir",    str(args.noisy_dir),
            "--enhanced_dir", str(enhanced_dir),
            "--dnsmos",
        ]
        print("$", " ".join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"WARNING: calc_metrics.py exited with code {result.returncode}.")

    try0_csv = try0_dir / "_results.csv"
    best_csv = best_dir / "_results.csv"

    print(f"\nStep 6a — calc_metrics on try_0 ({try0_dir.name}) …")
    if not try0_csv.exists():
        _run_metrics(try0_dir)
    else:
        print(f"  Found existing {try0_csv.name} — skipping re-computation.")

    print(f"\nStep 6b — calc_metrics on best_of_k_{short} …")
    _run_metrics(best_dir)

    # ------------------------------------------------------------------
    # 7. Build per-file×try CSV  (same-files join on filename)
    # ------------------------------------------------------------------
    print("\nStep 7 — Building per-file output CSV …")

    if best_csv.exists():
        best_metrics_df = (
            pd.read_csv(best_csv)[["filename", "sisdr_enh", "pesq", "estoi", "dnsmos_ovrl"]]
            .rename(columns={
                "filename":    "file_id",
                "sisdr_enh":   "sisdr_best",
                "pesq":        "pesq_best",
                "estoi":       "estoi_best",
                "dnsmos_ovrl": "dnsmos_best",
            })
        )
    else:
        best_metrics_df = None

    out_df = per_try_df.merge(sel_df[["file_id", "selected_try_index"]], on="file_id", how="left")
    if best_metrics_df is not None:
        out_df = out_df.merge(best_metrics_df, on="file_id", how="inner")

    out_csv = results_dir / f"{short}_selection_per_file.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"  Per-file CSV → {out_csv}  ({len(out_df)} rows)")

    # ------------------------------------------------------------------
    # 8. Spearman correlations: gate score ↔ quality
    #    (negative = gate is predictive of quality)
    # ------------------------------------------------------------------
    print(f"\n{corr_tag} Computing Spearman correlations ({method} vs quality) …")

    corr_df    = per_try_df.dropna(subset=[score_col])
    score_arr  = corr_df[score_col].to_numpy()
    dnsmos_arr = corr_df["dnsmos"].to_numpy()
    sisdr_arr  = corr_df["sisdr"].to_numpy()

    r_dnsmos = _spearman(score_arr, dnsmos_arr)
    r_sisdr  = _spearman(score_arr, sisdr_arr)

    def _corr_label(r: float, higher_is_better: bool = False) -> str:
        if not np.isfinite(r):
            return "n/a"
        good = (r > 0) if higher_is_better else (r < 0)
        return "predicts quality ✓" if good else "anti-correlated with quality ✗"

    print(f"\n{corr_tag}")
    print(f"  n_pairs                    = {len(corr_df)}   (K={K} × {len(filenames)} files)")
    print(f"  spearman({short}, DNSMOS)  = {r_dnsmos:+.4f}  ({_corr_label(r_dnsmos, higher_is_better)})")
    print(f"  spearman({short}, SI-SDR)  = {r_sisdr:+.4f}  ({_corr_label(r_sisdr, higher_is_better)})")

    # ------------------------------------------------------------------
    # 9. Summary comparison table  (same-files bug fix: join on filename)
    # ------------------------------------------------------------------
    metrics_cols = ["sisdr_enh", "pesq", "estoi", "dnsmos_ovrl"]
    columns: list[tuple[str, pd.DataFrame | None]] = []

    if try0_csv.exists():
        columns.append(("try_0", pd.read_csv(try0_csv)))
    else:
        print("  NOTE: try_0 _results.csv not found — summary column will be absent.")

    if best_csv.exists():
        columns.append((f"{short}_best_of_{K}", pd.read_csv(best_csv)))

    if args.baseline_enhanced_dir is not None:
        base_csv_path = args.baseline_enhanced_dir / "_results.csv"
        if base_csv_path.exists():
            base_df      = pd.read_csv(base_csv_path)
            subset_files = set(sel_df["file_id"])
            columns.insert(0, ("baseline (subset)",
                               base_df[base_df["filename"].isin(subset_files)].copy()))
        else:
            print(f"  WARNING: baseline _results.csv not found at {base_csv_path}")

    # Intersect on filename across all result DataFrames
    if len(columns) >= 2:
        common_files = None
        for _, cdf in columns:
            if cdf is not None and "filename" in cdf.columns:
                fset = set(cdf["filename"].tolist())
                common_files = fset if common_files is None else common_files & fset
        common_files = common_files or set()
        n_common = len(common_files)
        if n_common == 0:
            print("\nWARNING: no common files found between result CSVs — "
                  "check that --noisy_dir, --clean_dir, and --tries_root share the same split.")
        else:
            print(f"\n  Summary on {n_common} common files "
                  f"(filtered from {len(sel_df)} tries; guards against cross-dataset mixing).")
            columns = [
                (lbl, cdf[cdf["filename"].isin(common_files)].copy() if cdf is not None else None)
                for lbl, cdf in columns
            ]

    title = f"Best-of-K {method} Selection — Metrics Comparison (same file subset)"
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"  K = {K}   subset = {len(filenames)} files   "
          f"method = {method}")
    print()

    col_w    = 18
    try0_idx = 1 if args.baseline_enhanced_dir is not None else 0
    best_idx = len(columns) - 1

    header = f"  {'metric':<12}" + "".join(f"  {lbl:>{col_w}}" for lbl, _ in columns)
    if len(columns) >= 2:
        header += f"  {'delta(0→best)':>{col_w}}"
    print(header)
    print("  " + "-" * (12 + (col_w + 2) * (len(columns) + (1 if len(columns) >= 2 else 0))))

    table_lines = [title,
                   f"K={K}  subset={len(filenames)} files  method={method}",
                   header.strip()]

    for m in metrics_cols:
        vals = []
        for _, cdf in columns:
            if cdf is not None and m in cdf.columns:
                vals.append(float(cdf[m].mean(skipna=True)))
            else:
                vals.append(float("nan"))
        row = f"  {m:<12}" + "".join(f"  {v:>{col_w}.4f}" for v in vals)
        if len(vals) >= 2:
            delta = vals[best_idx] - vals[try0_idx]
            row  += f"  {f'{delta:+.4f}':>{col_w}}"
        print(row)
        table_lines.append(row.strip())

    print()
    print(f"  Spearman({short} ↔ DNSMOS)  = {r_dnsmos:+.4f}")
    print(f"  Spearman({short} ↔ SI-SDR)  = {r_sisdr:+.4f}")
    print()

    # ------------------------------------------------------------------
    # 10. Save text summary
    # ------------------------------------------------------------------
    summary_path = results_dir / f"{short}_selection_metrics.txt"
    with open(summary_path, "w") as f:
        for line in table_lines:
            f.write(line + "\n")
        f.write(f"\nSpearman({short}, DNSMOS) = {r_dnsmos:+.4f}\n")
        f.write(f"Spearman({short}, SI-SDR) = {r_sisdr:+.4f}\n")
        f.write(f"\nn_pairs_corr = {len(corr_df)}  (K={K} x {len(filenames)} files)\n")

    print(f"Summary saved → {summary_path}")
    print(f"Per-file CSV  → {out_csv}")


if __name__ == "__main__":
    main()
