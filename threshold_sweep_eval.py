#!/usr/bin/env python3
"""Adaptive best-of-K threshold sweep.

For each utterance, a baseline difficulty score d0 is computed on try_0.
  - Lower-is-better scores (leakage, stft_leakage, wiener_residual):
      difficulty = score        (high → hard → escalate)
  - Higher-is-better scores (nisqa):
      difficulty = -score       (low MOS → hard → escalate)

Rule: if difficulty(d0) > tau → use best-of-K_high tries (by same score)
      else                    → use try_0 as-is (K=1)

Sweeps tau from "always escalate" to "never escalate" and reports per tau:
  escalation_rate    fraction of files escalated  (0–1)
  avg_K              effective mean compute per utterance  (1 … K_high)
  mean_sisdr         overall mean SI-SDR of selected outputs
  mean_pesq          overall mean PESQ-WB of selected outputs
  mean_estoi         overall mean ESTOI of selected outputs
  mean_dnsmos        overall mean DNSMOS overall-MOS of selected outputs
  worst10_mean_sisdr mean SI-SDR of worst-10% files (by selected SI-SDR)
  harm_rate_vs_try0  P(selected SI-SDR ≤ try0 SI-SDR | escalated)

SI-SDR is used ONLY for reporting, never inside the selection policy.
Selection policy uses only the requested difficulty score (e.g. wiener_residual).

Runs all selected difficulty scores in one audio pass, then sweeps tau per
score independently.  Produces one CSV per score plus a combined CSV and
compute-quality curve plots.

Usage:
    python threshold_sweep_eval.py \\
        --tries_root voicebank/oracle_tries \\
        --clean_dir  voicebank/test_clean \\
        --noisy_dir  voicebank/test_noisy \\
        --work_dir   voicebank/exp/threshold_sweep \\
        --K_high 10 \\
        --scores leakage stft_leakage wiener_residual nisqa \\
        --n_tau 60 \\
        --compute_dnsmos
"""

import re
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.speech_gate import compute_speech_gate_score
from utils.dnsmos_helper import to_16k
from utils.nisqa_helper import compute_nisqa, is_available as nisqa_available

# ---------------------------------------------------------------------------
# Optional quality-metric imports (graceful fallback — never crash if absent)
# ---------------------------------------------------------------------------

try:
    from pesq import pesq as _pesq_fn
    _PESQ_AVAILABLE = True
except ImportError:
    _PESQ_AVAILABLE = False
    _pesq_fn = None  # type: ignore[assignment]

try:
    from pystoi import stoi as _stoi_fn
    _ESTOI_AVAILABLE = True
except ImportError:
    _ESTOI_AVAILABLE = False
    _stoi_fn = None  # type: ignore[assignment]

try:
    from utils.dnsmos_helper import compute_dnsmos as _compute_dnsmos
    from utils.dnsmos_helper import is_available as dnsmos_available
    _DNSMOS_IMPORTABLE = True
except ImportError:
    _DNSMOS_IMPORTABLE = False
    _compute_dnsmos = None  # type: ignore[assignment]
    def dnsmos_available() -> bool: return False  # noqa: E704


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_SCORE_METHODS = ["leakage", "stft_leakage", "wiener_residual", "nisqa"]
HIGHER_IS_BETTER  = {
    "leakage":          False,
    "stft_leakage":     False,
    "wiener_residual":  False,
    "nisqa":            True,
}

_VAD_WINDOW = 400  # samples


# ---------------------------------------------------------------------------
# VAD mask (shared with leakage_best_of_k_eval.py)
# ---------------------------------------------------------------------------

def _speech_mask(y: np.ndarray, percentile: float = 80.0) -> np.ndarray:
    energy = np.convolve(y ** 2, np.ones(_VAD_WINDOW) / _VAD_WINDOW, mode="same")
    return energy > np.percentile(energy, percentile)


# ---------------------------------------------------------------------------
# Difficulty / gate scores
# ---------------------------------------------------------------------------

def _leakage_score(y: np.ndarray, x_hat: np.ndarray) -> float:
    mask = _speech_mask(y)
    return compute_speech_gate_score(y, x_hat, mask)


def _stft_leakage_score(y: np.ndarray, x_hat: np.ndarray) -> float:
    import librosa

    _EPS  = 1e-8
    N_FFT = 512
    HOP   = 128

    T     = min(len(y), len(x_hat))
    y     = np.asarray(y[:T],     dtype=np.float32)
    x_hat = np.asarray(x_hat[:T], dtype=np.float32)

    speech_mask_samples = _speech_mask(y)
    win = np.hanning(N_FFT).astype(np.float32)
    S   = np.abs(librosa.stft(x_hat, n_fft=N_FFT, hop_length=HOP,
                               win_length=N_FFT, window=win)) ** 2

    fp       = S.sum(axis=0)
    n_frames = fp.shape[0]
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
    import librosa

    _EPS  = 1e-8
    N_FFT = 512
    HOP   = 128

    T     = min(len(y), len(x_hat))
    y     = np.asarray(y[:T],     dtype=np.float32)
    x_hat = np.asarray(x_hat[:T], dtype=np.float32)

    speech_mask_samples = _speech_mask(y)
    win = np.hanning(N_FFT).astype(np.float32)
    S   = np.abs(librosa.stft(x_hat, n_fft=N_FFT, hop_length=HOP,
                               win_length=N_FFT, window=win)) ** 2

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

    N_f        = np.median(S[:, non_mask], axis=1)
    N_f        = np.maximum(N_f, _EPS)
    S_speech   = S[:, frame_mask]
    speech_est = np.maximum(S_speech - N_f[:, None], 0.0)
    resid_est  = np.minimum(S_speech, N_f[:, None])
    return min(float(resid_est.sum()) / (float(speech_est.sum()) + _EPS), 1e6)


def _nisqa_score(audio_raw: np.ndarray, sr: int) -> float:
    raw = compute_nisqa(audio_raw, sr)
    return float(raw) if raw is not None else float("nan")


def _compute_all_scores(
    y:      np.ndarray,
    x_hat:  np.ndarray,
    sr:     int,
    methods: list,
) -> dict:
    """Compute all requested gate scores for one (noisy, enhanced) pair."""
    out: dict = {}
    for m in methods:
        try:
            if m == "leakage":
                out[m] = _leakage_score(y, x_hat)
            elif m == "stft_leakage":
                out[m] = _stft_leakage_score(y, x_hat)
            elif m == "wiener_residual":
                out[m] = _wiener_residual_score(y, x_hat)
            elif m == "nisqa":
                out[m] = _nisqa_score(x_hat, sr)
        except Exception as exc:
            print(f"    WARNING: {m} score failed: {exc}")
            out[m] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Quality metrics (reporting only — never used inside the selection policy)
# ---------------------------------------------------------------------------

def _sisdr(est: np.ndarray, ref: np.ndarray) -> float:
    ref   = np.asarray(ref, dtype=float)
    est   = np.asarray(est, dtype=float)
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    s_tar = alpha * ref
    e_noi = est - s_tar
    denom = np.dot(e_noi, e_noi)
    if denom < 1e-12:
        return float("nan")
    return float(10.0 * np.log10(np.dot(s_tar, s_tar) / denom))


def _compute_pesq(x_hat: np.ndarray, x: np.ndarray, sr: int) -> float:
    """PESQ wide-band (16 kHz). Returns nan on failure."""
    if not _PESQ_AVAILABLE:
        return float("nan")
    try:
        x_hat_16 = to_16k(x_hat, sr)
        x_16     = to_16k(x,     sr)
        T = min(len(x_hat_16), len(x_16))
        return float(_pesq_fn(16000, x_16[:T], x_hat_16[:T], "wb"))
    except Exception:
        return float("nan")


def _compute_estoi(x_hat: np.ndarray, x: np.ndarray, sr: int) -> float:
    """Extended STOI. Works at native sample rate."""
    if not _ESTOI_AVAILABLE:
        return float("nan")
    try:
        T = min(len(x_hat), len(x))
        return float(_stoi_fn(x[:T], x_hat[:T], sr, extended=True))
    except Exception:
        return float("nan")


def _compute_dnsmos_score(x_hat: np.ndarray, sr: int) -> float:
    """DNSMOS overall MOS (reference-free). Returns nan on failure."""
    if not _DNSMOS_IMPORTABLE or _compute_dnsmos is None:
        return float("nan")
    try:
        x16  = to_16k(x_hat, sr)
        raw  = _compute_dnsmos(x16, 16000)
        return float(raw) if raw is not None else float("nan")
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Natural sort
# ---------------------------------------------------------------------------

def _nat_key(p: Path) -> list:
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)]


# ---------------------------------------------------------------------------
# Tau sweep for one difficulty method
# ---------------------------------------------------------------------------

def _tau_sweep(
    scores_mat:  np.ndarray,         # (N, K_use) difficulty scores
    sisdr_mat:   np.ndarray,         # (N, K_use) SI-SDR — reporting only
    method:      str,
    K_high:      int,
    n_tau:       int,
    pesq_mat:    "np.ndarray | None" = None,   # (N, K_use) or None
    estoi_mat:   "np.ndarray | None" = None,   # (N, K_use) or None
    dnsmos_mat:  "np.ndarray | None" = None,   # (N, K_use) or None
) -> pd.DataFrame:
    """Vectorised tau sweep.

    Selection policy: d0 > tau → use try with argmin/argmax over the score
                      d0 ≤ tau → use try_0 as-is
    SI-SDR (and PESQ/ESTOI/DNSMOS) are used ONLY for reporting.

    Returns a DataFrame with columns:
        method, tau, escalation_rate, avg_K,
        mean_sisdr, worst10_mean_sisdr, harm_rate_vs_try0,
        mean_pesq, mean_estoi, mean_dnsmos,
        n_files
    """
    higher = HIGHER_IS_BETTER[method]
    K_use  = min(K_high, scores_mat.shape[1])

    # Slice to the K tries we actually use
    sc   = scores_mat[:, :K_use]   # (N, K_use)
    si   = sisdr_mat[:, :K_use]    # (N, K_use)

    # ---- auxiliary metric slices (None if not computed) ----
    pe  = pesq_mat[:, :K_use]  if pesq_mat  is not None else None
    es  = estoi_mat[:, :K_use] if estoi_mat  is not None else None
    dn  = dnsmos_mat[:, :K_use] if dnsmos_mat is not None else None

    # Difficulty: higher always means "harder" → escalate
    diff = sc * (-1.0 if higher else 1.0)   # (N, K_use)
    d0   = diff[:, 0]                        # (N,) difficulty from try_0

    # SI-SDR of try_0 (reporting baseline for harm_rate_vs_try0)
    sisdr0 = si[:, 0]  # (N,)

    # Pre-select "best of K_use" index for each file (policy; ignores tau).
    # Row-by-row loop avoids np.nanargmin/nanargmax evaluating on all-NaN rows
    # even inside np.where, which raises ValueError before the guard fires.
    best_try_idx = np.zeros(sc.shape[0], dtype=int)
    for _i in range(sc.shape[0]):
        row = sc[_i]
        if np.all(np.isnan(row)):
            best_try_idx[_i] = 0
        else:
            best_try_idx[_i] = np.nanargmax(row) if higher else np.nanargmin(row)
    rows_idx = np.arange(si.shape[0])
    best_sisdr  = si[rows_idx, best_try_idx]
    best_pesq   = pe[rows_idx, best_try_idx]  if pe  is not None else None
    best_estoi  = es[rows_idx, best_try_idx]  if es  is not None else None
    best_dnsmos = dn[rows_idx, best_try_idx]  if dn  is not None else None

    # try_0 quality metrics (reporting baseline)
    pesq0   = pe[:, 0]   if pe  is not None else None
    estoi0  = es[:, 0]   if es  is not None else None
    dnsmos0 = dn[:, 0]   if dn  is not None else None

    # Valid files: must have finite d0, sisdr0, best_sisdr
    valid = np.isfinite(d0) & np.isfinite(sisdr0) & np.isfinite(best_sisdr)
    if not valid.any():
        print(f"  [{method}] WARNING: no valid files for tau sweep.")
        return pd.DataFrame()

    d0_v       = d0[valid]
    sisdr0_v   = sisdr0[valid]
    best_si_v  = best_sisdr[valid]
    N_valid    = int(valid.sum())

    pesq0_v    = pesq0[valid]   if pesq0   is not None else None
    estoi0_v   = estoi0[valid]  if estoi0  is not None else None
    dnsmos0_v  = dnsmos0[valid] if dnsmos0 is not None else None
    b_pesq_v   = best_pesq[valid]   if best_pesq   is not None else None
    b_estoi_v  = best_estoi[valid]  if best_estoi  is not None else None
    b_dnsmos_v = best_dnsmos[valid] if best_dnsmos is not None else None

    # Tau grid covering full [100%, 0%] escalation range
    finite_d0 = d0_v[np.isfinite(d0_v)]
    taus = np.unique(np.concatenate([
        [finite_d0.min() - abs(finite_d0.min()) * 0.01 - 1e-10],
        np.percentile(finite_d0, np.linspace(0.0, 100.0, n_tau)),
        [finite_d0.max() + abs(finite_d0.max()) * 0.01 + 1e-10],
    ]))

    def _mean_selected(arr0, arr_best, esc_mask):
        """For each file: if escalated use arr_best, else arr0.  Return nanmean."""
        if arr0 is None or arr_best is None:
            return float("nan")
        sel = np.where(esc_mask, arr_best, arr0)
        finite = sel[np.isfinite(sel)]
        return float(finite.mean()) if len(finite) > 0 else float("nan")

    rows = []
    for tau in taus:
        escalate = d0_v > tau                                      # (N_valid,)
        sel_sisdr = np.where(escalate, best_si_v, sisdr0_v)        # (N_valid,)

        n_esc         = int(escalate.sum())
        escalation_rate = float(escalate.mean())
        avg_K         = 1.0 + (K_use - 1) * escalation_rate

        # Worst-10% SI-SDR (of the selected outputs)
        w10_thresh        = np.percentile(sel_sisdr, 10.0)
        worst10_mask      = sel_sisdr <= w10_thresh
        worst10_mean_sisdr = float(sel_sisdr[worst10_mask].mean()) if worst10_mask.any() else float("nan")

        # Harm rate: P(selected SI-SDR < try0 SI-SDR | escalated)
        # Strict < excludes ties caused by floating-point equality.
        if n_esc > 0:
            delta              = sel_sisdr[escalate] - sisdr0_v[escalate]
            harm_rate_vs_try0  = float((delta < 0.0).mean())
        else:
            harm_rate_vs_try0  = float("nan")

        # Auxiliary quality metrics
        mean_pesq   = _mean_selected(pesq0_v,   b_pesq_v,   escalate)
        mean_estoi  = _mean_selected(estoi0_v,  b_estoi_v,  escalate)
        mean_dnsmos = _mean_selected(dnsmos0_v, b_dnsmos_v, escalate)

        rows.append({
            "method":             method,
            "tau":                float(tau),
            "escalation_rate":    escalation_rate,
            "avg_K":              float(avg_K),
            "mean_sisdr":         float(np.nanmean(sel_sisdr)),
            "worst10_mean_sisdr": worst10_mean_sisdr,
            "harm_rate_vs_try0":  harm_rate_vs_try0,
            "mean_pesq":          mean_pesq,
            "mean_estoi":         mean_estoi,
            "mean_dnsmos":        mean_dnsmos,
            "n_files":            N_valid,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_curves(combined_df: pd.DataFrame, plots_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots.")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    methods = combined_df["method"].unique()
    cmap    = plt.get_cmap("tab10")
    colors  = {m: cmap(i) for i, m in enumerate(methods)}

    def _save(fig, name):
        out = plots_dir / name
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Plot → {out}")

    # 1. Compute–quality curve: mean SI-SDR vs avg_K
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in methods:
        sub = combined_df[combined_df["method"] == m].sort_values("avg_K")
        ax.plot(sub["avg_K"], sub["mean_sisdr"], label=m, color=colors[m], linewidth=1.8)
        ax.scatter([sub["avg_K"].iloc[0], sub["avg_K"].iloc[-1]],
                   [sub["mean_sisdr"].iloc[0], sub["mean_sisdr"].iloc[-1]],
                   color=colors[m], s=25, zorder=5)
    ax.set_xlabel("Average K (effective compute per utterance)")
    ax.set_ylabel("Mean SI-SDR (dB)")
    ax.set_title("Compute–Quality Curve (threshold sweep)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "compute_quality_curve.png")

    # 2. Worst-10% SI-SDR vs avg_K
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in methods:
        sub = combined_df[combined_df["method"] == m].sort_values("avg_K")
        ax.plot(sub["avg_K"], sub["worst10_mean_sisdr"], label=m, color=colors[m], linewidth=1.8)
    ax.set_xlabel("Average K")
    ax.set_ylabel("Worst-10% Mean SI-SDR (dB)")
    ax.set_title("Worst-10% Compute–Quality Curve")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "worst10_curve.png")

    # 3. Harm rate vs escalation rate
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in methods:
        sub = combined_df[combined_df["method"] == m].dropna(
            subset=["harm_rate_vs_try0"]
        ).sort_values("escalation_rate")
        ax.plot(sub["escalation_rate"] * 100, sub["harm_rate_vs_try0"],
                label=m, color=colors[m], linewidth=1.8)
    ax.set_xlabel("Escalation Rate (%)")
    ax.set_ylabel("Harm Rate vs try_0 (P(ΔSI-SDR ≤ 0 | escalated))")
    ax.set_title("Escalation Harm Rate vs Coverage")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "harm_rate_curve.png")

    # 4. PESQ vs avg_K (if available)
    if combined_df["mean_pesq"].notna().any():
        fig, ax = plt.subplots(figsize=(7, 5))
        for m in methods:
            sub = combined_df[combined_df["method"] == m].dropna(
                subset=["mean_pesq"]
            ).sort_values("avg_K")
            if sub.empty:
                continue
            ax.plot(sub["avg_K"], sub["mean_pesq"], label=m, color=colors[m], linewidth=1.8)
        ax.set_xlabel("Average K")
        ax.set_ylabel("Mean PESQ-WB")
        ax.set_title("Compute–PESQ Curve")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, "compute_pesq_curve.png")

    # 5. ESTOI vs avg_K (if available)
    if combined_df["mean_estoi"].notna().any():
        fig, ax = plt.subplots(figsize=(7, 5))
        for m in methods:
            sub = combined_df[combined_df["method"] == m].dropna(
                subset=["mean_estoi"]
            ).sort_values("avg_K")
            if sub.empty:
                continue
            ax.plot(sub["avg_K"], sub["mean_estoi"], label=m, color=colors[m], linewidth=1.8)
        ax.set_xlabel("Average K")
        ax.set_ylabel("Mean ESTOI")
        ax.set_title("Compute–ESTOI Curve")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, "compute_estoi_curve.png")

    # 6. DNSMOS vs avg_K (if available)
    if combined_df["mean_dnsmos"].notna().any():
        fig, ax = plt.subplots(figsize=(7, 5))
        for m in methods:
            sub = combined_df[combined_df["method"] == m].dropna(
                subset=["mean_dnsmos"]
            ).sort_values("avg_K")
            if sub.empty:
                continue
            ax.plot(sub["avg_K"], sub["mean_dnsmos"], label=m, color=colors[m], linewidth=1.8)
        ax.set_xlabel("Average K")
        ax.set_ylabel("Mean DNSMOS Overall MOS")
        ax.set_title("Compute–DNSMOS Curve")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, "compute_dnsmos_curve.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Adaptive best-of-K threshold sweep: compute–quality trade-off."
    )
    parser.add_argument("--tries_root",  required=True,  type=Path,
                        help="Directory containing try_0/, try_1/, … subdirs")
    parser.add_argument("--clean_dir",   required=True,  type=Path)
    parser.add_argument("--noisy_dir",   required=True,  type=Path)
    parser.add_argument("--work_dir",    required=True,  type=Path)
    parser.add_argument("--K_high",      type=int, default=10,
                        help="Number of tries used when a file is escalated (default: 10)")
    parser.add_argument("--scores",      nargs="+",
                        choices=ALL_SCORE_METHODS, default=ALL_SCORE_METHODS,
                        help="Difficulty scores to sweep (default: all four)")
    parser.add_argument("--n_tau",       type=int, default=60,
                        help="Number of tau grid points (default: 60)")
    parser.add_argument("--dir_pattern", type=str, default="*try*",
                        help="Glob pattern for try subdirectories (default: '*try*')")
    parser.add_argument("--file_list",   type=Path, default=None,
                        help="Optional file (one filename per line) to restrict processing")
    # Quality metric flags
    parser.add_argument("--no_pesq",     action="store_true", default=False,
                        help="Skip PESQ computation (default: compute if pesq package available)")
    parser.add_argument("--no_estoi",    action="store_true", default=False,
                        help="Skip ESTOI computation (default: compute if pystoi package available)")
    parser.add_argument("--compute_dnsmos", action="store_true", default=False,
                        help="Compute DNSMOS (slow; requires speechmos package)")
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.work_dir / "plots"

    # Resolve which quality metrics to compute
    do_pesq  = not args.no_pesq  and _PESQ_AVAILABLE
    do_estoi = not args.no_estoi and _ESTOI_AVAILABLE
    do_dnsmos = args.compute_dnsmos and _DNSMOS_IMPORTABLE and dnsmos_available()

    if not args.no_pesq  and not _PESQ_AVAILABLE:
        print("WARNING: pesq package not found — PESQ will be NaN.")
    if not args.no_estoi and not _ESTOI_AVAILABLE:
        print("WARNING: pystoi package not found — ESTOI will be NaN.")
    if args.compute_dnsmos and not (dnsmos_available() and _DNSMOS_IMPORTABLE):
        print("WARNING: --compute_dnsmos set but speechmos unavailable — DNSMOS will be NaN.")

    selected_methods = list(args.scores)
    print(f"Scores to sweep : {selected_methods}")
    print(f"K_high          : {args.K_high}")
    print(f"n_tau           : {args.n_tau}")
    print(f"PESQ            : {'yes' if do_pesq else 'no'}")
    print(f"ESTOI           : {'yes' if do_estoi else 'no'}")
    print(f"DNSMOS          : {'yes' if do_dnsmos else 'no'}")

    # Check NISQA availability
    if "nisqa" in selected_methods and not nisqa_available():
        print("WARNING: nisqa requested but nisqa package not available — removing from sweep.")
        selected_methods = [m for m in selected_methods if m != "nisqa"]

    # ------------------------------------------------------------------
    # 1. Locate try folders
    # ------------------------------------------------------------------
    discovered = sorted(
        [d for d in args.tries_root.glob(args.dir_pattern) if d.is_dir()],
        key=_nat_key,
    )
    if not discovered:
        print(f"ERROR: no directories matching '{args.dir_pattern}' found under {args.tries_root}")
        sys.exit(1)

    K_avail = len(discovered)
    if K_avail < args.K_high:
        print(f"WARNING: --K_high={args.K_high} but only {K_avail} try folders found; "
              f"using all {K_avail}.")

    try_dirs: list = list(enumerate(discovered))
    try_dir_map: dict = {k: d for k, d in try_dirs}
    try_ks = [k for k, _ in try_dirs]
    K_use  = min(args.K_high, K_avail)

    print(f"\nFound {K_avail} try folder(s); using first {K_use} for escalation:")
    for k, d in try_dirs[:K_use]:
        print(f"  [{k}] {d.name}")

    # ------------------------------------------------------------------
    # 2. Collect file list
    # ------------------------------------------------------------------
    try0_dir  = try_dir_map[try_ks[0]]
    wav_files = sorted(
        list(try0_dir.glob("*.wav")) + list(try0_dir.glob("*.flac")),
        key=lambda p: p.name,
    )
    if not wav_files:
        print(f"ERROR: no wav/flac files in {try0_dir}")
        sys.exit(1)
    filenames = [p.name for p in wav_files]

    if args.file_list is not None:
        wanted    = {ln.strip() for ln in args.file_list.read_text().splitlines() if ln.strip()}
        filenames = [f for f in filenames if f in wanted]
        print(f"  --file_list filter: {len(filenames)} files kept.")

    N = len(filenames)
    print(f"\n{N} files in {try0_dir.name}")

    # ------------------------------------------------------------------
    # 3. Load clean / noisy caches
    # ------------------------------------------------------------------
    print("\nLoading noisy files …")
    noisy_cache: dict = {}
    for fname in tqdm(filenames, desc="  noisy", leave=False):
        p = args.noisy_dir / fname
        if not p.exists():
            cands = list(args.noisy_dir.rglob(fname))
            p = cands[0] if cands else p
        if p.exists():
            y, sr = sf.read(str(p), dtype="float32")
            if y.ndim == 2:
                y = y.mean(axis=1)
            noisy_cache[fname] = (y, int(sr))
        else:
            print(f"  WARNING: noisy file not found: {fname}")

    print("Loading clean files …")
    clean_cache: dict = {}
    for fname in tqdm(filenames, desc="  clean", leave=False):
        clean_fname = fname.split("_")[0] + ".wav" if "dB" in fname else fname
        p = args.clean_dir / clean_fname
        if p.exists():
            x, sr = sf.read(str(p), dtype="float32")
            if x.ndim == 2:
                x = x.mean(axis=1)
            clean_cache[fname] = (x, int(sr))
        else:
            print(f"  WARNING: clean file not found: {clean_fname}")

    # ------------------------------------------------------------------
    # 4. Compute per-(file, try) scores + reporting metrics (single audio pass)
    # ------------------------------------------------------------------
    print(f"\nComputing scores and quality metrics for {K_use} tries × {N} files …")

    # Difficulty score matrices: shape (N, K_use)
    scores_mat: dict = {m: np.full((N, K_use), np.nan) for m in selected_methods}

    # Reporting metric matrices: shape (N, K_use)
    sisdr_mat  = np.full((N, K_use), np.nan)
    pesq_mat   = np.full((N, K_use), np.nan)
    estoi_mat  = np.full((N, K_use), np.nan)
    dnsmos_mat = np.full((N, K_use), np.nan)

    file_idx   = {fname: i for i, fname in enumerate(filenames)}
    records_rows = []

    for j, (k, try_dir) in enumerate(try_dirs[:K_use]):
        print(f"  Try {k}: {try_dir.name}")
        for fname in tqdm(filenames, desc=f"    try_{k}", leave=False):
            i        = file_idx[fname]
            enh_path = try_dir / fname
            if not enh_path.exists():
                continue

            x_hat, sr_h = sf.read(str(enh_path), dtype="float32")
            if x_hat.ndim == 2:
                x_hat = x_hat.mean(axis=1)

            # ---- Difficulty / gate scores (policy) ----
            if fname in noisy_cache:
                y, _ = noisy_cache[fname]
                T    = min(len(y), len(x_hat))
                sc   = _compute_all_scores(y[:T], x_hat[:T], sr_h, selected_methods)
            else:
                sc = {m: float("nan") for m in selected_methods}
            for m in selected_methods:
                scores_mat[m][i, j] = sc[m]

            # ---- Reporting metrics (never used in policy) ----
            if fname in clean_cache:
                x, sr_x = clean_cache[fname]
                T = min(len(x), len(x_hat))
                sisdr_mat[i, j] = _sisdr(x_hat[:T], x[:T])
                if do_pesq:
                    pesq_mat[i, j]  = _compute_pesq(x_hat[:T], x[:T], sr_h)
                if do_estoi:
                    estoi_mat[i, j] = _compute_estoi(x_hat[:T], x[:T], sr_h)

            if do_dnsmos:
                dnsmos_mat[i, j] = _compute_dnsmos_score(x_hat, sr_h)

            row = {
                "file_id":   fname,
                "try_index": k,
                "sisdr":     sisdr_mat[i, j],
                "pesq":      pesq_mat[i, j],
                "estoi":     estoi_mat[i, j],
                "dnsmos":    dnsmos_mat[i, j],
            }
            row.update({m: sc[m] for m in selected_methods})
            records_rows.append(row)

    raw_df  = pd.DataFrame(records_rows)
    raw_csv = args.work_dir / "per_file_per_try_scores.csv"
    raw_df.to_csv(raw_csv, index=False)
    print(f"\nRaw per-(file,try) data saved → {raw_csv}")

    # ------------------------------------------------------------------
    # 5. Tau sweep per difficulty method
    # ------------------------------------------------------------------
    print("\nRunning tau sweeps …")

    all_sweeps: list = []
    for m in selected_methods:
        print(f"  [{m}] sweeping {args.n_tau} tau values …")
        sweep_df = _tau_sweep(
            scores_mat[m],
            sisdr_mat,
            method=m,
            K_high=K_use,
            n_tau=args.n_tau,
            pesq_mat=pesq_mat   if do_pesq   else None,
            estoi_mat=estoi_mat if do_estoi  else None,
            dnsmos_mat=dnsmos_mat if do_dnsmos else None,
        )
        if sweep_df.empty:
            continue
        out_csv = args.work_dir / f"tau_sweep_{m}.csv"
        sweep_df.to_csv(out_csv, index=False)
        print(f"    Saved → {out_csv}")
        all_sweeps.append(sweep_df)

    if not all_sweeps:
        print("ERROR: no sweep results — nothing to plot.")
        sys.exit(1)

    combined_df  = pd.concat(all_sweeps, ignore_index=True)
    combined_csv = args.work_dir / "tau_sweep_combined.csv"
    combined_df.to_csv(combined_csv, index=False)
    print(f"\nCombined sweep CSV → {combined_csv}")

    # ------------------------------------------------------------------
    # 6. Summary table at key operating points
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("THRESHOLD SWEEP SUMMARY  —  Key Operating Points")
    print("=" * 90)

    target_pcts = [0, 25, 50, 75, 100]
    cw = 11  # column width

    # Build header dynamically based on what was computed
    extra_cols = []
    if do_pesq:   extra_cols.append(("PESQ",   "mean_pesq"))
    if do_estoi:  extra_cols.append(("ESTOI",  "mean_estoi"))
    if do_dnsmos: extra_cols.append(("DNSMOS", "mean_dnsmos"))

    hdr_parts = [f"  {'method':<18}", f"{'%esc':>{cw}}", f"{'avg_K':>{cw}}",
                 f"{'SI-SDR':>{cw}}", f"{'worst10':>{cw}}", f"{'harm_rt':>{cw}}"]
    for label, _ in extra_cols:
        hdr_parts.append(f"{label:>{cw}}")
    print("".join(hdr_parts))
    print("  " + "-" * (18 + cw * (5 + len(extra_cols))))

    for m in selected_methods:
        sub = combined_df[combined_df["method"] == m].copy()
        if sub.empty:
            continue
        for tgt in target_pcts:
            idx = (sub["escalation_rate"] * 100 - tgt).abs().idxmin()
            row = sub.loc[idx]
            hr  = row["harm_rate_vs_try0"]
            hr_s = f"{hr:.3f}" if np.isfinite(hr) else "   n/a"
            line = (f"  {m:<18}"
                    f"{row['escalation_rate']*100:>{cw}.1f}"
                    f"{row['avg_K']:>{cw}.2f}"
                    f"{row['mean_sisdr']:>{cw}.3f}"
                    f"{row['worst10_mean_sisdr']:>{cw}.3f}"
                    f"{hr_s:>{cw}}")
            for _, col in extra_cols:
                v = row[col]
                line += f"{v:>{cw}.3f}" if np.isfinite(v) else f"{'n/a':>{cw}}"
            print(line)
        print()

    # ------------------------------------------------------------------
    # 7. Spearman: difficulty score vs SI-SDR (try_0 only)
    # ------------------------------------------------------------------
    print("Spearman(difficulty_score, SI-SDR) on try_0:")
    df0  = raw_df[raw_df["try_index"] == try_ks[0]].copy()
    si0  = df0["sisdr"].to_numpy()

    def _spearman(x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            return float("nan")
        x, y = x[mask], y[mask]
        n  = len(x)
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        d2 = np.sum((rx - ry) ** 2)
        return float(1.0 - 6.0 * d2 / (n * (n ** 2 - 1)))

    for m in selected_methods:
        if m not in df0.columns:
            continue
        sc0  = df0[m].to_numpy()
        r    = _spearman(sc0, si0)
        hib  = HIGHER_IS_BETTER[m]
        if np.isfinite(r):
            good = (r < 0) if not hib else (r > 0)
            tag  = "predicts difficulty ✓" if good else "anti-correlated ✗"
        else:
            tag  = "n/a"
        print(f"  {m:<22} r = {r:+.4f}   ({tag})")

    # ------------------------------------------------------------------
    # 8. Plots
    # ------------------------------------------------------------------
    print("\nGenerating plots …")
    _plot_curves(combined_df, plots_dir)

    print("\nDone.")
    print(f"  Combined CSV  → {combined_csv}")
    print(f"  Raw scores    → {raw_csv}")
    print(f"  Plots         → {plots_dir}/")


if __name__ == "__main__":
    main()
