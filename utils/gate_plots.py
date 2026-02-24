"""Gate analysis plots for calibration diagnostics."""
import os
import shutil
from typing import List

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np


def plot_gate_statistics(traj_logs: List, out_dir: str, enhanced_dir: str = None) -> None:
    """Generate and save three gate calibration diagnostic plots.

    Plots written to out_dir:
        hist_G.png            — histogram of G = max_k g_k per trajectory
        hist_g_steps.png      — per-step g_k histograms at 5 selected steps
        quantiles_vs_step.png — median / p90 / p95 of g_k vs diffusion step

    Args:
        traj_logs : List[GateTrajectoryLog] — must have g_steps populated
                    (i.e. inference was run with --gate_compute_tau)
        out_dir   : destination directory (created if absent)
    """
    os.makedirs(out_dir, exist_ok=True)

    valid = [tl for tl in traj_logs if tl.g_steps]
    if not valid:
        print("gate_plots: no per-step scores found in traj_logs; skipping plots.")
        return

    # Crop all trajectories to the shortest so we get a rectangular [N, K] matrix.
    K = min(len(tl.g_steps) for tl in valid)
    mat = np.array([tl.g_steps[:K] for tl in valid], dtype=np.float64)  # [N, K]
    N = len(valid)

    # ── Plot 1: histogram of G = max_k g_k ──────────────────────────────────
    G_vals = mat.max(axis=1)                        # [N]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(G_vals, bins=40, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(float(np.median(G_vals)), color="orange", linestyle="--",
               linewidth=1.2, label=f"median = {np.median(G_vals):.3f}")
    ax.set_xlabel("G = max$_k$ g$_k$  (leakage ratio)")
    ax.set_ylabel("Count")
    ax.set_title(f"Trajectory score distribution  (N = {N}, K = {K} steps)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_G.png"), dpi=120)
    plt.close(fig)

    # ── Plot 2: per-step histograms at selected step indices ─────────────────
    sel = _selected_step_indices(K)
    n_sel = len(sel)
    fig, axes = plt.subplots(1, n_sel, figsize=(3 * n_sel, 4), sharey=False)
    if n_sel == 1:
        axes = [axes]
    for ax, k in zip(axes, sel):
        ax.hist(mat[:, k], bins=30, color="coral", edgecolor="white", linewidth=0.5)
        ax.set_title(f"step {k}")
        ax.set_xlabel("g$_k$")
        if k == sel[0]:
            ax.set_ylabel("Count")
    fig.suptitle("Per-step score histograms at selected steps", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_g_steps.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 3: quantile curves vs step index ────────────────────────────────
    step_idx = np.arange(K)
    p50 = np.percentile(mat, 50, axis=0)
    p90 = np.percentile(mat, 90, axis=0)
    p95 = np.percentile(mat, 95, axis=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(step_idx, p50, label="median (p50)", color="steelblue", linewidth=1.5)
    ax.plot(step_idx, p90, label="p90",           color="orange",    linewidth=1.5)
    ax.plot(step_idx, p95, label="p95",           color="firebrick", linewidth=1.5)
    ax.set_xlabel("Diffusion step index k")
    ax.set_ylabel("g$_k$  (leakage ratio)")
    ax.set_title("Gate score quantiles vs diffusion step")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "quantiles_vs_step.png"), dpi=120)
    plt.close(fig)

    # ── WAV examples and worst-score lists at selected steps ─────────────────
    _save_step_wav_examples(valid, mat, sel, out_dir, enhanced_dir=enhanced_dir)
    _save_worst_wavs(valid, mat, sel, out_dir)

    print(
        f"Gate plots saved to {out_dir}: "
        "hist_G.png, hist_g_steps.png, quantiles_vs_step.png"
    )


def plot_delta_G(delta_vals: List[float], plots_dir: str) -> None:
    """Histogram of delta_G = G_try0 - G_best over samples that had restarts.

    Positive delta_G = restart improved the leakage score.
    Saves hist_delta_G.png to plots_dir.
    """
    os.makedirs(plots_dir, exist_ok=True)
    arr = np.array(delta_vals, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(arr, bins=40, color="mediumseagreen", edgecolor="white", linewidth=0.5)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, label="no change")
    ax.axvline(float(np.mean(arr)), color="orange", linestyle="--",
               linewidth=1.2, label=f"mean = {np.mean(arr):.3f}")
    ax.set_xlabel("delta_G = G_try0 − G_best  (positive = improvement)")
    ax.set_ylabel("Count")
    ax.set_title(f"Restart improvement (n = {len(arr)} samples with restarts)")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(plots_dir, "hist_delta_G.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"delta_G histogram saved to {out}")


def _save_step_wav_examples(valid: List, mat: "np.ndarray", sel: List[int],
                            out_dir: str, enhanced_dir: str = None) -> None:
    """For each selected step, copy the trajectory whose g_k is closest to the
    median at that step.  Files land in out_dir/step_examples/.
    Skips silently if the source WAV does not exist.
    """
    ex_dir = os.path.join(out_dir, "step_examples")
    os.makedirs(ex_dir, exist_ok=True)
    wav_root = enhanced_dir if enhanced_dir else out_dir
    for k in sel:
        scores_k = mat[:, k]
        median_val = float(np.median(scores_k))
        nearest_idx = int(np.argmin(np.abs(scores_k - median_val)))
        example_id = valid[nearest_idx].example_id or ""
        src = os.path.join(wav_root, example_id)
        if not os.path.isfile(src):
            print(f"gate_plots: WAV not found, skipping step {k} example: {src}")
            continue
        # Flatten any subdirectory separators so the filename stays flat
        flat_stem = example_id.replace(os.sep, "_").replace("/", "_")
        dst = os.path.join(ex_dir, f"step_{k:03d}_median_{flat_stem}")
        shutil.copy2(src, dst)
    print(f"Step WAV examples saved to {ex_dir}/")


def _save_worst_wavs(valid: List, mat: "np.ndarray", sel: List[int],
                     out_dir: str, worst_pct: float = 0.20) -> None:
    """For each selected step, write a ranked .txt of the worst-scoring
    trajectories (top worst_pct by g_k).  Files land in out_dir/worst_wavs/.

    Each line: <score TAB filename>  sorted descending (worst first).
    """
    worst_dir = os.path.join(out_dir, "worst_wavs")
    os.makedirs(worst_dir, exist_ok=True)
    N = mat.shape[0]
    n_worst = max(1, int(np.ceil(N * worst_pct)))
    for k in sel:
        scores_k = mat[:, k]
        # argsort ascending → take last n_worst, then reverse for descending
        worst_indices = np.argsort(scores_k)[-n_worst:][::-1]
        txt_path = os.path.join(worst_dir, f"step_{k:03d}_worst{int(worst_pct * 100)}pct.txt")
        with open(txt_path, "w") as f:
            f.write(f"# {n_worst}/{N} worst trajectories at step {k} "
                    f"(top {worst_pct:.0%} by g_k score, descending)\n")
            for idx in worst_indices:
                f.write(f"{scores_k[idx]:.6f}\t{valid[idx].example_id}\n")
    print(f"Worst-WAV lists ({int(worst_pct * 100)}% per step) saved to {worst_dir}/")


def _selected_step_indices(K: int) -> List[int]:
    """Return up to 5 deduplicated step indices: [0, K//4, K//2, 3K//4, K-1]."""
    if K <= 1:
        return [0]
    candidates = [0, K // 4, K // 2, (3 * K) // 4, K - 1]
    seen: set = set()
    result = []
    for idx in candidates:
        if idx not in seen:
            seen.add(idx)
            result.append(idx)
    return result
