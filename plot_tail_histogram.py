#!/usr/bin/env python3
"""Plot SI-SDR distribution histograms: baseline vs adaptive sampling.

Reads pre-computed per-utterance SI-SDR directly from _results.csv files.
All curves have the same number of utterances (one SI-SDR value per file).

Saves: voicebank/tail_histogram.jpeg
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config — one entry per operating point, matching the paper table
# ---------------------------------------------------------------------------
BASELINE = {
    "csv":   "voicebank/test_enhanced_baseline/_results.csv",
    "label": "Baseline",
    "color": "gray",
}

OPERATING_POINTS = [
    {
        "csv":   "voicebank/test_enhanced_adaptiveK_tau0.002625/_results.csv",
        "label": r"$\tau=0.0026$, ~22% esc.",
        "color": "#6ACC65",
    },
]

OUT_PATH    = "voicebank/tail_histogram.jpeg"
BINS        = 40
ALPHA       = 0.45
TAIL_PCT    = 1    # percentile to mark on tail zoom panel


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["filename"] = df["filename"].str.lstrip("./").str.lstrip("/")
    return df.set_index("filename")


baseline_df    = load_df(BASELINE["csv"])
op_dfs         = [load_df(op["csv"]) for op in OPERATING_POINTS]

# align on shared filenames
shared = baseline_df.index
for df in op_dfs:
    shared = shared.intersection(df.index)
shared = sorted(shared)

baseline_sisdr = baseline_df.loc[shared, "sisdr_enh"].to_numpy(dtype=float)
op_sisdrs      = [df.loc[shared, "sisdr_enh"].to_numpy(dtype=float) for df in op_dfs]

n = len(baseline_sisdr)
print(f"Utterances per curve: {n}")

# ---------------------------------------------------------------------------
# Shared bin edges
# ---------------------------------------------------------------------------
all_vals  = np.concatenate([baseline_sisdr] + op_sisdrs)
x_lo      = float(np.percentile(all_vals, 0.5))
x_hi      = float(np.percentile(all_vals, 99.5))
bin_edges = np.linspace(x_lo, x_hi, BINS + 1)

tail_max  = float(np.percentile(all_vals, 10))   # zoom: worst 10%
tail_lo   = float(np.percentile(all_vals, 0.1))
tail_edges = np.linspace(tail_lo, tail_max, BINS // 2 + 1)


# ---------------------------------------------------------------------------
# Plot — one row per curve (baseline on top, each tau below)
# ---------------------------------------------------------------------------
n_rows = 1 + len(OPERATING_POINTS)   # baseline + 3 taus
fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.2 * n_rows),
                         sharex="col", sharey="col")

# compute shared x-limits per column
x_full_lim = (x_lo, x_hi)
x_tail_lim = (tail_lo, tail_max)

all_rows = [{"sisdr": baseline_sisdr, **BASELINE}] + \
           [{"sisdr": s, **op} for s, op in zip(op_sisdrs, OPERATING_POINTS)]

for row_idx, entry in enumerate(all_rows):
    sisdr  = entry["sisdr"]
    color  = entry["color"]
    label  = entry["label"]
    ax_full = axes[row_idx, 0]
    ax_tail = axes[row_idx, 1]

    p10 = float(np.percentile(sisdr, TAIL_PCT))

    # baseline shown as light fill in every row for reference
    if row_idx > 0:
        ax_full.hist(baseline_sisdr, bins=bin_edges, alpha=0.25,
                     color=BASELINE["color"], density=True)
        ax_tail.hist(baseline_sisdr, bins=tail_edges, alpha=0.25,
                     color=BASELINE["color"], density=True)
        p10_base = float(np.percentile(baseline_sisdr, TAIL_PCT))
        ax_tail.axvline(p10_base, color=BASELINE["color"], linestyle=":",
                        linewidth=1.2)

    # main distribution
    ax_full.hist(sisdr, bins=bin_edges, alpha=0.85,
                 color=color, density=True)
    ax_tail.hist(sisdr, bins=tail_edges, alpha=0.85,
                 color=color, density=True)

    # P10 marker on tail panel
    ax_tail.axvline(p10, color=color, linestyle="--", linewidth=1.6,
                    label=f"P{TAIL_PCT}={p10:.2f} dB")
    ax_tail.legend(fontsize=9, loc="upper left")

    # row label on left panel
    ax_full.set_ylabel("Density", fontsize=10)
    ax_full.set_title(label, fontsize=11, loc="left", color=color, fontweight="bold")
    ax_tail.set_title("Tail zoom (worst 25%)" if row_idx == 0 else "",
                      fontsize=11)
    ax_full.grid(True, alpha=0.3)
    ax_tail.grid(True, alpha=0.3)

    if row_idx == 0:
        ax_full.set_title("Full Distribution", fontsize=11)

# x-labels only on bottom row
axes[-1, 0].set_xlabel("SI-SDR (dB)", fontsize=11)
axes[-1, 1].set_xlabel("SI-SDR (dB)", fontsize=11)

fig.suptitle(
    f"Adaptive Sampling — SI-SDR tail improvement  (n={n} utterances per curve)",
    fontsize=13, y=1.01
)
plt.tight_layout()
plt.savefig(OUT_PATH, format="jpeg", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_PATH}")

# Print P1 summary
print(f"\nP{TAIL_PCT} summary:")
print(f"  Baseline:  {np.percentile(baseline_sisdr, TAIL_PCT):.2f} dB")
for op, sisdr in zip(OPERATING_POINTS, op_sisdrs):
    print(f"  {op['label']}: {np.percentile(sisdr, TAIL_PCT):.2f} dB")


# ---------------------------------------------------------------------------
# CDF figure
# ---------------------------------------------------------------------------
def _cdf(arr):
    s = np.sort(arr)
    c = np.arange(1, len(s) + 1) / len(s)
    return s, c


all_curves = [{"sisdr": baseline_sisdr, **BASELINE}] + \
             [{"sisdr": s, **op} for s, op in zip(op_sisdrs, OPERATING_POINTS)]

fig2, (ax_cdf, ax_tail_cdf) = plt.subplots(1, 2, figsize=(12, 4.5))

# ---- full CDF ----
for entry in all_curves:
    xs, ys = _cdf(entry["sisdr"])
    ax_cdf.plot(xs, ys, color=entry["color"], linewidth=2, label=entry["label"])

ax_cdf.set_xlabel("SI-SDR (dB)", fontsize=12)
ax_cdf.set_ylabel("CDF", fontsize=12)
ax_cdf.set_title("CDF — full range", fontsize=13)
ax_cdf.legend(fontsize=10)
ax_cdf.grid(True, alpha=0.3)

# ---- tail zoom: x-axis limited to worst 15% ----
tail_x_max = float(np.percentile(all_vals, 15))
tail_x_min = float(np.percentile(all_vals, 0.1))

for entry in all_curves:
    xs, ys = _cdf(entry["sisdr"])
    ax_tail_cdf.plot(xs, ys, color=entry["color"], linewidth=2.5, label=entry["label"])

# mark P1 for each curve
for entry in all_curves:
    p1 = float(np.percentile(entry["sisdr"], TAIL_PCT))
    cdf_at_p1 = TAIL_PCT / 100
    ax_tail_cdf.axvline(p1, color=entry["color"], linestyle="--",
                        linewidth=1.4, alpha=0.8)
    ax_tail_cdf.annotate(f"P{TAIL_PCT}={p1:.1f}",
                         xy=(p1, cdf_at_p1),
                         xytext=(p1 + 0.3, cdf_at_p1 + 0.015),
                         fontsize=8, color=entry["color"])

ax_tail_cdf.set_xlim(tail_x_min, tail_x_max)
ax_tail_cdf.set_ylim(0, 0.20)
ax_tail_cdf.set_xlabel("SI-SDR (dB)", fontsize=12)
ax_tail_cdf.set_ylabel("CDF", fontsize=12)
ax_tail_cdf.set_title("CDF — tail zoom (worst 15%)", fontsize=13)
ax_tail_cdf.legend(fontsize=10)
ax_tail_cdf.grid(True, alpha=0.3)

fig2.suptitle(
    f"Adaptive Sampling — SI-SDR CDF  (n={n} utterances per curve)",
    fontsize=13
)
plt.tight_layout()
cdf_path = OUT_PATH.replace("tail_histogram", "tail_cdf")
plt.savefig(cdf_path, format="jpeg", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {cdf_path}")


# ---------------------------------------------------------------------------
# Delta SI-SDR histogram  (per-utterance: adaptive − baseline)
# ---------------------------------------------------------------------------
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5))

for op, sisdr in zip(OPERATING_POINTS, op_sisdrs):
    delta = sisdr - baseline_sisdr

    pct_improved = 100 * np.mean(delta > 0)
    pct_harmed   = 100 * np.mean(delta < 0)
    mean_delta   = float(np.mean(delta))
    p1_base      = float(np.percentile(baseline_sisdr, 1))
    p1_adap      = float(np.percentile(sisdr, 1))

    # ---- left: full delta histogram ----
    ax = axes3[0]
    bins_d = np.linspace(np.percentile(delta, 0.5), np.percentile(delta, 99.5), 50)
    ax.hist(delta, bins=bins_d, color=op["color"], alpha=0.85, density=True)
    ax.axvline(0,          color="black", linewidth=1.2, linestyle="--")
    ax.axvline(mean_delta, color="red",   linewidth=1.4, linestyle="-",
               label=f"mean Δ={mean_delta:+.2f} dB")
    ax.set_xlabel("ΔSI-SDR = adaptive − baseline (dB)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"ΔSI-SDR — {op['label']}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.text(0.97, 0.95, f"improved: {pct_improved:.0f}%\nharmed:   {pct_harmed:.0f}%",
            transform=ax.transAxes, fontsize=10, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    # ---- right: zoom on utterances that were harmed (delta < 0) ----
    ax2 = axes3[1]
    harmed_delta = delta[delta < 0]
    if len(harmed_delta) > 0:
        bins_h = np.linspace(harmed_delta.min(), 0, 30)
        ax2.hist(harmed_delta, bins=bins_h, color="salmon", alpha=0.85, density=True)
    ax2.set_xlabel("ΔSI-SDR (harmed utterances only)", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title(f"Harmed utterances ({pct_harmed:.0f}% of test set)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.97, 0.95,
             f"P1 baseline: {p1_base:.2f} dB\nP1 adaptive: {p1_adap:.2f} dB\nΔP1={p1_adap-p1_base:+.2f} dB",
             transform=ax2.transAxes, fontsize=10, va="top", ha="right",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

fig3.suptitle(
    f"Per-utterance ΔSI-SDR (adaptive − baseline)  (n={len(delta)} utterances)",
    fontsize=13
)
plt.tight_layout()
delta_path = OUT_PATH.replace("tail_histogram", "delta_sisdr")
plt.savefig(delta_path, format="jpeg", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {delta_path}")


# ---------------------------------------------------------------------------
# Scatter: ΔSI-SDR vs baseline SI-SDR
# ---------------------------------------------------------------------------
fig4, ax = plt.subplots(figsize=(8, 5))

for op, sisdr in zip(OPERATING_POINTS, op_sisdrs):
    delta = sisdr - baseline_sisdr
    ax.scatter(baseline_sisdr, delta, color=op["color"],
               alpha=0.35, s=12, label=op["label"])

ax.axhline(0, color="black", linewidth=1.2, linestyle="--")

# trend line
coeffs = np.polyfit(baseline_sisdr, delta, 1)
x_fit  = np.linspace(baseline_sisdr.min(), baseline_sisdr.max(), 200)
ax.plot(x_fit, np.polyval(coeffs, x_fit),
        color="red", linewidth=1.8, linestyle="-",
        label=f"trend  slope={coeffs[0]:+.3f}")

ax.set_xlabel("Baseline SI-SDR (dB)", fontsize=12)
ax.set_ylabel("ΔSI-SDR = adaptive − baseline (dB)", fontsize=12)
ax.set_title("Where does adaptive sampling help?", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

fig4.suptitle(
    f"ΔSI-SDR vs baseline difficulty  (n={len(delta)} utterances)",
    fontsize=13
)
plt.tight_layout()
scatter_path = OUT_PATH.replace("tail_histogram", "delta_vs_baseline")
plt.savefig(scatter_path, format="jpeg", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {scatter_path}")
