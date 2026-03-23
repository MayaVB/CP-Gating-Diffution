#!/usr/bin/env python3
"""Test: do "bad" files benefit more from higher K?

For each file, simulate best-of-K selection using wiener_residual as the
selection score (lowest score = best try), for K in {1, 3, 5, 10}.

Checks:
  1. How often does K=3 beat K=10? K=5 beat K=10?
  2. Does baseline difficulty d0 (wiener_residual of try_0) predict
     which K is optimal?
  3. Per-file: does lower baseline SI-SDR → larger gain from higher K?

Input:
  voicebank/threshold_sweep_exp_50%dataset/per_file_per_try_scores.csv

Output:
  voicebank/k_vs_quality.jpeg    — plots
  voicebank/k_vs_quality.csv     — per-file table
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
CSV_IN   = "voicebank/threshold_sweep_exp_omlsa_op3/per_file_per_try_scores.csv"
OUT_JPEG = "voicebank/k_vs_quality_omlsa.jpeg"
OUT_CSV  = "voicebank/k_vs_quality_omlsa.csv"
K_LIST   = [1, 3, 5, 10]
SCORE_COL = "omlsa_residual"   # lower = better
# ---------------------------------------------------------------------------

raw = pd.read_csv(CSV_IN)

# ---------------------------------------------------------------------------
# Build per-file table
# ---------------------------------------------------------------------------
records = []
for fid, grp in raw.groupby("file_id"):
    grp = grp.sort_values("try_index").reset_index(drop=True)

    d0       = float(grp.loc[grp["try_index"] == 0, SCORE_COL].values[0])
    sisdr_k1 = float(grp.loc[grp["try_index"] == 0, "sisdr"].values[0])

    row = {"file_id": fid, "d0": d0, "sisdr_K1": sisdr_k1}

    for K in K_LIST[1:]:   # K=1 already done
        subset   = grp[grp["try_index"] < K]
        best_idx = subset[SCORE_COL].idxmin()
        row[f"sisdr_K{K}"] = float(grp.loc[best_idx, "sisdr"])

    # which K is actually best?
    sisdr_vals = {K: row[f"sisdr_K{K}"] for K in K_LIST}
    row["best_K"]       = max(sisdr_vals, key=sisdr_vals.get)
    row["gain_K10_K1"]  = row["sisdr_K10"] - row["sisdr_K1"]
    row["gain_K3_K1"]   = row["sisdr_K3"]  - row["sisdr_K1"]
    row["gain_K5_K1"]   = row["sisdr_K5"]  - row["sisdr_K1"]
    records.append(row)

df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False)
print(f"Per-file table: {OUT_CSV}  ({len(df)} files)")

# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------
n = len(df)
print(f"\n=== Best-of-K comparison (n={n}) ===")
print(f"  Mean SI-SDR per K:")
for K in K_LIST:
    m = df[f"sisdr_K{K}"].mean()
    print(f"    K={K:2d}: {m:.3f} dB")

print(f"\n  How often does K=3 beat K=10?  {100*(df['sisdr_K3'] > df['sisdr_K10']).mean():.1f}%")
print(f"  How often does K=5 beat K=10?  {100*(df['sisdr_K5'] > df['sisdr_K10']).mean():.1f}%")
print(f"  How often does K=1 beat K=10?  {100*(df['sisdr_K1'] > df['sisdr_K10']).mean():.1f}%")

print(f"\n  Best K distribution:")
for K in K_LIST:
    pct = 100 * (df["best_K"] == K).mean()
    print(f"    K={K:2d} is best: {pct:.1f}%")

# correlation: d0 vs gain from K=10
r_d0_gain = np.corrcoef(df["d0"], df["gain_K10_K1"])[0, 1]
r_sisdr_gain = np.corrcoef(df["sisdr_K1"], df["gain_K10_K1"])[0, 1]
print(f"\n  Corr(d0,          gain K10 over K1): {r_d0_gain:+.3f}")
print(f"  Corr(baseline_sisdr, gain K10 over K1): {r_sisdr_gain:+.3f}")
print(f"  (positive corr with d0 = harder files → more gain, as expected)")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

colors = {1: "gray", 3: "#4878CF", 5: "#6ACC65", 10: "#D65F5F"}

# ---- (0,0): mean SI-SDR vs K bar chart ----
ax = axes[0, 0]
means = [df[f"sisdr_K{K}"].mean() for K in K_LIST]
bars = ax.bar([str(K) for K in K_LIST], means,
              color=[colors[K] for K in K_LIST], alpha=0.85)
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9)
ax.set_xlabel("K (best-of-K selection)", fontsize=11)
ax.set_ylabel("Mean SI-SDR (dB)", fontsize=11)
ax.set_title("Mean SI-SDR vs K", fontsize=12)
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(min(means) - 0.3, max(means) + 0.4)

# ---- (0,1): CDF of SI-SDR for each K, tail zoom ----
ax = axes[0, 1]
all_sisdrs = np.concatenate([df[f"sisdr_K{K}"].values for K in K_LIST])
tail_max = float(np.percentile(all_sisdrs, 15))
tail_min = float(np.percentile(all_sisdrs, 0.1))

for K in K_LIST:
    s = np.sort(df[f"sisdr_K{K}"].values)
    c = np.arange(1, len(s) + 1) / len(s)
    ax.plot(s, c, color=colors[K], linewidth=2, label=f"K={K}")
ax.set_xlim(tail_min, tail_max)
ax.set_ylim(0, 0.18)
ax.set_xlabel("SI-SDR (dB)", fontsize=11)
ax.set_ylabel("CDF", fontsize=11)
ax.set_title("CDF tail zoom — worst 15%", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# ---- (1,0): scatter d0 vs gain(K10 over K1), colored by best_K ----
ax = axes[1, 0]
sc = ax.scatter(df["d0"], df["gain_K10_K1"],
                c=df["best_K"], cmap="RdYlGn_r",
                alpha=0.5, s=14)
ax.axhline(0, color="black", linewidth=1, linestyle="--")
coeffs = np.polyfit(df["d0"], df["gain_K10_K1"], 1)
x_fit  = np.linspace(df["d0"].min(), df["d0"].max(), 200)
ax.plot(x_fit, np.polyval(coeffs, x_fit),
        color="red", linewidth=1.8, label=f"trend slope={coeffs[0]:+.2f}")
ax.set_xlabel("d0 = Wiener residual of try 0 (difficulty)", fontsize=11)
ax.set_ylabel("Gain of K=10 over K=1 (dB)", fontsize=11)
ax.set_title("Does difficulty predict gain from higher K?", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.colorbar(sc, ax=ax, label="best K")

# ---- (1,1): scatter baseline SI-SDR vs gain, colored by best_K ----
ax = axes[1, 1]
sc2 = ax.scatter(df["sisdr_K1"], df["gain_K10_K1"],
                 c=df["best_K"], cmap="RdYlGn_r",
                 alpha=0.5, s=14)
ax.axhline(0, color="black", linewidth=1, linestyle="--")
coeffs2 = np.polyfit(df["sisdr_K1"], df["gain_K10_K1"], 1)
x_fit2  = np.linspace(df["sisdr_K1"].min(), df["sisdr_K1"].max(), 200)
ax.plot(x_fit2, np.polyval(coeffs2, x_fit2),
        color="red", linewidth=1.8, label=f"trend slope={coeffs2[0]:+.2f}")
ax.set_xlabel("SI-SDR at K=1 (baseline quality)", fontsize=11)
ax.set_ylabel("Gain of K=10 over K=1 (dB)", fontsize=11)
ax.set_title("Does baseline quality predict gain from higher K?", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.colorbar(sc2, ax=ax, label="best K")

fig.suptitle(
    f"Best-of-K analysis: does higher K help harder files?  (n={n})",
    fontsize=13
)
plt.tight_layout()
plt.savefig(OUT_JPEG, format="jpeg", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved: {OUT_JPEG}")
