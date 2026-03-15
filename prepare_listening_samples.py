"""
Prepare listening samples: 10% worst SI-SDR (baseline) vs best-of-K (oracle resampled).

For each selected file, copies 4 versions to listen_samples/:
  {name}_noisy.wav
  {name}_clean.wav
  {name}_baseline.wav   (worst SI-SDR, baseline enhanced)
  {name}_best_k.wav     (best-of-K oracle resampled, improved SI-SDR)

Selection strategy: picks ~10 files spread across the worst-SI-SDR range,
preferring those with large improvement from best-of-K.
"""
import pandas as pd
import numpy as np
import shutil
import os

# Paths
BASE = "/home/dsi/mayavb/PythonProjects/sgmse/voicebank"
SUMMARY_CSV = os.path.join(BASE, "oracle_exp/oracle_summary.csv")
NOISY_DIR = os.path.join(BASE, "test_noisy")
CLEAN_DIR = os.path.join(BASE, "test_clean")
BASELINE_DIR = os.path.join(BASE, "test_enhanced_baseline")
ORACLE_ROOT = os.path.join(BASE, "oracle_exp")
OUT_DIR = os.path.join(BASE, "listen_samples")

N_SAMPLES = 10  # number of files to prepare

# Load summary, filter to worst subset
df = pd.read_csv(SUMMARY_CSV)
df = df[df["subset"] == "worst"].copy()
df["improvement"] = df["best_of_K"] - df["baseline_sisdr_enh"]
df = df.sort_values("baseline_sisdr_enh").reset_index(drop=True)

print(f"Worst subset: {len(df)} files")
print(f"Baseline SI-SDR range: {df['baseline_sisdr_enh'].min():.2f} .. {df['baseline_sisdr_enh'].max():.2f}")
print(f"Improvement range: {df['improvement'].min():.2f} .. {df['improvement'].max():.2f}")
print()

# Select N_SAMPLES files: evenly spaced by rank but weighted toward worst + high improvement
# Strategy: take the worst 4, then some middle with big improvement, then a "less bad" example
n_worst = 4
worst = df.head(n_worst)

# From the remaining, take those with highest improvement
remaining = df.iloc[n_worst:]
n_highimp = N_SAMPLES - n_worst - 1
high_imp = remaining.nlargest(n_highimp, "improvement")

# One "less bad" example (from top 20% by baseline SI-SDR)
top20 = df.tail(max(1, len(df) // 5))
less_bad = top20.nlargest(1, "improvement")

selected = pd.concat([worst, high_imp, less_bad]).drop_duplicates(subset="filename")
selected = selected.sort_values("baseline_sisdr_enh").reset_index(drop=True)

print(f"Selected {len(selected)} files:")
print(selected[["filename", "baseline_sisdr_enh", "best_of_K", "improvement"]].to_string(index=False))
print()

# Determine which try had best_of_K for each file
try_cols = [c for c in df.columns if c.startswith("try_")]

def find_best_try(row):
    vals = {col: row[col] for col in try_cols}
    return max(vals, key=vals.get)

selected["best_try"] = selected.apply(find_best_try, axis=1)

# Copy files
os.makedirs(OUT_DIR, exist_ok=True)

for _, row in selected.iterrows():
    fname = row["filename"]
    stem = os.path.splitext(fname)[0]
    best_try_dir = os.path.join(ORACLE_ROOT, f"worst_{row['best_try']}")

    sources = {
        "noisy":    os.path.join(NOISY_DIR, fname),
        "clean":    os.path.join(CLEAN_DIR, fname),
        "baseline": os.path.join(BASELINE_DIR, fname),
        "best_k":   os.path.join(best_try_dir, fname),
    }

    ok = True
    for tag, src in sources.items():
        dst = os.path.join(OUT_DIR, f"{stem}__{tag}.wav")
        if not os.path.exists(src):
            print(f"  MISSING: {src}")
            ok = False
            continue
        shutil.copy2(src, dst)

    if ok:
        print(
            f"  {stem:20s}  baseline={row['baseline_sisdr_enh']:6.2f} dB  "
            f"best_k={row['best_of_K']:6.2f} dB  "
            f"delta=+{row['improvement']:.2f} dB  "
            f"(from {row['best_try']})"
        )

print(f"\nDone. Files in: {OUT_DIR}")
