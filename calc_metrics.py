import os
from os.path import join
from glob import glob
from argparse import ArgumentParser
from soundfile import read
from tqdm import tqdm
from pesq import pesq
import json
import numpy as np
import pandas as pd
import librosa

from pystoi import stoi

from sgmse.util.other import energy_ratios, mean_std


def sisdr(est: np.ndarray, ref: np.ndarray) -> float:
    """Scale-invariant SDR (dB): SI-SDR(est, ref).

    Projects est onto ref to obtain the target signal, then computes the ratio
    of target energy to error energy.  Only (est, ref) are used — the mixture
    or noise signal is NOT part of this formula.

      alpha    = <est, ref> / ||ref||²
      s_target = alpha * ref
      e_noise  = est - s_target
      SI-SDR   = 10 · log10( ||s_target||² / ||e_noise||² )   [dB]
    """
    ref = np.asarray(ref, dtype=float)
    est = np.asarray(est, dtype=float)
    alpha    = np.dot(est, ref) / np.dot(ref, ref)
    s_target = alpha * ref
    e_noise  = est - s_target
    return float(10.0 * np.log10(np.dot(s_target, s_target) / np.dot(e_noise, e_noise)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--clean_dir", type=str, required=True, help='Directory containing the clean data')
    parser.add_argument("--noisy_dir", type=str, required=True, help='Directory containing the noisy data')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    args = parser.parse_args()

    records = []  # list[dict]; add new metric keys here (e.g. "pesq", "dnsmos") when ready

    # Evaluate standard metrics
    noisy_files = []
    noisy_files += sorted(glob(join(args.noisy_dir, '*.wav')))
    noisy_files += sorted(glob(join(args.noisy_dir, '**', '*.wav')))
    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.replace(args.noisy_dir, "")[1:]
        enh_path = join(args.enhanced_dir, filename)
        if not os.path.isfile(enh_path):
            continue  # subset run: only score files that were actually enhanced
        if 'dB' in filename:
            clean_filename = filename.split("_")[0] + ".wav"
        else:
            clean_filename = filename
        x, sr_x = read(join(args.clean_dir, clean_filename))
        y, sr_y = read(join(args.noisy_dir, filename))
        x_hat, sr_x_hat = read(enh_path)
        assert sr_x == sr_y == sr_x_hat
        n = y - x
        x_hat_16k = librosa.resample(x_hat, orig_sr=sr_x_hat, target_sr=16000) if sr_x_hat != 16000 else x_hat
        x_16k     = librosa.resample(x,     orig_sr=sr_x,     target_sr=16000) if sr_x     != 16000 else x
        _, si_sir_enh, si_sar_enh = energy_ratios(x_hat, x, n)
        sisdr_enh_val   = sisdr(x_hat, x)   # SI-SDR(enhanced, clean) — absolute
        sisdr_noisy_val = sisdr(y, x)        # SI-SDR(noisy,    clean) — for delta verification only
        records.append({
            "filename":    filename,
            "sisdr_enh":   sisdr_enh_val,
            "sisdr_noisy": sisdr_noisy_val,
            "delta_sisdr": sisdr_enh_val - sisdr_noisy_val,
            "pesq":        pesq(16000, x_16k, x_hat_16k, 'wb'),
            "estoi":       stoi(x, x_hat, sr_x, extended=True),
            "si_sir":      si_sir_enh,
            "si_sar":      si_sar_enh,
            # "dnsmos":  ...,  # add when ready
        })

    # Save results as DataFrame
    df = pd.DataFrame(records)

    # Print results
    print("SI-SDR: {:.1f} ± {:.1f}".format(*mean_std(df["sisdr_enh"].to_numpy())))

    # Save average results to file
    log = open(join(args.enhanced_dir, "_avg_results.txt"), "w")
    log.write("SI-SDR: {:.1f} ± {:.2f}".format(*mean_std(df["sisdr_enh"].to_numpy())) + "\n")

    # Save DataFrame as csv file
    df.to_csv(join(args.enhanced_dir, "_results.csv"), index=False)

    # Tail SI-SDR stats (per-utterance arrays)
    # Table 2 uses sisdr_enh (ABSOLUTE SI-SDR enhanced vs clean) only.
    # delta_sisdr is stored for sanity checking but NOT reported in Table 2.
    enh   = df["sisdr_enh"].to_numpy()
    noisy = df["sisdr_noisy"].to_numpy()
    delta = df["delta_sisdr"].to_numpy()
    enh_p10, enh_p5, enh_p1 = np.percentile(enh, [10, 5, 1], method="linear")
    tail_stats = {
        "n":              len(enh),
        "sisdr_enh_mean": float(np.mean(enh)),
        "sisdr_enh_p10":  float(enh_p10),
        "sisdr_enh_p5":   float(enh_p5),
        "sisdr_enh_p1":   float(enh_p1),
    }
    table2_line = (
        f"sisdr_enh (absolute):  "
        f"mean={tail_stats['sisdr_enh_mean']:.2f}  "
        f"p10={tail_stats['sisdr_enh_p10']:.2f}  "
        f"p5={tail_stats['sisdr_enh_p5']:.2f}  "
        f"p1={tail_stats['sisdr_enh_p1']:.2f}"
    )
    latex_line = (
        f"LaTeX row:  "
        f"{tail_stats['sisdr_enh_mean']:.1f} & "
        f"{tail_stats['sisdr_enh_p10']:.1f} & "
        f"{tail_stats['sisdr_enh_p5']:.1f} & "
        f"{tail_stats['sisdr_enh_p1']:.1f}"
    )
    print("\n--- Tail SI-SDR (Table 2) ---")
    print(f"n={tail_stats['n']}")
    print(table2_line)
    print(latex_line)
    log.write("\n--- Tail SI-SDR (Table 2) ---\n")
    log.write(f"n={tail_stats['n']}\n")
    log.write(table2_line + "\n")
    log.write(latex_line + "\n")
    log.close()
    _csv_path  = join(args.enhanced_dir, "tail_sisdr_perutt.csv")
    _json_path = join(args.enhanced_dir, "tail_sisdr_stats.json")
    df[["filename", "sisdr_enh", "sisdr_noisy", "delta_sisdr"]].to_csv(_csv_path, index=False)
    with open(_json_path, "w") as _f:
        json.dump(tail_stats, _f, indent=2)

    # Verification
    import os as _os
    print("\n--- Verification ---")
    print(f"tail_sisdr_perutt.csv : {_os.path.abspath(_csv_path)}")
    print(f"tail_sisdr_stats.json : {_os.path.abspath(_json_path)}")
    print(f"n={tail_stats['n']}  sisdr_enh min={enh.min():.2f}  max={enh.max():.2f}")
    print(f"sisdr_noisy mean={noisy.mean():.2f}  delta mean={delta.mean():.2f}  (sanity: sisdr_enh_mean - sisdr_noisy_mean = {np.mean(enh)-np.mean(noisy):.2f})")
    assert tail_stats['sisdr_enh_p10'] >= tail_stats['sisdr_enh_p5'] >= tail_stats['sisdr_enh_p1'], \
        "FAIL: percentiles not monotone!"
    print("Monotonicity check: p10 >= p5 >= p1  PASS")
    print("First 3 rows (filename | sisdr_noisy | sisdr_enh | delta):")
    for _i in range(min(3, len(records))):
        r = records[_i]
        print(f"  {r['filename']}  |  {r['sisdr_noisy']:.2f}  |  {r['sisdr_enh']:.2f}  |  {r['delta_sisdr']:.2f}")
