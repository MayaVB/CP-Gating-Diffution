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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--clean_dir", type=str, required=True, help='Directory containing the clean data')
    parser.add_argument("--noisy_dir", type=str, required=True, help='Directory containing the noisy data')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    args = parser.parse_args()

    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [], "si_sar": [],
            "si_sdr_noisy": [], "si_sdr_enh": [], "delta_si_sdr": []}

    # Evaluate standard metrics
    noisy_files = []
    noisy_files += sorted(glob(join(args.noisy_dir, '*.wav')))
    noisy_files += sorted(glob(join(args.noisy_dir, '**', '*.wav')))
    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.replace(args.noisy_dir, "")[1:]
        if 'dB' in filename:
            clean_filename = filename.split("_")[0] + ".wav"
        else:
            clean_filename = filename
        x, sr_x = read(join(args.clean_dir, clean_filename))
        y, sr_y = read(join(args.noisy_dir, filename))
        x_hat, sr_x_hat = read(join(args.enhanced_dir, filename))
        assert sr_x == sr_y == sr_x_hat
        n = y - x 
        x_hat_16k = librosa.resample(x_hat, orig_sr=sr_x_hat, target_sr=16000) if sr_x_hat != 16000 else x_hat
        x_16k = librosa.resample(x, orig_sr=sr_x, target_sr=16000) if sr_x != 16000 else x
        data["filename"].append(filename)
        data["pesq"].append(pesq(16000, x_16k, x_hat_16k, 'wb'))
        data["estoi"].append(stoi(x, x_hat, sr_x, extended=True))
        si_sdr_enh, si_sir_enh, si_sar_enh = energy_ratios(x_hat, x, n)
        si_sdr_noisy = energy_ratios(y, x, n)[0]
        data["si_sdr"].append(si_sdr_enh)
        data["si_sir"].append(si_sir_enh)
        data["si_sar"].append(si_sar_enh)
        data["si_sdr_noisy"].append(si_sdr_noisy)
        data["si_sdr_enh"].append(si_sdr_enh)
        data["delta_si_sdr"].append(si_sdr_enh - si_sdr_noisy)

    # Save results as DataFrame    
    df = pd.DataFrame(data)

    # Print results
    print("PESQ: {:.2f} ± {:.2f}".format(*mean_std(df["pesq"].to_numpy())))
    print("ESTOI: {:.2f} ± {:.2f}".format(*mean_std(df["estoi"].to_numpy())))
    print("SI-SDR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sdr"].to_numpy())))
    print("SI-SIR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sir"].to_numpy())))
    print("SI-SAR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sar"].to_numpy())))

    # Save average results to file
    log = open(join(args.enhanced_dir, "_avg_results.txt"), "w")
    log.write("PESQ: {:.2f} ± {:.2f}".format(*mean_std(df["pesq"].to_numpy())) + "\n")
    log.write("ESTOI: {:.2f} ± {:.2f}".format(*mean_std(df["estoi"].to_numpy())) + "\n")
    log.write("SI-SDR: {:.1f} ± {:.2f}".format(*mean_std(df["si_sdr"].to_numpy())) + "\n")
    log.write("SI-SIR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sir"].to_numpy())) + "\n")
    log.write("SI-SAR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sar"].to_numpy())) + "\n")

    # Save DataFrame as csv file
    df.to_csv(join(args.enhanced_dir, "_results.csv"), index=False)

    # Tail SI-SDR stats (per-utterance arrays)
    enh    = np.array(data["si_sdr_enh"])
    deltas = np.array(data["delta_si_sdr"])
    enh_p10,   enh_p5,   enh_p1   = np.percentile(enh,    [10, 5, 1])
    delta_p10, delta_p5, delta_p1 = np.percentile(deltas, [10, 5, 1])
    tail_stats = {
        "n":                  len(deltas),
        "sisdr_enh_mean":     float(np.mean(enh)),
        "sisdr_enh_p10":      float(enh_p10),
        "sisdr_enh_p5":       float(enh_p5),
        "sisdr_enh_p1":       float(enh_p1),
        "delta_sisdr_mean":   float(np.mean(deltas)),
        "delta_sisdr_p10":    float(delta_p10),
        "delta_sisdr_p5":     float(delta_p5),
        "delta_sisdr_p1":     float(delta_p1),
    }
    print("\n--- Tail SI-SDR (Table 2) ---")
    print("n={n}".format(**tail_stats))
    print("sisdr_enh:   mean={sisdr_enh_mean:.2f}  p10={sisdr_enh_p10:.2f}  p5={sisdr_enh_p5:.2f}  p1={sisdr_enh_p1:.2f}".format(**tail_stats))
    print("delta_sisdr: mean={delta_sisdr_mean:.2f}  p10={delta_sisdr_p10:.2f}  p5={delta_sisdr_p5:.2f}  p1={delta_sisdr_p1:.2f}".format(**tail_stats))
    log.write("\n--- Tail SI-SDR (Table 2) ---\n")
    log.write("n={n}\n".format(**tail_stats))
    log.write("sisdr_enh:   mean={sisdr_enh_mean:.2f}  p10={sisdr_enh_p10:.2f}  p5={sisdr_enh_p5:.2f}  p1={sisdr_enh_p1:.2f}\n".format(**tail_stats))
    log.write("delta_sisdr: mean={delta_sisdr_mean:.2f}  p10={delta_sisdr_p10:.2f}  p5={delta_sisdr_p5:.2f}  p1={delta_sisdr_p1:.2f}\n".format(**tail_stats))
    log.close()
    _csv_path  = join(args.enhanced_dir, "tail_sisdr_perutt.csv")
    _json_path = join(args.enhanced_dir, "tail_sisdr_stats.json")
    pd.DataFrame({
        "filename":    data["filename"],
        "sisdr_noisy": data["si_sdr_noisy"],
        "sisdr_enh":   data["si_sdr_enh"],
        "delta_sisdr": data["delta_si_sdr"],
    }).to_csv(_csv_path, index=False)
    with open(_json_path, "w") as _f:
        json.dump(tail_stats, _f, indent=2)

    # Verification
    import os as _os
    print("\n--- Verification ---")
    print(f"tail_sisdr_perutt.csv : {_os.path.abspath(_csv_path)}")
    print(f"tail_sisdr_stats.json : {_os.path.abspath(_json_path)}")
    print(f"len(sisdr_enh)={len(enh)}  len(deltas)={len(deltas)}  n={tail_stats['n']}  match={len(enh)==len(deltas)==tail_stats['n']}")
    print(f"sisdr_enh  min={enh.min():.2f}  max={enh.max():.2f}")
    print(f"delta_sisdr min={deltas.min():.2f}  max={deltas.max():.2f}")
    print("First 3 rows (filename | sisdr_noisy | sisdr_enh | delta_sisdr):")
    for _i in range(min(3, len(data["filename"]))):
        print(f"  {data['filename'][_i]}  |  {data['si_sdr_noisy'][_i]:.2f}  |  {data['si_sdr_enh'][_i]:.2f}  |  {data['delta_si_sdr'][_i]:.2f}")
