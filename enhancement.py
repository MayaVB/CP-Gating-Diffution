import glob
import random
import csv
import numpy as np
import torch
from tqdm import tqdm
from os import makedirs
from soundfile import write
from torchaudio import load
from os.path import join, dirname
from argparse import ArgumentParser
from librosa import resample

# Set CUDA architecture list
from sgmse.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()

from sgmse.model import ScoreModel
from sgmse.util.other import pad_spec


def _set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the test data')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    parser.add_argument("--ckpt", type=str,  help='Path to model checkpoint')
    parser.add_argument("--sampler_type", type=str, default="pc", help="Sampler type for the PC sampler.")
    parser.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald", help="Corrector class for the PC sampler.")
    parser.add_argument("--corrector_steps", type=int, default=1, help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynmaics")
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")
    parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum process time (0.03 by default)")
    parser.add_argument("--gate_calibration", action="store_true", help="Collect speech gate scores for calibration")
    parser.add_argument("--gate_enable", action="store_true", help="Enable speech quality gate")
    parser.add_argument("--gate_tau", type=float, default=-0.2072, help="Gate threshold: trigger resample if score > tau")
    parser.add_argument("--gate_seed", type=int, default=0, help="Base random seed for reproducible gated sampling")
    parser.add_argument("--gate_resample_once", action="store_true", help="[Deprecated] Use --gate_max_tries 2 instead")
    parser.add_argument("--gate_max_tries", type=int, default=1, help="Max sampling attempts per utterance (K=1: no retries, K>1: retry if triggered)")
    parser.add_argument("--gate_early_stop", action="store_true", help="Stop retrying as soon as a sample passes tau; otherwise always do K tries")
    args = parser.parse_args()

    # Backward compatibility: --gate_resample_once maps to --gate_max_tries 2
    if args.gate_resample_once:
        print("Warning: --gate_resample_once is deprecated. Use --gate_max_tries 2 instead.")
        if args.gate_max_tries == 1:
            args.gate_max_tries = 2

    # Load score model
    model = ScoreModel.load_from_checkpoint(args.ckpt, map_location=args.device)
    model.t_eps = args.t_eps
    model.eval()

    # Get list of noisy files
    noisy_files = []
    noisy_files += sorted(glob.glob(join(args.test_dir, '*.wav')))
    noisy_files += sorted(glob.glob(join(args.test_dir, '**', '*.wav')))
    noisy_files += sorted(glob.glob(join(args.test_dir, '*.flac')))
    noisy_files += sorted(glob.glob(join(args.test_dir, '**', '*.flac')))

    # Check if the model is trained on 48 kHz data
    if model.backbone == 'ncsnpp_48k':
        target_sr = 48000
        pad_mode = "reflection"
    elif model.backbone == 'ncsnpp_v2':
        target_sr = 16000
        pad_mode = "reflection"
    else:
        target_sr = 16000
        pad_mode = "zero_pad"

    if args.gate_calibration:
        from utils.speech_gate import compute_speech_gate_score
        gate_scores = []

    if args.gate_enable:
        from utils.speech_gate import compute_speech_gate_score
        gate_total = 0
        gate_triggered = 0
        gate_accepted_early = 0
        gate_maxed_out = 0
        gate_replaced = 0
        gate_tries_total = 0
        gate_log = []

    # Enhance files
    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.replace(args.test_dir, "")
        filename = filename[1:] if filename.startswith("/") else filename

        # Load wav
        y, sr = load(noisy_file)

        # Resample if necessary
        if sr != target_sr:
            y = torch.tensor(resample(y.numpy(), orig_sr=sr, target_sr=target_sr))

        T_orig = y.size(1)   

        # Normalize
        norm_factor = y.abs().max()
        y = y / norm_factor
        
        # Prepare DNN input
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.to(args.device))), 0)
        Y = pad_spec(Y, mode=pad_mode)
        
        # Helper closure: build a fresh sampler for the current Y
        def _build_sampler():
            if model.sde.__class__.__name__ == 'OUVESDE':
                if args.sampler_type == 'pc':
                    return model.get_pc_sampler('reverse_diffusion', args.corrector, Y.to(args.device), N=args.N,
                        corrector_steps=args.corrector_steps, snr=args.snr)
                elif args.sampler_type == 'ode':
                    return model.get_ode_sampler(Y.to(args.device), N=args.N)
                else:
                    raise ValueError(f"Sampler type {args.sampler_type} not supported")
            elif model.sde.__class__.__name__ == 'SBVESDE':
                sampler_type = 'ode' if args.sampler_type == 'pc' else args.sampler_type
                return model.get_sb_sampler(sde=model.sde, y=Y.cuda(), sampler_type=sampler_type)
            else:
                raise ValueError(f"SDE {model.sde.__class__.__name__} not supported")

        # Reverse sampling (seed first run only when gate is active)
        if args.gate_enable:
            _set_seeds(args.gate_seed)
        sampler = _build_sampler()
        sample, _ = sampler()
        
        # Backward transform in time domain
        x_hat = model.to_audio(sample.squeeze(), T_orig)

        # Renormalize
        x_hat = x_hat * norm_factor

        if args.gate_enable:
            y_np = y.squeeze().cpu().numpy()

            # Energy-based VAD (same as calibration)
            _energy = np.convolve(y_np ** 2, np.ones(400) / 400, mode='same')
            speech_mask = _energy > np.percentile(_energy, 60)

            s0 = compute_speech_gate_score(y_np, (x_hat / norm_factor).cpu().numpy(), speech_mask)
            gate_total += 1

            best_score = s0
            best_x_hat = x_hat
            best_try_idx = 0
            tries_used = 1
            triggered = s0 > args.gate_tau
            accepted_early = False

            if triggered:
                gate_triggered += 1
                for i in range(1, args.gate_max_tries):
                    _set_seeds(args.gate_seed + i)
                    sample_i, _ = _build_sampler()()
                    x_hat_i = model.to_audio(sample_i.squeeze(), T_orig)
                    x_hat_i = x_hat_i * norm_factor
                    s_i = compute_speech_gate_score(y_np, (x_hat_i / norm_factor).cpu().numpy(), speech_mask)
                    tries_used += 1

                    if s_i < best_score:
                        best_score = s_i
                        best_x_hat = x_hat_i
                        best_try_idx = i

                    if args.gate_early_stop and s_i <= args.gate_tau:
                        accepted_early = True
                        gate_accepted_early += 1
                        break

                if not accepted_early:
                    gate_maxed_out += 1

                x_hat = best_x_hat

            if best_try_idx != 0:
                gate_replaced += 1
            gate_tries_total += tries_used

            gate_log.append({
                "filename": filename,
                "s0": round(s0, 6),
                "tries_used": tries_used,
                "accepted_early": int(accepted_early),
                "best_score": round(best_score, 6),
                "best_try_idx": best_try_idx,
            })

        if args.gate_calibration:
            # Use normalized versions of both signals so the ratio is meaningful
            y_np = y.squeeze().cpu().numpy()                    # shape [T], normalized
            x_hat_np = (x_hat / norm_factor).cpu().numpy()     # shape [T], normalized

            # Energy-based VAD: moving average of y**2 over ~400 samples
            window = 400
            energy = np.convolve(y_np ** 2, np.ones(window) / window, mode='same')
            threshold = np.percentile(energy, 60)
            speech_mask = energy > threshold

            score = compute_speech_gate_score(y_np, x_hat_np, speech_mask)
            gate_scores.append(score)

        # Write enhanced wav file
        makedirs(dirname(join(args.enhanced_dir, filename)), exist_ok=True)
        write(join(args.enhanced_dir, filename), x_hat.cpu().numpy(), target_sr)

    if args.gate_enable:
        trigger_rate = gate_triggered / gate_total if gate_total > 0 else 0.0
        replaced_rate = gate_replaced / gate_total if gate_total > 0 else 0.0
        early_rate = gate_accepted_early / gate_triggered if gate_triggered > 0 else 0.0
        maxed_rate = gate_maxed_out / gate_triggered if gate_triggered > 0 else 0.0
        avg_tries = gate_tries_total / gate_total if gate_total > 0 else 0.0
        print("\n--- Speech Gate Summary ---")
        print(f"Total:            {gate_total}")
        print(f"Triggered:        {gate_triggered}  ({trigger_rate:.1%})")
        print(f"Accepted early:   {gate_accepted_early}  ({early_rate:.1%} of triggered)")
        print(f"Maxed out:        {gate_maxed_out}  ({maxed_rate:.1%} of triggered)")
        print(f"Replaced:         {gate_replaced}  ({replaced_rate:.1%})")
        print(f"Avg tries used:   {avg_tries:.2f}")
        log_path = join(args.enhanced_dir, "_gate_log.csv")
        with open(log_path, "w", newline="") as _f:
            _writer = csv.DictWriter(_f, fieldnames=["filename", "s0", "tries_used", "accepted_early", "best_score", "best_try_idx"])
            _writer.writeheader()
            _writer.writerows(gate_log)
        print(f"Gate log saved to {log_path}")

    if args.gate_calibration:
        gate_scores = np.array(gate_scores)
        np.save("speech_gate_scores.npy", gate_scores)
        percentiles = np.percentile(gate_scores, [50, 75, 90, 95, 99])
        print("\n--- Speech Gate Calibration Summary ---")
        print(f"N={len(gate_scores)}")
        print(f"Mean:  {gate_scores.mean():.4f}")
        print(f"Std:   {gate_scores.std():.4f}")
        print(f"p50:   {percentiles[0]:.4f}")
        print(f"p75:   {percentiles[1]:.4f}")
        print(f"p90:   {percentiles[2]:.4f}")
        print(f"p95:   {percentiles[3]:.4f}")
        print(f"p99:   {percentiles[4]:.4f}")
        print("Scores saved to speech_gate_scores.npy")
