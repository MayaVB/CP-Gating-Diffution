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
    parser.add_argument("--gate_tau_path", type=str, default=None, help="Path to _calib_tau.json; enables per-step running-max rejection + restart")
    parser.add_argument("--gate_max_restarts", type=int, default=10, help="Max restarts per utterance when per-step gate triggers (used with --gate_tau_path)")
    parser.add_argument("--gate_compute_tau", action="store_true", help="Log per-step scores so G = max_k g_k; enables conformal tau calibration from gate_traj_logs")
    parser.add_argument("--gate_log_every", type=int, default=1, help="Subsample per-step score logging: store every N-th step (default: 1 = every step)")
    parser.add_argument("--gate_plot", action="store_true", help="Save gate diagnostic plots after inference (requires --gate_compute_tau and --gate_enable)")
    parser.add_argument("--gate_alpha", type=float, default=0.1, help="Miscoverage level for conformal calibration; tau is saved to _calib_tau.json when --gate_compute_tau is set")
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
        from utils.speech_gate import compute_speech_gate_score, GateTrajectoryLog
        gate_total = 0
        gate_triggered = 0
        gate_accepted_early = 0
        gate_maxed_out = 0
        gate_replaced = 0
        gate_tries_total = 0
        gate_log = []
        gate_traj_logs = []  # List[GateTrajectoryLog], one per example (example_id set on each)

    if args.gate_tau_path or args.gate_compute_tau:
        from utils.speech_gate import gate_step_score, SamplingAborted

    if args.gate_tau_path:
        from utils.conformal_calib import load_calibration_json
        _calib = load_calibration_json(args.gate_tau_path)
        _gate_tau_loaded = float(_calib["tau"])
        print(f"Per-step gate: tau={_gate_tau_loaded:.4f}  "
              f"alpha={_calib.get('alpha', '?')}  "
              f"(from {args.gate_tau_path})")

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
        # step_callback is only supported for OUVESDE + PC sampler
        def _build_sampler(step_callback=None):
            if model.sde.__class__.__name__ == 'OUVESDE':
                if args.sampler_type == 'pc':
                    return model.get_pc_sampler('reverse_diffusion', args.corrector, Y.to(args.device), N=args.N,
                        corrector_steps=args.corrector_steps, snr=args.snr,
                        step_callback=step_callback)
                elif args.sampler_type == 'ode':
                    return model.get_ode_sampler(Y.to(args.device), N=args.N)
                else:
                    raise ValueError(f"Sampler type {args.sampler_type} not supported")
            elif model.sde.__class__.__name__ == 'SBVESDE':
                sampler_type = 'ode' if args.sampler_type == 'pc' else args.sampler_type
                return model.get_sb_sampler(sde=model.sde, y=Y.cuda(), sampler_type=sampler_type)
            else:
                raise ValueError(f"SDE {model.sde.__class__.__name__} not supported")

        # Per-step score buffer: populated when gate_compute_tau is active.
        _accepted_g_steps = []

        # Build per-step cache once per utterance whenever per-step scoring is needed.
        if args.gate_tau_path or args.gate_compute_tau:
            _yf_power = (Y[0].abs() ** 2).sum(dim=(0, 1)).detach().cpu().numpy()
            _gate_cache = {
                "speech_mask_frames": _yf_power > np.percentile(_yf_power, 80),
                "eps": 1e-8,
            }

        # Reverse sampling
        if args.gate_tau_path:
            # Per-step running-max rejection with restarts
            _num_restarts = 0
            _first_reject_step = None
            _final_running_max = -float("inf")
            for _attempt in range(args.gate_max_restarts + 1):
                _is_last = (_attempt == args.gate_max_restarts)
                _running_max = [-float("inf")]  # list for mutation inside closure
                _attempt_g_steps = []           # per-step buffer; discarded on abort
                def _step_cb(_si, _rm=_running_max, _tau=_gate_tau_loaded, _il=_is_last,
                             _cache=_gate_cache, _buf=_attempt_g_steps):
                    g_k = gate_step_score(_si, _cache)
                    _rm[0] = max(_rm[0], g_k)
                    # DEBUG: per-step trace for one file
                    if filename == "p257_019.wav":
                        reject_flag = (not _il) and (_rm[0] > _tau)
                        print(f"  [DBG step] k={_si['step_idx']:3d}  g_k={g_k:.4f}  "
                              f"run_max={_rm[0]:.4f}  tau={_tau:.4f}  reject={reject_flag}")
                    if args.gate_compute_tau and _si["step_idx"] % args.gate_log_every == 0:
                        _buf.append(g_k)
                    if not _il and _rm[0] > _tau:
                        raise SamplingAborted(_si["step_idx"], _rm[0])
                _set_seeds(args.gate_seed + _attempt)
                if filename == "p257_019.wav":
                    print(f"[DBG restart] attempt={_attempt}  is_last={_is_last}  "
                          f"num_restarts_so_far={_num_restarts}")
                try:
                    sample, _ = _build_sampler(step_callback=_step_cb)()
                except SamplingAborted as _exc:
                    _num_restarts += 1
                    if _first_reject_step is None:
                        _first_reject_step = _exc.step_idx
                    # DEBUG: restart event
                    if filename == "p257_019.wav":
                        print(f"[DBG restart] ABORTED at step={_exc.step_idx}  "
                              f"run_max={_exc.running_max:.4f} > tau={_gate_tau_loaded:.4f}  "
                              f"→ attempt {_attempt} -> {_attempt+1}")
                    continue
                _accepted_g_steps = _attempt_g_steps  # commit accepted attempt's buffer
                _final_running_max = _running_max[0]
                break
        else:
            if args.gate_enable:
                _set_seeds(args.gate_seed)
            if args.gate_compute_tau:
                # Calibration-only pass: collect per-step scores, no gate decisions.
                _attempt_g_steps = []
                def _step_cb_cal(_si, _cache=_gate_cache, _buf=_attempt_g_steps):
                    if _si["step_idx"] % args.gate_log_every == 0:
                        _buf.append(gate_step_score(_si, _cache))
                sample, _ = _build_sampler(step_callback=_step_cb_cal)()
                _accepted_g_steps = _attempt_g_steps
            else:
                sample, _ = _build_sampler()()
            _num_restarts = 0
            _first_reject_step = None
            _final_running_max = None
        
        # Backward transform in time domain
        x_hat = model.to_audio(sample.squeeze(), T_orig)

        # Renormalize
        x_hat = x_hat * norm_factor

        if args.gate_enable:
            y_np = y.squeeze().cpu().numpy()

            # Energy-based VAD (same as calibration)
            _energy = np.convolve(y_np ** 2, np.ones(400) / 400, mode='same')
            speech_mask = _energy > np.percentile(_energy, 80)

            # One trajectory log for this entire example (all attempts)
            traj_log = GateTrajectoryLog(gate_name="leakage_gate", example_id=filename)

            s0 = compute_speech_gate_score(y_np, (x_hat / norm_factor).cpu().numpy(), speech_mask)
            gate_total += 1
            traj_log.log_final(s0, attempt_idx=0)

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
                    traj_log.log_final(s_i, attempt_idx=i)

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

            # In per-step mode the restart loop owns tries; sync so CSV is consistent.
            # best_try_idx is used for the CSV (restart attempt index).
            # finalize() takes an index into attempt_scores, which has exactly one entry
            # (s0 at index 0) in per-step mode since the K-tries loop never runs.
            if args.gate_tau_path:
                tries_used = 1 + _num_restarts
                best_try_idx = _num_restarts   # accepted attempt index (0-based), for CSV
            _finalize_idx = 0 if args.gate_tau_path else best_try_idx

            # Finalize trajectory log: G = score of the kept attempt
            traj_log.finalize(_finalize_idx)
            # Commit per-step scores from the accepted attempt so trajectory_score = max_k g_k
            if args.gate_compute_tau:
                for _gs in _accepted_g_steps:
                    traj_log.log_step(_gs)
            gate_traj_logs.append(traj_log)

            if best_try_idx != 0:
                gate_replaced += 1
            gate_tries_total += tries_used

            # gate_passed: True  → accepted attempt satisfied tau (clean accept)
            #              False → all restarts exhausted, last attempt forced (may exceed tau)
            if args.gate_tau_path and _final_running_max is not None:
                _gate_passed = int(_final_running_max <= _gate_tau_loaded)
            else:
                _gate_passed = ""

            gate_log.append({
                "filename": filename,
                "s0": round(s0, 6),
                "tries_used": tries_used,
                "accepted_early": int(accepted_early),
                "best_score": round(best_score, 6),
                "best_try_idx": best_try_idx,
                "num_restarts": _num_restarts,
                "first_reject_step": _first_reject_step if _first_reject_step is not None else "",
                "final_running_max": round(_final_running_max, 6) if _final_running_max is not None else "",
                "gate_passed": _gate_passed,
            })

        if args.gate_calibration:
            # Use normalized versions of both signals so the ratio is meaningful
            y_np = y.squeeze().cpu().numpy()                    # shape [T], normalized
            x_hat_np = (x_hat / norm_factor).cpu().numpy()     # shape [T], normalized

            # Energy-based VAD: moving average of y**2 over ~400 samples
            window = 400
            energy = np.convolve(y_np ** 2, np.ones(window) / window, mode='same')
            threshold = np.percentile(energy, 80)
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
            _writer = csv.DictWriter(_f, fieldnames=["filename", "s0", "tries_used", "accepted_early", "best_score", "best_try_idx", "num_restarts", "first_reject_step", "final_running_max", "gate_passed"])
            _writer.writeheader()
            _writer.writerows(gate_log)
        print(f"Gate log saved to {log_path}")
        if args.gate_compute_tau and args.gate_plot:
            from utils.gate_plots import plot_gate_statistics
            plot_gate_statistics(gate_traj_logs, args.enhanced_dir)
        if args.gate_compute_tau:
            from utils.conformal_calib import calibrate_tau_alpha, save_calibration_json
            _calib = calibrate_tau_alpha(gate_traj_logs, args.gate_alpha)
            _tau_out = join(args.enhanced_dir, "_calib_tau.json")
            save_calibration_json(_tau_out, _calib, gate_name="leakage_gate",
                                  extra_meta={"gate_alpha": args.gate_alpha})
            print(f"\n--- Conformal Calibration (G = max_k g_k) ---")
            print(f"alpha = {args.gate_alpha}  →  tau = {_calib['tau']:.4f}")
            print(f"n = {_calib['n']}  G_mean = {_calib['G_mean']:.4f}  G_std = {_calib['G_std']:.4f}")
            print(f"Saved to {_tau_out}")

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
