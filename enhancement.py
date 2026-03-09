import glob
import os
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


def _get_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _run_legacy_gate_restart_sampling(
    build_sampler,
    gate_tau,
    gate_cache,
    gate_seed,
    gate_max_restarts,
    gates,
    gate_combine,
    k_start,
    gate_debug_file,
    filename,
    gate_log_count,
    gate_compute_tau,
    gate_log_every,
    save_step_wavs,
    do_save_step,
    model,
    T_orig,
    norm_factor,
    target_sr,
    enhanced_dir,
    dnsmos_available,
):
    """Legacy CP / per-step gate + restart path.

    Runs the diffusion sampler with a per-step running-max gate that aborts and
    restarts when the gate score exceeds tau.  At most gate_max_restarts + 1
    attempts are made; the final attempt is always forced to completion.

    Returns
    -------
    sample              : accepted diffusion sample tensor
    accepted_g_steps    : per-step gate scores for the accepted attempt
    num_restarts        : number of attempts that were aborted before acceptance
    try0_running_max    : running_max from attempt 0 (set even if attempt 0 aborted)
    try0_rng_state      : RNG snapshot taken just before attempt 0 (for DNSMOS replay)
    first_reject_step   : diffusion step index of the first SamplingAborted (or None)
    final_running_max   : running_max of the accepted attempt
    """
    from utils.speech_gate import (SamplingAborted,
                                   compute_gate_scores_per_step, combine_gate_scores)

    _num_restarts      = 0
    _try0_running_max  = None        # running_max of attempt 0 (for delta_G)
    _try0_rng_state    = None        # exact RNG snapshot before attempt 0
    _first_reject_step = None
    _final_running_max = -float("inf")
    _accepted_g_steps  = []

    for _attempt in range(gate_max_restarts + 1):
        _is_last        = (_attempt == gate_max_restarts)
        _running_max    = [-float("inf")]  # list for mutation inside closure
        _attempt_g_steps = []              # per-step buffer; discarded on abort

        def _step_cb(_si, _rm=_running_max, _tau=gate_tau, _il=_is_last,
                     _cache=gate_cache, _buf=_attempt_g_steps):
            if _si["step_idx"] < k_start:   # late-start: skip early steps
                return
            g_k = combine_gate_scores(
                compute_gate_scores_per_step(_si, _cache, gates),
                gate_combine,
            )
            if gate_debug_file and filename == gate_debug_file:
                print(f"[gate_debug] attempt={_attempt} step={_si['step_idx']:3d} "
                      f"g_k={g_k:.4f} rm={_rm[0]:.4f} tau={_tau:.4f}")
            _rm[0] = max(_rm[0], g_k)
            if gate_compute_tau and _si["step_idx"] % gate_log_every == 0:
                _buf.append(g_k)
            # Per-step WAV saving for test mode (first 2 utterances, target steps)
            if save_step_wavs and gate_log_count < 2 and _si["step_idx"] in (0, 7, 15, 22, 29):
                k = _si["step_idx"]
                _wav_k = (model.to_audio(_si["xt_mean"].squeeze(), T_orig) * norm_factor).cpu().numpy()
                _stem = os.path.splitext(os.path.basename(filename))[0]
                _step_dir = join(enhanced_dir, f"test_step_{k:03d}")
                makedirs(_step_dir, exist_ok=True)
                _wav_path = join(_step_dir, f"{_stem}_step{k:03d}_g{g_k:.4f}.wav")
                write(_wav_path, _wav_k, target_sr)
                print(f"[SAVE_STEP_WAV_TEST] idx={gate_log_count} step={k} "
                      f"g={g_k:.4f} path={_wav_path}")
            do_save_step(_si)
            if not _il and _rm[0] > _tau:
                raise SamplingAborted(_si["step_idx"], _rm[0])

        _set_seeds(gate_seed + _attempt)
        if _attempt == 0 and dnsmos_available:
            _try0_rng_state = _get_rng_state()  # snapshot exact state for try0 replay
        try:
            sample, _ = build_sampler(step_callback=_step_cb)()
        except SamplingAborted as _exc:
            _num_restarts += 1
            if _first_reject_step is None:
                _first_reject_step = _exc.step_idx
            if _attempt == 0:
                _try0_running_max = _exc.running_max
            continue
        if _attempt == 0:
            _try0_running_max = _running_max[0]  # attempt 0 completed cleanly
        _accepted_g_steps  = _attempt_g_steps    # commit accepted attempt's buffer
        _final_running_max = _running_max[0]
        break

    return (sample, _accepted_g_steps, _num_restarts, _try0_running_max,
            _try0_rng_state, _first_reject_step, _final_running_max)


def _run_calibration_sampling(
    build_sampler,
    gate_cache,
    gates,
    gate_combine,
    k_start,
    gate_log_every,
    do_save_step,
    debug_level,
    gate_traj_log_count,
    enhanced_dir,
    filename,
    model,
    T_orig,
    norm_factor,
    target_sr,
):
    """Calibration-only / plain sampling path (no gate decisions).

    Collects per-step gate scores for conformal tau calibration without ever
    aborting or restarting.  Used when --gate_compute_tau is set but
    --gate_tau_path is not.

    Returns
    -------
    sample           : diffusion sample tensor (single pass, no rejection)
    accepted_g_steps : per-step gate scores collected during this pass
    """
    from utils.speech_gate import compute_gate_scores_per_step, combine_gate_scores

    _attempt_g_steps = []

    def _step_cb_cal(_si, _cache=gate_cache, _buf=_attempt_g_steps):
        if _si["step_idx"] < k_start:   # late-start: skip early steps
            return
        if _si["step_idx"] % gate_log_every == 0:
            g_k = combine_gate_scores(
                compute_gate_scores_per_step(_si, _cache, gates),
                gate_combine,
            )
            _buf.append(g_k)
            do_save_step(_si)
            # Per-step WAV saving and debug prints (debug_level > 0 only)
            if debug_level > 0 and _si["step_idx"] in (0, 7, 15, 22, 29):
                _intended = join(enhanced_dir, f"step_{_si['step_idx']:03d}", filename)
                print(f"[DEBUG_GATE_SAVE] step={_si['step_idx']:3d}  g_k={g_k:.6f}  "
                      f"intended_path={_intended}")
                # Save WAV for first 2 utterances only
                if gate_traj_log_count < 2:
                    k = _si["step_idx"]
                    _wav_k = model.to_audio(_si["xt_mean"].squeeze(), T_orig)
                    _wav_k = (_wav_k * norm_factor).cpu().numpy()
                    # Sanity check
                    _peak   = float(np.max(np.abs(_wav_k)))
                    _rms    = float(np.sqrt(np.mean(_wav_k ** 2)))
                    _n      = len(_wav_k)
                    _finite = bool(np.isfinite(_wav_k).all())
                    print(f"[STEP_WAV_STATS] idx={gate_traj_log_count} step={k} "
                          f"n={_n} peak={_peak:.4f} rms={_rms:.4f} finite={_finite}")
                    if not _finite or _peak == 0 or _rms < 1e-6:
                        print(f"[STEP_WAV_STATS] WARNING: idx={gate_traj_log_count} "
                              f"step={k} — audio is {'non-finite' if not _finite else 'silent/zero'}")
                    _stem     = os.path.splitext(os.path.basename(filename))[0]
                    _step_dir = join(enhanced_dir, f"step_{k:03d}")
                    makedirs(_step_dir, exist_ok=True)
                    _wav_path = join(_step_dir, f"{_stem}_step{k:03d}_g{g_k:.4f}.wav")
                    write(_wav_path, _wav_k, target_sr)
                    print(f"[SAVE_STEP_WAV] idx={gate_traj_log_count} step={k} "
                          f"g={g_k:.4f} path={_wav_path}")

    sample, _ = build_sampler(step_callback=_step_cb_cal)()
    return sample, _attempt_g_steps


def _run_adaptive_k_sampling(
    build_sampler,
    adaptive_tau,
    adaptive_kmax,
    base_seed,
    model,
    T_orig,
    norm_factor,
    y_np,
):
    """Adaptive-K inference policy: binary escalation based on baseline difficulty.

    Score metric: wiener_residual (reference-free, lower = better).

    Algorithm
    ---------
    1. Generate baseline sample x0 (try 0).
    2. Compute d0 = wiener_residual(x0).
    3. If d0 <= adaptive_tau: return x0 (no escalation, K=1).
    4. Else: generate tries 1..adaptive_kmax-1, score each,
             return the try with the lowest score (including try 0).

    Parameters
    ----------
    build_sampler   : callable() → sampler (as built per-utterance in main loop)
    adaptive_tau    : float — difficulty threshold; d0 > tau triggers escalation
    adaptive_kmax   : int   — max total number of tries when escalating
    base_seed       : int or None — per-utterance RNG base; try j uses base_seed+j.
                      If None, seeding is skipped (stochastic / non-reproducible).
                      Caller computes: gate_seed + 1000 * utterance_idx.
                      This is independent of the legacy restart/try0_rng_state logic.
    model           : ScoreModel instance
    T_orig          : int   — original waveform length (for model.to_audio)
    norm_factor     : float — amplitude normalisation factor used pre-inference
    y_np            : np.ndarray [T] — normalised noisy input (for score computation)

    Returns
    -------
    x_hat        : best enhanced waveform tensor [T], scaled by norm_factor
    chosen_k     : 1 if not escalated, adaptive_kmax if escalated
    chosen_try   : index (0-based) of the selected try
    d0           : baseline difficulty score (wiener_residual of try 0)
    best_score   : wiener_residual score of the selected try
    escalated    : True if extra tries were generated
    """
    from utils.speech_gate import _wiener_residual_score

    def _score(x_hat_tensor):
        """Wiener residual score for adaptive-K waveform selection (lower = better)."""
        return _wiener_residual_score(y_np, (x_hat_tensor / norm_factor).cpu().numpy())

    # --- Try 0: baseline ---
    if base_seed is not None:
        _set_seeds(base_seed + 0)
    sample_0, _ = build_sampler()()
    x_hat_0 = model.to_audio(sample_0.squeeze(), T_orig) * norm_factor
    d0 = _score(x_hat_0)

    if d0 <= adaptive_tau:
        # Baseline is good enough — no escalation
        return x_hat_0, 1, 0, d0, d0, False

    # --- Escalate: tries 1..adaptive_kmax-1 ---
    scores = [d0]
    x_hats = [x_hat_0]
    for k in range(1, adaptive_kmax):
        if base_seed is not None:
            _set_seeds(base_seed + k)
        s_k, _ = build_sampler()()
        x_k = model.to_audio(s_k.squeeze(), T_orig) * norm_factor
        scores.append(_score(x_k))
        x_hats.append(x_k)

    chosen_try = int(np.argmin(scores))
    best_score = float(scores[chosen_try])
    return x_hats[chosen_try], adaptive_kmax, chosen_try, d0, best_score, True


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
    parser.add_argument("--gate_seed", type=int, default=0, help="Base random seed for reproducible gated sampling")
    parser.add_argument("--gate_tau_path", type=str, default=None, help="Path to _calib_tau.json; enables per-step running-max rejection + restart")
    parser.add_argument("--gate_max_restarts", type=int, default=10, help="Max restarts per utterance when per-step gate triggers (used with --gate_tau_path)")
    parser.add_argument("--gate_compute_tau", action="store_true", help="Log per-step scores so G = max_k g_k; enables conformal tau calibration from gate_traj_logs")
    parser.add_argument("--gate_log_every", type=int, default=1, help="Subsample per-step score logging: store every N-th step (default: 1 = every step)")
    parser.add_argument("--gate_plot", action="store_true", help="Save gate diagnostic plots after inference (requires --gate_compute_tau)")
    parser.add_argument("--gate_alpha", type=float, default=0.1, help="Miscoverage level for conformal calibration; tau is saved to _calib_tau.json when --gate_compute_tau is set")
    parser.add_argument("--gates", nargs="+", default=["leakage"], help='Gate names to use (default: ["leakage"]); supported: "leakage", "wiener_residual", "stft_leakage", "nisqa" (post-hoc only)')
    parser.add_argument("--gate_combine", choices=["max", "mean"], default="max", help='How to combine scores from multiple gates (default: "max")')
    parser.add_argument("--gate_start_frac", type=float, default=0.0, help="Skip per-step gate scoring for the first gate_start_frac fraction of diffusion steps (0.0 = gate from step 0)")
    parser.add_argument("--gate_debug_file", type=str, default="", help="If non-empty, print per-step gate debug info only for this filename")
    parser.add_argument("--file_list", type=str, default=None, help="Optional path to a text file listing filenames (relative to --test_dir) to process; one per line. If omitted, all WAV/FLAC in test_dir are processed.")
    parser.add_argument("--debug_level", type=int, default=0, help="0=off, 1=enable debug prints and per-step WAV saving")
    parser.add_argument("--save_step_wavs", action="store_true", help="Save per-step WAV snapshots at steps {0,7,15,22,29} for first 2 test utterances (test mode only)")
    parser.add_argument("--save_steps", type=str, default="0,7,15,22,29", help="Comma-separated diffusion step indices to snapshot (used with --save_steps_all or --save_steps_limit)")
    parser.add_argument("--save_steps_all", action="store_true", help="Save step snapshots for all utterances (requires OUVESDE + PC sampler)")
    parser.add_argument("--save_steps_dir_prefix", type=str, default="step", help="Directory prefix for step snapshots; outputs to {enhanced_dir}/{prefix}_{k:03d}/")
    parser.add_argument("--save_steps_limit", type=int, default=0, help="If >0, save step snapshots only for first N utterances (0 = no limit; use with --save_steps_all)")
    # --- Adaptive-K policy args ---
    parser.add_argument("--policy", choices=["legacy", "adaptive_k"], default="legacy",
                        help="Inference policy: 'legacy' = existing per-step gate/restart (default); "
                             "'adaptive_k' = reference-free difficulty-based K escalation")
    parser.add_argument("--adaptive_score", choices=["wiener_residual"], default="wiener_residual",
                        help="Post-hoc score used for adaptive-K difficulty and selection (default: wiener_residual)")
    parser.add_argument("--adaptive_tau", type=float, default=None,
                        help="Difficulty threshold for adaptive_k policy: if d0 > tau, escalate to K_max tries. "
                             "Required when --policy adaptive_k.")
    parser.add_argument("--adaptive_kmax", type=int, default=10,
                        help="Max number of tries for adaptive_k escalation (default: 10)")
    args = parser.parse_args()

    if args.policy == "adaptive_k" and args.adaptive_tau is None:
        raise ValueError("--adaptive_tau is required when --policy adaptive_k")

    # --- Adaptive-K: warn once about legacy-only args that are active but will be ignored ---
    if args.policy == "adaptive_k":
        _legacy_active = []
        if args.gate_tau_path is not None:
            _legacy_active.append("gate_tau_path")
        if args.gate_compute_tau:
            _legacy_active.append("gate_compute_tau")
        if args.gate_calibration:
            _legacy_active.append("gate_calibration")
        if args.gates != ["leakage"]:
            _legacy_active.append("gates")
        if args.gate_combine != "max":
            _legacy_active.append("gate_combine")
        if args.gate_max_restarts != 10:
            _legacy_active.append("gate_max_restarts")
        if args.save_steps_all:
            _legacy_active.append("save_steps_all")
        if args.save_steps_limit > 0:
            _legacy_active.append("save_steps_limit")
        if args.gate_plot:
            _legacy_active.append("gate_plot")
        if args.gate_debug_file:
            _legacy_active.append("gate_debug_file")
        if _legacy_active:
            print(f"[adaptive_k] Ignoring legacy gating args: {', '.join(_legacy_active)}")

    # Research output directories (rooted under enhanced_dir)
    _rd_calib = join(args.enhanced_dir, "calib")
    _rd_test  = join(args.enhanced_dir, "test")
    _rd_plots = join(args.enhanced_dir, "plots")
    # Only create calib/ in calibration mode so test runs stay clean
    _active_dirs = [_rd_test, _rd_plots]
    if args.gate_compute_tau:
        _active_dirs.append(_rd_calib)
    for _d in _active_dirs:
        makedirs(_d, exist_ok=True)

    # Load score model
    model = ScoreModel.load_from_checkpoint(args.ckpt, map_location=args.device)
    model.t_eps = args.t_eps
    model.eval()

    # Generic step-snapshot feature (--save_steps_all / --save_steps_limit)
    _save_steps_set = {int(s.strip()) for s in args.save_steps.split(',') if s.strip()}
    _save_steps_enabled = args.save_steps_all or args.save_steps_limit > 0
    if _save_steps_enabled:
        _step_cb_supported = (
            model.sde.__class__.__name__ == 'OUVESDE' and args.sampler_type == 'pc'
        )
        if not _step_cb_supported:
            print("Warning: --save_steps_all/--save_steps_limit requires OUVESDE + PC sampler; "
                  "step saving will be skipped.")
            _save_steps_enabled = False
    _save_steps_utt_idx = [0]  # mutable counter: utterances processed so far

    # args.N = number of reverse diffusion steps (default 30), NOT dataset size.
    # k_start is clamped to N-1 so at least the final step is always gated.
    _k_start = min(int(args.gate_start_frac * args.N), args.N - 1)
    if _k_start > 0:
        print(f"Late-start gating: first {_k_start}/{args.N} steps skipped "
              f"(gate_start_frac={args.gate_start_frac})")

    # Get list of noisy files
    if args.file_list is not None:
        with open(args.file_list) as _fl:
            _entries = [s for line in _fl if (s := line.strip()) and not s.startswith('#')]
        noisy_files = []
        for _rel in _entries:
            _full = join(args.test_dir, _rel)
            if not os.path.isfile(_full):
                print(f"Warning: file_list entry not found, skipping: {_full}")
            else:
                noisy_files.append(_full)
        if not noisy_files:
            print("Error: no valid files found from --file_list. Exiting.")
            raise SystemExit(1)
        print(f"file_list mode: {len(noisy_files)} files to process (from {args.file_list})")
    else:
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

    if args.gate_compute_tau or args.gate_tau_path:
        from utils.speech_gate import (compute_speech_gate_score, GateTrajectoryLog,
                                       compute_posthoc_gate_score)
        gate_log = []
        gate_traj_logs = []  # List[GateTrajectoryLog], one per example; used by gate_compute_tau

    if args.gate_tau_path or args.gate_compute_tau:
        from utils.speech_gate import (SamplingAborted,
                                       compute_gate_scores_per_step, combine_gate_scores)

    if args.gate_tau_path:
        from utils.conformal_calib import load_calibration_json
        _calib = load_calibration_json(args.gate_tau_path)
        _gate_tau_loaded = float(_calib["tau"])
        print(f"Per-step gate: tau={_gate_tau_loaded:.4f}  "
              f"alpha={_calib.get('alpha', '?')}  "
              f"(from {args.gate_tau_path})")

    # DNSMOS availability (test mode only; graceful no-op if speechmos missing)
    _dnsmos_available = False
    if args.gate_tau_path:
        from utils.dnsmos_helper import is_available as _dnsmos_is_available, compute_dnsmos
        _dnsmos_available = _dnsmos_is_available()
        if not _dnsmos_available:
            print("Warning: speechmos not found; DNSMOS columns will be blank. "
                  "Install with: pip install speechmos")

    # NISQA availability (post-hoc only; graceful no-op if nisqa missing)
    if "nisqa" in args.gates and (args.gate_compute_tau or args.gate_tau_path):
        from utils.nisqa_helper import is_available as _nisqa_is_available
        if not _nisqa_is_available():
            print("Warning: nisqa not found; 'nisqa' gate will score 0.0. "
                  "Install with: pip install nisqa and set NISQA_CKPT_PATH.")

    if args.debug_level > 0:
        print(f"[DEBUG_GATE_SAVE] enhanced_dir = {args.enhanced_dir}")
        print(f"[DEBUG_GATE_SAVE] cwd          = {os.getcwd()}")
        print(f"[DEBUG_GATE_SAVE] gate_compute_tau = {args.gate_compute_tau}")
        print(f"[DEBUG_GATE_SAVE] WAV save condition (not gate_compute_tau) = {not args.gate_compute_tau}")

    # Per-utterance log for adaptive_k mode (None in legacy mode so accidental use raises early)
    _ak_log = [] if args.policy == "adaptive_k" else None

    # Enhance files
    for _utt_idx, noisy_file in enumerate(tqdm(noisy_files)):
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

        # Generic step-snapshot callback (independent of gating)
        def _do_save_step(_si):
            if not _save_steps_enabled:
                return
            _k = _si["step_idx"]
            if _k not in _save_steps_set:
                return
            if not args.save_steps_all:
                if args.save_steps_limit > 0 and _save_steps_utt_idx[0] >= args.save_steps_limit:
                    return
            _stem = os.path.splitext(os.path.basename(filename))[0]
            _wav_k = (model.to_audio(_si["xt_mean"].squeeze(), T_orig) * norm_factor).cpu().numpy()
            _step_dir = join(args.enhanced_dir, f"{args.save_steps_dir_prefix}_{_k:03d}")
            makedirs(_step_dir, exist_ok=True)
            _wav_path = join(_step_dir, f"{_stem}_step{_k:03d}.wav")
            write(_wav_path, _wav_k, target_sr)

        # -----------------------------------------------------------------------
        # Dispatch: three mutually exclusive inference paths
        #   1. adaptive_k  — reference-free difficulty-based K escalation (this block)
        #   2. legacy CP   — per-step gate + restart (gate_tau_path / gate_compute_tau)
        #   3. plain       — single-pass diffusion, no gating
        # Legacy gating args (gate_tau_path, gate_compute_tau, gates, gate_combine, …)
        # are intentionally ignored in adaptive_k mode; a startup warning lists them.
        # -----------------------------------------------------------------------
        if args.policy == "adaptive_k":
            y_np = y.squeeze().cpu().numpy()
            # Per-utterance base seed: deterministic from gate_seed + utterance index.
            # Each try j within this utterance uses base_seed + j, keeping tries independent.
            # If gate_seed is None, stochastic (non-reproducible) mode is used.
            _ak_base_seed = (args.gate_seed + 1000 * _utt_idx) if args.gate_seed is not None else None
            (x_hat,
             _ak_chosen_k,
             _ak_chosen_try,
             _ak_d0,
             _ak_best_score,
             _ak_escalated) = _run_adaptive_k_sampling(
                build_sampler=_build_sampler,
                adaptive_tau=args.adaptive_tau,
                adaptive_kmax=args.adaptive_kmax,
                base_seed=_ak_base_seed,
                model=model,
                T_orig=T_orig,
                norm_factor=norm_factor,
                y_np=y_np,
            )
            print(f"[adaptive_k] file={filename}  utt_idx={_utt_idx}  "
                  f"base_seed={_ak_base_seed}  d0={_ak_d0:.4f}  "
                  f"escalated={_ak_escalated}  chosen_k={_ak_chosen_k}  "
                  f"chosen_try={_ak_chosen_try}  best_score={_ak_best_score:.4f}")
            _ak_log.append({
                "utterance_idx":  _utt_idx,
                "filename":       filename,
                "base_seed_used": _ak_base_seed,
                "adaptive_score": args.adaptive_score,
                "adaptive_tau":   args.adaptive_tau,
                "adaptive_kmax":  args.adaptive_kmax,
                "d0":             round(float(_ak_d0), 6),
                "escalated":      int(_ak_escalated),
                "chosen_k":       _ak_chosen_k,
                "chosen_try":     _ak_chosen_try,
                "best_score":     round(float(_ak_best_score), 6),
            })
            makedirs(dirname(join(args.enhanced_dir, filename)), exist_ok=True)
            write(join(args.enhanced_dir, filename), x_hat.cpu().numpy(), target_sr)
            _save_steps_utt_idx[0] += 1
            continue

        # Build per-step cache once per utterance whenever per-step scoring is needed.
        if args.gate_tau_path or args.gate_compute_tau:
            _yf_power = (Y[0].abs() ** 2).sum(dim=(0, 1)).detach().cpu().numpy()
            _gate_cache = {
                "speech_mask_frames": _yf_power > np.percentile(_yf_power, 80),
                "eps": 1e-8,
            }

        # --- Path 2 / Path 3: legacy CP gate or plain sampling ---
        if args.gate_tau_path:
            # Path 2a: legacy CP — per-step running-max rejection + restart
            (sample,
             _accepted_g_steps,
             _num_restarts,
             _try0_running_max,
             _try0_rng_state,
             _first_reject_step,
             _final_running_max) = _run_legacy_gate_restart_sampling(
                build_sampler=_build_sampler,
                gate_tau=_gate_tau_loaded,
                gate_cache=_gate_cache,
                gate_seed=args.gate_seed,
                gate_max_restarts=args.gate_max_restarts,
                gates=args.gates,
                gate_combine=args.gate_combine,
                k_start=_k_start,
                gate_debug_file=args.gate_debug_file,
                filename=filename,
                gate_log_count=len(gate_log),
                gate_compute_tau=args.gate_compute_tau,
                gate_log_every=args.gate_log_every,
                save_step_wavs=args.save_step_wavs,
                do_save_step=_do_save_step,
                model=model,
                T_orig=T_orig,
                norm_factor=norm_factor,
                target_sr=target_sr,
                enhanced_dir=args.enhanced_dir,
                dnsmos_available=_dnsmos_available,
            )
        else:
            if args.gate_compute_tau:
                # Path 2b: legacy calibration-only — collect per-step scores, no gate decisions.
                sample, _accepted_g_steps = _run_calibration_sampling(
                    build_sampler=_build_sampler,
                    gate_cache=_gate_cache,
                    gates=args.gates,
                    gate_combine=args.gate_combine,
                    k_start=_k_start,
                    gate_log_every=args.gate_log_every,
                    do_save_step=_do_save_step,
                    debug_level=args.debug_level,
                    gate_traj_log_count=len(gate_traj_logs),
                    enhanced_dir=args.enhanced_dir,
                    filename=filename,
                    model=model,
                    T_orig=T_orig,
                    norm_factor=norm_factor,
                    target_sr=target_sr,
                )
            else:
                # Path 3: plain single-pass sampling (no gating)
                _cb = _do_save_step if _save_steps_enabled else None
                sample, _ = _build_sampler(step_callback=_cb)()
                _accepted_g_steps = []
            _num_restarts      = 0
            _try0_running_max  = None
            _first_reject_step = None
            _final_running_max = None
        
        # Backward transform in time domain
        x_hat = model.to_audio(sample.squeeze(), T_orig)

        # Renormalize
        x_hat = x_hat * norm_factor

        if args.gate_compute_tau or args.gate_tau_path:
            y_np = y.squeeze().cpu().numpy()
            _x_hat_norm = (x_hat / norm_factor).cpu().numpy()

            s0 = compute_posthoc_gate_score(
                args.gates, y_np, _x_hat_norm,
                gate_combine=args.gate_combine, sr=target_sr,
            )

            if args.gate_compute_tau:
                # Trajectory log: records per-step scores for conformal calibration
                traj_log = GateTrajectoryLog(gate_name="leakage_gate", example_id=filename)
                traj_log.log_final(s0, attempt_idx=0)
                traj_log.finalize(0)
                for _gs in _accepted_g_steps:
                    traj_log.log_step(_gs)
                gate_traj_logs.append(traj_log)

            if args.gate_tau_path:
                # gate_passed: True  → accepted attempt satisfied tau (clean accept)
                #              False → all restarts exhausted, last attempt forced (may exceed tau)
                _gate_passed = int(_final_running_max <= _gate_tau_loaded) if _final_running_max is not None else ""

                _delta_G = (
                    round(_try0_running_max - _final_running_max, 6)
                    if (_try0_running_max is not None and _final_running_max is not None)
                    else ""
                )
                # DNSMOS evaluation (one inference pass per file)
                _dnsmos_best = _dnsmos_try0 = _delta_dnsmos = ""
                if _dnsmos_available:
                    _best_val = compute_dnsmos(x_hat.cpu().numpy(), target_sr)
                    if _num_restarts == 0:
                        _try0_val = _best_val      # no restarts: try0 == best
                    else:
                        # attempt 0 was aborted; replay exact RNG state → identical trajectory
                        _set_rng_state(_try0_rng_state)
                        _s_try0, _ = _build_sampler()()   # no callback → runs to completion
                        _xh_try0 = model.to_audio(_s_try0.squeeze(), T_orig) * norm_factor
                        _try0_val = compute_dnsmos(_xh_try0.cpu().numpy(), target_sr)
                    if _best_val is not None:
                        _dnsmos_best = round(float(_best_val), 4)
                    if _try0_val is not None:
                        _dnsmos_try0 = round(float(_try0_val), 4)
                    if _dnsmos_best != "" and _dnsmos_try0 != "":
                        _delta_dnsmos = round(float(_best_val) - float(_try0_val), 4)

                gate_log.append({
                    "filename": filename,
                    "s0": round(s0, 6),
                    "num_restarts": _num_restarts,
                    "first_reject_step": _first_reject_step if _first_reject_step is not None else "",
                    "final_running_max": round(_final_running_max, 6) if _final_running_max is not None else "",
                    "gate_passed": _gate_passed,
                    "G_try0": round(_try0_running_max, 6) if _try0_running_max is not None else "",
                    "delta_G": _delta_G,
                    "DNSMOS_try0": _dnsmos_try0,
                    "DNSMOS_best": _dnsmos_best,
                    "delta_DNSMOS": _delta_dnsmos,
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

        # Write enhanced wav file (skip in calibration mode — we only need gate scores)
        if not args.gate_compute_tau:
            if args.debug_level > 0:
                print(f"[DEBUG_GATE_SAVE] triggered WAV save → {join(args.enhanced_dir, filename)}")
            makedirs(dirname(join(args.enhanced_dir, filename)), exist_ok=True)
            write(join(args.enhanced_dir, filename), x_hat.cpu().numpy(), target_sr)

        _save_steps_utt_idx[0] += 1

    if args.gate_compute_tau:
        if args.gate_plot:
            from utils.gate_plots import plot_gate_statistics
            plot_gate_statistics(gate_traj_logs, _rd_plots, enhanced_dir=args.enhanced_dir)
        from utils.conformal_calib import calibrate_tau_alpha, save_calibration_json
        _calib = calibrate_tau_alpha(gate_traj_logs, args.gate_alpha)
        _tau_out = join(_rd_calib, "_calib_tau.json")
        save_calibration_json(_tau_out, _calib, gate_name="leakage_gate",
                              extra_meta={"gate_alpha": args.gate_alpha,
                                          "gate_start_frac": args.gate_start_frac,
                                          "gates": args.gates})
        print(f"\n--- Conformal Calibration (G = max_k g_k) ---")
        print(f"alpha = {args.gate_alpha}  →  tau = {_calib['tau']:.4f}")
        print(f"n = {_calib['n']}  G_mean = {_calib['G_mean']:.4f}  G_std = {_calib['G_std']:.4f}")
        print(f"Saved to {_tau_out}")

    if args.gate_tau_path:
        log_path = join(_rd_test, "_gate_log.csv")
        with open(log_path, "w", newline="") as _f:
            _writer = csv.DictWriter(_f, fieldnames=[
                "filename", "s0", "num_restarts", "first_reject_step",
                "final_running_max", "gate_passed", "G_try0", "delta_G",
                "DNSMOS_try0", "DNSMOS_best", "delta_DNSMOS",
            ])
            _writer.writeheader()
            _writer.writerows(gate_log)
        print(f"Gate log saved to {log_path}")
        # Restart improvement diagnostics
        from utils.gate_plots import plot_delta_G
        _rows_with_restarts = [r for r in gate_log if r["num_restarts"] > 0]
        _n_restarted = len(_rows_with_restarts)
        _frac_restarted = _n_restarted / len(gate_log) if gate_log else 0.0
        _delta_vals = [r["delta_G"] for r in _rows_with_restarts if r["delta_G"] != ""]
        _mean_dG  = float(np.mean(_delta_vals))  if _delta_vals else float("nan")
        _med_dG   = float(np.median(_delta_vals)) if _delta_vals else float("nan")
        _summary_path = join(_rd_test, "restart_summary.txt")
        with open(_summary_path, "w") as _sf:
            _sf.write("=== Restart Improvement Summary ===\n")
            _sf.write(f"Total samples:          {len(gate_log)}\n")
            _sf.write(f"Samples with restarts:  {_n_restarted}  ({_frac_restarted:.1%})\n")
            _sf.write(f"delta_G = G_try0 - G_final  (positive = improvement)\n")
            _sf.write(f"  mean delta_G:   {_mean_dG:.4f}\n")
            _sf.write(f"  median delta_G: {_med_dG:.4f}\n")
            _sf.write(f"  n with delta_G: {len(_delta_vals)}\n")
        print(f"Restart summary saved to {_summary_path}")
        if _delta_vals:
            plot_delta_G(_delta_vals, _rd_plots)
        # DNSMOS correlation analysis (only when restarts occurred and DNSMOS is available)
        if _dnsmos_available and _rows_with_restarts:
            _corr_rows = [r for r in _rows_with_restarts
                          if r["delta_DNSMOS"] != "" and r["delta_G"] != ""]
            if _corr_rows:
                from scipy import stats
                from utils.gate_plots import plot_deltaG_vs_deltaDNSMOS
                _MIN_DG = 0.01   # filter numerical-noise band; only rows with |delta_G| > this

                def _corr_block(rows, label):
                    """Compute and return (dG_arr, dDNS_arr, pearson_r, spearman_r) or None."""
                    _filt = [r for r in rows if abs(float(r["delta_G"])) > _MIN_DG]
                    if len(_filt) < 2:
                        return None
                    _dG   = np.array([r["delta_G"]     for r in _filt], dtype=float)
                    _dDNS = np.array([r["delta_DNSMOS"] for r in _filt], dtype=float)
                    _pr, _pp = stats.pearsonr(_dG, _dDNS)
                    _sr, _sp = stats.spearmanr(_dG, _dDNS)
                    return _dG, _dDNS, _pr, _pp, _sr, _sp, label

                _subsets = [
                    ("all restarted",      _corr_rows),
                    ("gate_passed == 1",   [r for r in _corr_rows if r["gate_passed"] == 1]),
                    ("gate_passed == 0",   [r for r in _corr_rows if r["gate_passed"] == 0]),
                ]
                _corr_path = join(_rd_test, "dnsmos_correlation.txt")
                print(f"\n--- DNSMOS Correlation  (|delta_G| > {_MIN_DG}) ---")
                with open(_corr_path, "w") as _cf:
                    _cf.write("=== DNSMOS Correlation with Gate Improvement ===\n")
                    _cf.write(f"Filter: |delta_G| > {_MIN_DG}\n")
                    _first_result = None
                    for _label, _rows in _subsets:
                        _res = _corr_block(_rows, _label)
                        if _res is None:
                            _cf.write(f"\n--- {_label}: too few samples ---\n")
                            continue
                        _dG, _dDNS, _pr, _pp, _sr, _sp, _lbl = _res
                        _cf.write(f"\n--- {_lbl} (n={len(_dG)}) ---\n")
                        _cf.write(f"  mean delta_G:      {float(np.mean(_dG)):.4f}\n")
                        _cf.write(f"  mean delta_DNSMOS: {float(np.mean(_dDNS)):.4f}\n")
                        _cf.write(f"  Pearson  r={_pr:.4f}  p={_pp:.4f}\n")
                        _cf.write(f"  Spearman r={_sr:.4f}  p={_sp:.4f}\n")
                        print(f"  {_lbl} (n={len(_dG)}): "
                              f"Pearson r={_pr:.4f}  Spearman ρ={_sr:.4f}")
                        if _first_result is None:
                            _first_result = (_dG, _dDNS)
                print(f"  Saved to {_corr_path}")
                if _first_result is not None:
                    plot_deltaG_vs_deltaDNSMOS(_first_result[0], _first_result[1], _rd_plots)

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

    if args.policy == "adaptive_k" and _ak_log:
        _ak_csv_path = join(args.enhanced_dir, "adaptive_k_log.csv")
        _ak_fields = ["utterance_idx", "filename", "base_seed_used",
                      "adaptive_score", "adaptive_tau", "adaptive_kmax",
                      "d0", "escalated", "chosen_k", "chosen_try", "best_score"]
        with open(_ak_csv_path, "w", newline="") as _f:
            _writer = csv.DictWriter(_f, fieldnames=_ak_fields)
            _writer.writeheader()
            _writer.writerows(_ak_log)
        print(f"\nAdaptive-K log saved to {_ak_csv_path}")
        _n_files     = len(_ak_log)
        _n_escalated = sum(r["escalated"] for r in _ak_log)
        _esc_rate    = _n_escalated / _n_files if _n_files else 0.0
        _mean_k      = sum(r["chosen_k"]    for r in _ak_log) / _n_files if _n_files else 0.0
        _mean_d0     = sum(r["d0"]          for r in _ak_log) / _n_files if _n_files else 0.0
        _mean_best   = sum(r["best_score"]  for r in _ak_log) / _n_files if _n_files else 0.0
        print("\n--- Adaptive-K Run Summary ---")
        print(f"n_files         = {_n_files}")
        print(f"n_escalated     = {_n_escalated}  ({_esc_rate:.1%})")
        print(f"mean_chosen_k   = {_mean_k:.2f}")
        print(f"mean_d0         = {_mean_d0:.4f}")
        print(f"mean_best_score = {_mean_best:.4f}")
