import math
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


def compute_speech_gate_score(y, x_hat, speech_mask, eps=1e-8):
    """
    Compute a speech-aware quality gate score for an enhanced waveform.

    Args:
        y:            mixture waveform, 1D numpy array or torch tensor [T]
        x_hat:        enhanced waveform, same shape as y
        speech_mask:  boolean array [T], True = speech region
        eps:          small constant for numerical stability

    Returns:
        score (float): lower is better
    """
    # Convert torch tensors to numpy if needed
    try:
        import torch
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(x_hat, torch.Tensor):
            x_hat = x_hat.detach().cpu().numpy()
        if isinstance(speech_mask, torch.Tensor):
            speech_mask = speech_mask.detach().cpu().numpy().astype(bool)
    except ImportError:
        pass

    y = np.asarray(y, dtype=np.float64)
    x_hat = np.asarray(x_hat, dtype=np.float64)
    speech_mask = np.asarray(speech_mask, dtype=bool)
    non_speech_mask = ~speech_mask

    # Non-speech leakage
    if non_speech_mask.any():
        leakage = np.mean(x_hat[non_speech_mask] ** 2) / (
            np.mean(y[non_speech_mask] ** 2) + eps
        )
    else:
        leakage = 0.0

    # Speech distortion
    if speech_mask.any():
        distortion = np.mean((y[speech_mask] - x_hat[speech_mask]) ** 2) / (
            np.mean(y[speech_mask] ** 2) + eps
        )
    else:
        distortion = 0.0

    score = 0.6 * math.log(leakage + eps) + 0.4 * distortion

    return float(score)


class SamplingAborted(Exception):
    """
    Raised inside a step_callback to abort the current sampling trajectory.
    Caught by the outer restart loop in enhancement.py.
    """
    def __init__(self, step_idx: int, running_max: float):
        self.step_idx = step_idx
        self.running_max = running_max


def gate_step_score(step_info: dict, cache: dict) -> float:
    """
    Per-step gate score hook.  Called at every diffusion step by the PC sampler
    when a step_callback is registered.

    Computes a leakage ratio in the TF domain using xt_mean (the denoised latent
    estimate), aligned with a precomputed frame-level speech mask stored in cache.

    step_info keys:
        step_idx : int           - diffusion step index (0 = first step from noise)
        t        : float         - current diffusion time (decreasing from T to eps)
        xt       : torch.Tensor  - noisy latent state at this step  [B, C, F, T_spec]
        xt_mean  : torch.Tensor  - denoised estimate (noise-free)   [B, C, F, T_spec]

    cache keys:
        speech_mask_frames : np.ndarray [T_spec], bool  (CPU)
        eps                : float (default 1e-8)

    Returns:
        float: non-speech-to-speech frame-power ratio (lower = better).
               Same sign convention as the post-hoc leakage gate.
    """
    xt_mean = step_info["xt_mean"]           # [B, C, F, T_spec]
    mask    = cache["speech_mask_frames"]    # [T_spec] bool, CPU numpy
    eps     = cache.get("eps", 1e-8)

    # Per-frame power: sum over C and F for the first batch item.
    # Stays on device until the single .cpu() call to avoid per-step round-trips.
    frame_power = (xt_mean[0].abs() ** 2).sum(dim=(0, 1))  # [T_spec], same device
    fp = frame_power.detach().cpu().numpy()                  # [T_spec], CPU numpy

    # Align lengths: Y padding and sampler internals may occasionally differ by a frame.
    T = min(len(fp), len(mask))
    fp   = fp[:T]
    mask = mask[:T]

    non_mask = ~mask

    # Edge cases: no speech frames → score 0.0 (don't penalise);
    #             no non-speech frames → numerator is 0 → return 0.0.
    if not mask.any() or not non_mask.any():
        return 0.0

    leakage_num = float(np.mean(fp[non_mask]))
    leakage_den = float(np.mean(fp[mask])) + eps

    return leakage_num / leakage_den


def gate_step_wiener_residual(step_info: dict, cache: dict) -> float:
    """Per-step Wiener-residual gate — reference-free SI-SDR proxy.

    Estimates noise PSD from VAD-off (non-speech) frames via median across time,
    then computes the ratio of residual (noise-like) energy to speech-excess
    energy during speech frames.  Higher score = more noise leaking through =
    worse enhancement.

    step_info keys (used):
        xt_mean : torch.Tensor [B, C, F, T_spec] — denoised latent estimate

    cache keys (used):
        speech_mask_frames : np.ndarray [T_spec], bool — True = speech frame
        eps                : float (default 1e-8)
        wiener_alpha       : float (default 1.0) — noise over-subtraction factor

    Returns:
        float: E_resid / (E_speech + eps) — higher = worse.
               Returns 0.0 on edge cases (too few speech/non-speech frames).
    """
    xt_mean = step_info["xt_mean"]        # [B, C, F, T_spec]
    mask    = cache["speech_mask_frames"] # [T_spec] bool, CPU numpy
    eps     = cache.get("eps", 1e-8)
    alpha   = cache.get("wiener_alpha", 1.0)

    # Power spectrogram for batch item 0, sum over C → [F, T_spec].  Single .cpu() call.
    S = (xt_mean[0].abs() ** 2).sum(dim=0).detach().cpu().numpy()  # [F, T_spec]

    # Align lengths.
    T = min(S.shape[1], len(mask))
    S    = S[:, :T]
    mask = mask[:T]

    non = ~mask
    if non.sum() < 5 or mask.sum() < 10:
        return 0.0

    # Noise PSD estimate from non-speech frames (median over time).
    N_f = np.median(S[:, non], axis=1)   # [F]
    N_f = np.maximum(N_f, eps)

    # Speech-excess and residual energy during speech frames.
    S_speech   = S[:, mask]                                         # [F, N_speech]
    speech_est = np.maximum(S_speech - alpha * N_f[:, None], 0.0)
    resid_est  = np.minimum(S_speech, alpha * N_f[:, None])

    E_speech = float(speech_est.sum())
    E_resid  = float(resid_est.sum())

    score = E_resid / (E_speech + eps)
    return min(score, 1e6)


def gate_step_stft_leakage(step_info: dict, cache: dict) -> float:
    """Per-step non-speech/speech frame-power ratio on the TF spectrogram.

    Sum power over frequency bins to get a per-frame scalar, then return
    mean(non-speech frames) / mean(speech frames).  Identical formula to
    the post-hoc _stft_leakage_score in leakage_best_of_k_eval.py.
    Higher = more energy leaking into silence = worse.

    step_info keys (used):
        xt_mean : torch.Tensor [B, C, F, T_spec] — denoised latent estimate

    cache keys (used):
        speech_mask_frames : np.ndarray [T_spec], bool — True = speech frame
        eps                : float (default 1e-8)

    Returns:
        float: mean_power(non-speech) / mean_power(speech) — higher = worse.
               Returns 0.0 on edge cases (no speech or non-speech frames).
    """
    xt_mean = step_info["xt_mean"]        # [B, C, F, T_spec]
    mask    = cache["speech_mask_frames"] # [T_spec] bool, CPU numpy
    eps     = cache.get("eps", 1e-8)

    # Sum power over channels and frequency → per-frame scalar [T_spec].  Single .cpu() call.
    frame_power = (xt_mean[0].abs() ** 2).sum(dim=(0, 1)).detach().cpu().numpy()  # [T_spec]

    T = min(len(frame_power), len(mask))
    fp   = frame_power[:T]
    mask = mask[:T]

    non = ~mask
    if not mask.any() or not non.any():
        return 0.0

    return float(np.mean(fp[non])) / (float(np.mean(fp[mask])) + eps)


def compute_gate_scores_per_step(step_info: dict, cache: dict, gates: list) -> dict:
    """Return {gate_name: score} for each requested gate at this diffusion step.

    Supported gates:
        "leakage"          — non-speech-to-speech frame-power ratio (reference-free).
        "wiener_residual"  — Wiener-like residual-to-speech-excess energy ratio over
                             speech frames; reference-free SI-SDR proxy.
                             Higher = more noise leaking through = worse.
        "stft_leakage"     — Simple non-speech/speech frame-power ratio on the TF
                             spectrogram (sum over F then mean per region).  Matches
                             the post-hoc _stft_leakage_score exactly.
                             Higher = more leakage = worse.

    Adding a new gate requires registering it in the if/elif chain below.
    """
    scores = {}
    for gate in gates:
        if gate == "leakage":
            scores["leakage"] = gate_step_score(step_info, cache)
        elif gate == "wiener_residual":
            scores["wiener_residual"] = gate_step_wiener_residual(step_info, cache)
        elif gate == "stft_leakage":
            scores["stft_leakage"] = gate_step_stft_leakage(step_info, cache)
    return scores


def combine_gate_scores(scores: dict, method: str = "max") -> float:
    """Reduce a {gate_name: score} dict to a single scalar.

    method: "max" → worst-gate score (conservative); "mean" → average.
    """
    vals = list(scores.values())
    if not vals:
        return 0.0
    if method == "max":
        return float(max(vals))
    if method == "mean":
        return float(sum(vals) / len(vals))
    raise ValueError(f"Unknown gate_combine method: {method!r}")


def compute_gate_score(step_info: dict) -> float:
    """
    Thin wrapper around compute_speech_gate_score for use in both
    post-hoc and future per-step gating.

    step_info keys (required):
        audio        : np.ndarray [T]  - current waveform estimate (normalized)
        y_np         : np.ndarray [T]  - noisy mixture (normalized, same scale)
    step_info keys (optional):
        speech_mask  : np.ndarray [T]  - boolean VAD mask; if absent, all frames
                                         are treated as speech (leakage term = 0)
        t            : float | None    - diffusion time at this step (None = post-hoc)
        step_idx     : int | None      - step index k (None = post-hoc)
    """
    mask = step_info.get("speech_mask", None)
    if mask is None:
        mask = np.ones(len(step_info["audio"]), dtype=bool)
    return compute_speech_gate_score(
        step_info["y_np"],
        step_info["audio"],
        mask,
    )


@dataclass
class GateTrajectoryLog:
    """
    Records gate scores for ONE input example across all sampling attempts.

    Post-hoc mode  : call log_final(score, attempt_idx) for each attempt;
                     after all attempts, set accepted_attempt_idx and G manually
                     (or pass accepted=True on the final kept attempt).
    Per-step mode  : call log_step(score, t) for each diffusion step within
                     an attempt, then log_final(...) at the end of that attempt.

    G is the trajectory-level summary score:
        - for now: G = score of the accepted attempt  (set externally)
        - future:  G = max(g_k) over diffusion steps
    """
    gate_name: str
    example_id: Optional[str] = None          # filename or utterance id
    mode: str = "posthoc"                     # "posthoc" | "perstep"
    K: int = 0                                # number of attempts logged
    t_k: Optional[List[float]] = None        # diffusion times (per-step mode)
    g_k: Optional[List[float]] = None        # gate scores    (per-step mode)
    g_final: Optional[float] = None          # score of the accepted attempt
    G: Optional[float] = None                # trajectory score placeholder
    attempt_scores: List[float] = field(default_factory=list)   # score per attempt
    accepted_attempt_idx: Optional[int] = None                  # which attempt was kept
    g_steps: List[float] = field(default_factory=list)          # per-step scores for accepted attempt

    def log_final(self, score: float, attempt_idx: int = 0, accepted: bool = False) -> None:
        """
        Record the post-hoc score for one sampling attempt.
        Call once per attempt; set accepted=True on the kept attempt,
        or update accepted_attempt_idx externally after all attempts finish.
        """
        self.attempt_scores.append(score)
        self.K = len(self.attempt_scores)
        if accepted:
            self.g_final = score
            self.accepted_attempt_idx = attempt_idx
            self.G = score  # placeholder — later will be max_k g_k

    def finalize(self, accepted_attempt_idx: int) -> None:
        """
        Call once after all attempts are logged to mark the kept sample.
        Sets g_final and G from attempt_scores[accepted_attempt_idx].
        """
        self.accepted_attempt_idx = accepted_attempt_idx
        self.g_final = self.attempt_scores[accepted_attempt_idx]
        self.G = self.g_final  # placeholder — later will be max_k g_k

    @property
    def trajectory_score(self) -> float:
        """
        The trajectory-level summary score used for conformal calibration.

        If per-step scores have been logged via log_step() (g_steps non-empty):
            G = max(g_steps)   — worst-case leakage over the accepted trajectory
        Otherwise falls back to the post-hoc score set by finalize():
            G = attempt_scores[accepted_attempt_idx]

        Raises RuntimeError if neither source is available.
        """
        if self.g_steps:
            return float(max(self.g_steps))
        if self.G is None:
            raise RuntimeError(
                f"trajectory_score accessed on example '{self.example_id}' "
                "before finalize() was called."
            )
        return self.G

    def log_step(self, score: float, t: Optional[float] = None, step_idx: Optional[int] = None) -> None:
        """Accumulate a per-step gate score for the accepted attempt.

        score     : gate score at this diffusion step
        t         : diffusion time (optional metadata)
        step_idx  : step index (optional metadata)
        """
        self.g_steps.append(score)
        # Legacy g_k / t_k fields kept for backward compatibility
        if self.g_k is None:
            self.g_k = []
        self.g_k.append(score)
        if t is not None:
            if self.t_k is None:
                self.t_k = []
            self.t_k.append(float(t))


# ---------------------------------------------------------------------------
# Post-hoc gate score functions
# Exact copies of the helpers in leakage_best_of_k_eval.py — lower = worse.
# ---------------------------------------------------------------------------

_VAD_WINDOW = 400  # samples


def _speech_mask(y: np.ndarray, percentile: float = 80.0) -> np.ndarray:
    """Energy-based VAD mask on the noisy signal."""
    energy = np.convolve(y ** 2, np.ones(_VAD_WINDOW) / _VAD_WINDOW, mode="same")
    return energy > np.percentile(energy, percentile)


def _stft_leakage_score(y: np.ndarray, x_hat: np.ndarray) -> float:
    """Post-hoc STFT frame-power leakage gate.  Lower = better.

    Computes STFT of the enhanced signal, sums power over frequency bins to get
    a per-frame scalar, then returns mean(non-speech) / mean(speech).
    Identical formula to gate_step_stft_leakage above.
    """
    import librosa

    _EPS  = 1e-8
    N_FFT = 512
    HOP   = 128

    T = min(len(y), len(x_hat))
    y     = np.asarray(y[:T],     dtype=np.float32)
    x_hat = np.asarray(x_hat[:T], dtype=np.float32)

    speech_mask_samples = _speech_mask(y)
    win = np.hanning(N_FFT).astype(np.float32)
    S   = np.abs(librosa.stft(x_hat, n_fft=N_FFT, hop_length=HOP,
                               win_length=N_FFT, window=win)) ** 2  # [F, T_frames]

    # Per-frame power: sum over frequency bins
    fp = S.sum(axis=0)  # [T_frames]

    # Convert sample-level mask → frame-level
    n_frames   = fp.shape[0]
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
    """Post-hoc Wiener residual gate.  Lower = better.

    Estimates noise PSD from non-speech frames (median across time), then
    computes the ratio of residual (noise-like) energy to speech-excess energy
    over speech frames.  High values = more noise leaking through = worse.
    """
    import librosa

    _EPS   = 1e-8
    N_FFT  = 512
    HOP    = 128

    T = min(len(y), len(x_hat))
    y     = np.asarray(y[:T],     dtype=np.float32)
    x_hat = np.asarray(x_hat[:T], dtype=np.float32)

    speech_mask_samples = _speech_mask(y)
    win = np.hanning(N_FFT).astype(np.float32)
    S   = np.abs(librosa.stft(x_hat, n_fft=N_FFT, hop_length=HOP,
                               win_length=N_FFT, window=win)) ** 2  # [F, T_frames]

    # Convert sample-level mask → frame-level
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

    # Noise PSD estimate from non-speech frames
    N_f = np.median(S[:, non_mask], axis=1)  # [F]
    N_f = np.maximum(N_f, _EPS)

    S_speech   = S[:, frame_mask]                                   # [F, N_speech]
    speech_est = np.maximum(S_speech - N_f[:, None], 0.0)
    resid_est  = np.minimum(S_speech, N_f[:, None])

    E_speech = float(speech_est.sum())
    E_resid  = float(resid_est.sum())
    return min(E_resid / (E_speech + _EPS), 1e6)


def compute_posthoc_gate_score(
    gates: list,
    y_np: np.ndarray,
    x_hat_np: np.ndarray,
    gate_combine: str = "max",
    sr: int = None,
) -> float:
    """Post-hoc gate score on the final waveform.  Convention: lower = worse.

    gates       : list of gate names (same as --gates in enhancement.py)
    y_np        : noisy input waveform, 1-D numpy float array (normalized)
    x_hat_np    : enhanced waveform, same scale as y_np (normalized)
    gate_combine: "max" or "mean" — how to combine multiple gates
    sr          : sample rate of x_hat_np in Hz (required for "nisqa")

    Gate directions:
        leakage          lower = better (less noise leaked through)
        wiener_residual  lower = better (less residual noise)
        stft_leakage     lower = better (less power in silence frames)
        nisqa            higher = better → negated so convention stays lower = worse
    """
    scores = {}
    for gate in gates:
        if gate == "leakage":
            mask = _speech_mask(y_np)
            scores["leakage"] = compute_speech_gate_score(y_np, x_hat_np, mask)
        elif gate == "wiener_residual":
            scores["wiener_residual"] = _wiener_residual_score(y_np, x_hat_np)
        elif gate == "stft_leakage":
            scores["stft_leakage"] = _stft_leakage_score(y_np, x_hat_np)
        elif gate == "nisqa":
            if sr is None:
                raise ValueError("sr is required for the 'nisqa' gate")
            from .nisqa_helper import compute_nisqa
            raw = compute_nisqa(x_hat_np, sr)
            # Negate: nisqa is higher = better; convention here is lower = worse
            scores["nisqa"] = -float(raw) if raw is not None else 0.0
    return combine_gate_scores(scores, gate_combine)
