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
