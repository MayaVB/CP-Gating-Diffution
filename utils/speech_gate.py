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


# def gate_step_wiener_tf(step_info: dict, cache: dict) -> float:
#     """TF-domain Wiener gate with IMCRA adaptive noise tracking. Lower = better.

#     An upgrade over gate_step_wiener_residual that replaces:
#       - Static per-bin median noise estimate (from VAD frames) →
#         IMCRA recursive noise tracking on PY (frame-by-frame adaptive)
#       - Binary VAD speech mask →
#         per-bin SNR-based speech detection against adaptive noise floor

#     Both PY (noisy reference) and PX (enhanced latent) are in the model's
#     TF domain — no waveform conversion needed.

#     Score = E_residual / (E_speech + eps)
#       E_residual : energy of PX sitting below the adaptive noise floor λ_d
#       E_speech   : energy of PX exceeding λ_d (speech-like excess)

#     Higher score = more noise leaking through = worse enhancement.

#     step_info keys (used):
#         xt_mean : torch.Tensor [B, C, F, T_spec]

#     cache keys (required):
#         y_PY    : np.ndarray [F, T_spec] — model-domain noisy power spectrum
#                   (set by enhancement.py when mid_wiener_score=wiener_tf)

#     cache keys (optional):
#         eps          : float (default 1e-8)
#         wiener_alpha : float (default 1.0) — noise floor over-subtraction factor
#     """
#     xt_mean = step_info["xt_mean"]                                   # [B, C, F, T_spec]
#     eps     = cache.get("eps", 1e-8)
#     alpha   = cache.get("wiener_alpha", 1.0)

#     PX = (xt_mean[0].abs() ** 2).sum(dim=0).detach().cpu().numpy()  # [F, T_spec]

#     if "y_PY" in cache:
#         PY = cache["y_PY"]
#     else:
#         # fallback: no noisy reference — degrade to static median (old behaviour)
#         mask = cache.get("speech_mask_frames")
#         F, T = PX.shape
#         if mask is not None:
#             T = min(T, len(mask))
#             PX = PX[:, :T]
#             non = ~mask[:T]
#         else:
#             non = np.zeros(T, dtype=bool)
#         if non.sum() < 5:
#             return 0.0
#         N_f = np.maximum(np.median(PX[:, non], axis=1, keepdims=True), eps)
#         E_resid  = float(np.minimum(PX, alpha * N_f).sum())
#         E_speech = float(np.maximum(PX - alpha * N_f, 0.0).sum())
#         return min(E_resid / (E_speech + eps), 1e6)

#     # --- Align shapes ---
#     F  = min(PY.shape[0], PX.shape[0])
#     T  = min(PY.shape[1], PX.shape[1])
#     PY = np.asarray(PY[:F, :T], dtype=np.float32)
#     PX = np.asarray(PX[:F, :T], dtype=np.float32)

#     if T < 4:
#         return 0.0

#     # --- IMCRA-lite params (subset of full OMLSA params) ---
#     alpha_s   = 0.9      # temporal smoothing of freq-smoothed spectrum
#     alpha_d   = 0.85     # noise PSD update rate (slow, speech-absent frames)
#     delta_s   = 1.67     # IMCRA smoothed threshold
#     delta_y   = 4.6      # IMCRA local threshold
#     delta_yt  = 3.0      # IMCRA upper SNR cap for phat
#     Bmin      = 1.66     # bias correction
#     Nwin      = 8        # sliding window subframes
#     Vwin      = 15       # sliding window period

#     # Hann freq-smoothing kernel (half-width=1, same as full OMLSA)
#     b = np.hanning(3).astype(np.float32)
#     b /= b.sum()

#     def _fs(x):
#         return np.convolve(x, b, mode="same")

#     # --- Initial state (from first frame of PY) ---
#     Sf0    = _fs(np.maximum(PY[:, 0], eps))
#     S      = Sf0.copy()
#     St     = Sf0.copy()
#     Smin   = Sf0.copy()
#     SMact  = Sf0.copy()
#     Smint  = Sf0.copy()
#     SMactt = Sf0.copy()
#     SW     = np.tile(S[:, None],  (1, Nwin))
#     SWt    = np.tile(St[:, None], (1, Nwin))
#     lam_dav = np.maximum(PY[:, 0].copy(), eps)
#     lam_d   = 1.4685 * lam_dav
#     eta_2t  = np.ones(F, np.float32)
#     eta_min = 10 ** (-18.0 / 10)
#     l_sw    = 0

#     E_resid  = 0.0
#     E_speech = 0.0

#     for l in range(T):
#         Ya2 = np.maximum(PY[:, l], eps)
#         X2  = np.maximum(PX[:, l], 0.0)

#         # Freq smooth + temporal smooth (on noisy PY)
#         Sf = _fs(Ya2)
#         if l == 0:
#             S = Sf.copy(); St = Sf.copy()
#             Smin = S.copy(); SMact = S.copy()
#             Smint = St.copy(); SMactt = St.copy()
#             lam_dav = Ya2.copy()
#         else:
#             S = alpha_s * S + (1.0 - alpha_s) * Sf
#             if l < 14:
#                 Smin = S.copy(); SMact = S.copy()
#             else:
#                 Smin = np.minimum(Smin, S); SMact = np.minimum(SMact, S)

#         # IMCRA conditioned tracker
#         if l < 14:
#             St = S.copy(); Smint = St.copy(); SMactt = St.copy()
#         else:
#             I_f    = ((Ya2 < delta_y * Bmin * Smin) &
#                       (S   < delta_s * Bmin * Smin)).astype(np.float32)
#             cI     = _fs(I_f)
#             Sft    = St.copy()
#             idx    = cI > 0
#             if idx.any():
#                 Sft[idx] = _fs(I_f * Ya2)[idx] / np.maximum(cI[idx], eps)
#             St     = alpha_s * St + (1.0 - alpha_s) * Sft
#             Smint  = np.minimum(Smint, St)
#             SMactt = np.minimum(SMactt, St)

#         # Sliding window refresh
#         l_sw += 1
#         if l_sw == Vwin:
#             l_sw = 0
#             SW    = np.concatenate([SW[:,  1:], SMact[:, None]],  axis=1)
#             Smin  = np.min(SW,  axis=1); SMact  = S.copy()
#             SWt   = np.concatenate([SWt[:, 1:], SMactt[:, None]], axis=1)
#             Smint = np.min(SWt, axis=1); SMactt = St.copy()

#         # SNR for phat
#         gamma  = Ya2 / np.maximum(lam_d, eps)
#         eta    = 0.95 * eta_2t + 0.05 * np.maximum(gamma - 1.0, 0.0)
#         eta    = np.maximum(eta, eta_min)
#         v      = gamma * eta / (1.0 + eta)
#         eta_2t = eta

#         # IMCRA phat from Smint
#         g_mint = Ya2 / np.maximum(Bmin * Smint, eps)
#         zetat  = S   / np.maximum(Bmin * Smint, eps)
#         phat   = np.zeros(F, np.float32)
#         idx    = (g_mint > 1.0) & (g_mint < delta_yt) & (zetat < delta_s)
#         if idx.any():
#             qh       = (delta_yt - g_mint[idx]) / (delta_yt - 1.0)
#             phat[idx] = 1.0 / (1.0 + qh / np.maximum(1.0 - qh, eps)
#                                * (1.0 + eta[idx]) * np.exp(-v[idx]))
#         phat[(g_mint >= delta_yt) | (zetat >= delta_s)] = 1.0

#         # Noise PSD update → adaptive noise floor
#         a_dt    = alpha_d + (1.0 - alpha_d) * phat
#         lam_dav = a_dt * lam_dav + (1.0 - a_dt) * Ya2
#         lam_d   = np.maximum(1.4685 * lam_dav, eps)

#         # Wiener score accumulation against adaptive floor
#         floor    = alpha * lam_d
#         E_resid  += float(np.minimum(X2, floor).sum())
#         E_speech += float(np.maximum(X2 - floor, 0.0).sum())

#     return min(E_resid / (E_speech + eps), 1e6)

# dmd ver
def gate_step_wiener_tf(step_info: dict, cache: dict) -> float:
    """TF-domain noise-remnant gate for DEMAND-like noise. Lower = better.

    Main idea:
      Score how much of the noisy background structure still remains inside PX,
      not just how much PX falls below an adaptive floor.

    Uses:
      - PY: noisy power spectrum in model TF domain
      - PX: current latent/enhanced power spectrum in model TF domain
      - IMCRA-lite adaptive noise tracker on PY -> lam_d
      - phat: per-bin speech-presence probability

    Score:
      numerator   = residual matched-to-noise + noise-dominant excess
      denominator = speech-preserving excess in speech-present bins

    This is more appropriate for DEMAND-like real environmental noise, where
    leftover noise may remain both below and above the estimated noise floor.

    step_info keys:
      xt_mean : torch.Tensor [B, C, F, T_spec]

    cache keys:
      y_PY    : np.ndarray [F, T_spec]   (preferred; model TF noisy power)
      eps     : float                    (optional, default 1e-8)

    optional cache keys:
      wiener_alpha : float  noise-floor scale in numerator (default 1.15)
      speech_gamma : float  softer speech subtraction in denominator (default 0.50)
    """
    import numpy as np

    xt_mean = step_info["xt_mean"]                                   # [B, C, F, T]
    eps     = cache.get("eps", 1e-8)

    # DEMAND-oriented score hyperparameters
    beta  = cache.get("wiener_alpha", 1.15)   # noise floor scale for residual matching
    gamma = cache.get("speech_gamma", 0.50)   # softer subtraction for speech denominator

    # Weights for final numerator
    W_RESID_MATCH  = 1.00
    W_NOISE_EXCESS = 1.35

    PX = (xt_mean[0].abs() ** 2).sum(dim=0).detach().cpu().numpy()   # [F, T]

    if "y_PY" in cache:
        PY = cache["y_PY"]
    else:
        # Fallback: old-style static fallback if noisy TF reference is unavailable
        mask = cache.get("speech_mask_frames")
        F, T = PX.shape
        if mask is not None:
            T = min(T, len(mask))
            PX = PX[:, :T]
            non = ~mask[:T]
        else:
            non = np.zeros(T, dtype=bool)

        if non.sum() < 5:
            return 0.0

        N_f = np.maximum(np.median(PX[:, non], axis=1, keepdims=True), eps)
        resid_match  = np.minimum(PX, beta * N_f)
        noise_excess = np.maximum(PX - beta * N_f, 0.0)
        speech_keep  = np.maximum(PX - gamma * N_f, 0.0)

        E_resid  = float((W_RESID_MATCH * resid_match + W_NOISE_EXCESS * noise_excess).sum())
        E_speech = float(speech_keep.sum())
        return min(E_resid / (E_speech + eps), 1e6)

    # --- Align shapes ---
    F = min(PY.shape[0], PX.shape[0])
    T = min(PY.shape[1], PX.shape[1])
    PY = np.asarray(PY[:F, :T], dtype=np.float32)
    PX = np.asarray(PX[:F, :T], dtype=np.float32)

    if T < 4:
        return 0.0

    # ===== DEMAND-oriented IMCRA-lite parameters =====
    alpha_s   = 0.82   # faster temporal smoothing
    alpha_d   = 0.72   # faster noise PSD update
    delta_s   = 1.45
    delta_y   = 3.20
    delta_yt  = 2.20
    Bmin      = 1.66
    Nwin      = 6
    Vwin      = 8

    # a-priori SNR recursion
    alpha_eta = 0.90
    eta_min   = 10 ** (-16.0 / 10.0)

    # Frequency smoothing kernel
    b = np.hanning(3).astype(np.float32)
    b /= np.maximum(b.sum(), eps)

    def _fs(x: np.ndarray) -> np.ndarray:
        return np.convolve(x, b, mode="same")

    # --- Initial state ---
    Sf0     = _fs(np.maximum(PY[:, 0], eps))
    S       = Sf0.copy()
    St      = Sf0.copy()
    Smin    = Sf0.copy()
    SMact   = Sf0.copy()
    Smint   = Sf0.copy()
    SMactt  = Sf0.copy()
    SW      = np.tile(S[:, None],  (1, Nwin))
    SWt     = np.tile(St[:, None], (1, Nwin))

    lam_dav = np.maximum(PY[:, 0].copy(), eps)
    lam_d   = np.maximum(1.4685 * lam_dav, eps)

    eta_2t  = np.ones(F, dtype=np.float32)
    l_sw    = 0

    E_resid  = 0.0
    E_speech = 0.0

    for l in range(T):
        Ya2 = np.maximum(PY[:, l], eps)     # noisy power
        X2  = np.maximum(PX[:, l], 0.0)     # cleaned/latent power

        # ----- Smoothed noisy spectrum -----
        Sf = _fs(Ya2)
        if l == 0:
            S = Sf.copy()
            St = Sf.copy()
            Smin = S.copy()
            SMact = S.copy()
            Smint = St.copy()
            SMactt = St.copy()
            lam_dav = Ya2.copy()
        else:
            S = alpha_s * S + (1.0 - alpha_s) * Sf
            if l < 14:
                Smin = S.copy()
                SMact = S.copy()
            else:
                Smin = np.minimum(Smin, S)
                SMact = np.minimum(SMact, S)

        # ----- IMCRA-like conditioned update -----
        if l < 14:
            St = S.copy()
            Smint = St.copy()
            SMactt = St.copy()
        else:
            I_f = ((Ya2 < delta_y * Bmin * Smin) &
                   (S   < delta_s * Bmin * Smin)).astype(np.float32)

            cI  = _fs(I_f)
            Sft = St.copy()
            idx = cI > 0
            if np.any(idx):
                Sft[idx] = _fs(I_f * Ya2)[idx] / np.maximum(cI[idx], eps)

            St     = alpha_s * St + (1.0 - alpha_s) * Sft
            Smint  = np.minimum(Smint, St)
            SMactt = np.minimum(SMactt, St)

        # ----- Sliding minima refresh -----
        l_sw += 1
        if l_sw == Vwin:
            l_sw = 0
            SW    = np.concatenate([SW[:, 1:], SMact[:, None]], axis=1)
            Smin  = np.min(SW, axis=1)
            SMact = S.copy()

            SWt    = np.concatenate([SWt[:, 1:], SMactt[:, None]], axis=1)
            Smint  = np.min(SWt, axis=1)
            SMactt = St.copy()

        # ----- SNR recursion -----
        gamma_post = Ya2 / np.maximum(lam_d, eps)
        eta        = alpha_eta * eta_2t + (1.0 - alpha_eta) * np.maximum(gamma_post - 1.0, 0.0)
        eta        = np.maximum(eta, eta_min)
        v          = gamma_post * eta / (1.0 + eta)
        eta_2t     = eta

        # ----- Speech presence probability phat -----
        g_mint = Ya2 / np.maximum(Bmin * Smint, eps)
        zetat  = S   / np.maximum(Bmin * Smint, eps)

        phat = np.zeros(F, dtype=np.float32)
        idx = (g_mint > 1.0) & (g_mint < delta_yt) & (zetat < delta_s)
        if np.any(idx):
            qh = (delta_yt - g_mint[idx]) / (delta_yt - 1.0)
            phat[idx] = 1.0 / (
                1.0
                + qh / np.maximum(1.0 - qh, eps)
                * (1.0 + eta[idx])
                * np.exp(-v[idx])
            )
        phat[(g_mint >= delta_yt) | (zetat >= delta_s)] = 1.0

        # ----- Adaptive noise PSD -----
        a_dt    = alpha_d + (1.0 - alpha_d) * phat
        lam_dav = a_dt * lam_dav + (1.0 - a_dt) * Ya2
        lam_d   = np.maximum(1.4685 * lam_dav, eps)

        # ============================================================
        # New DEMAND-oriented noise-remnant score
        # ============================================================
        # N_ref = estimated background-noise structure in the noisy signal
        N_ref = lam_d

        speech_conf = phat                       # [F], soft speech presence
        noise_conf  = 1.0 - speech_conf

        # 1) matched residual: energy in X2 that still matches estimated noise floor
        resid_match = np.minimum(X2, beta * N_ref)

        # 2) noise-dominant excess: energy above floor but in bins likely to be noise
        noise_excess = noise_conf * np.maximum(X2 - beta * N_ref, 0.0)

        # 3) speech-preserving excess: only count speech-present bins as useful
        speech_keep = speech_conf * np.maximum(X2 - gamma * N_ref, 0.0)

        E_resid  += float((W_RESID_MATCH * resid_match + W_NOISE_EXCESS * noise_excess).sum())
        E_speech += float(speech_keep.sum())

    score = E_resid / (E_speech + eps)
    if not np.isfinite(score):
        return 1e6
    return min(float(score), 1e6)


# def gate_step_omlsa_residual(step_info: dict, cache: dict) -> float:
#     """Mid-step OMLSA-residual gate.

#     Converts the intermediate latent estimate to time-domain audio, then
#     applies _omlsa_residual_score(y_np, x_hat_mid_np).  Lower = better.

#     cache keys required:
#         model       : ScoreModel — to call model.to_audio
#         T_orig      : int        — original waveform length
#         norm_factor : float      — amplitude normalisation factor
#         y_np        : np.ndarray [T] — normalised noisy input waveform
#     """
#     xt_mean     = step_info["xt_mean"]
#     model       = cache["model"]
#     T_orig      = cache["T_orig"]
#     norm_factor = cache["norm_factor"]
#     y_np        = cache["y_np"]

#     x_hat_mid = model.to_audio(xt_mean.squeeze(), T_orig) * norm_factor
#     return _omlsa_residual_score(y_np, x_hat_mid.cpu().numpy())


def _omlsa_residual_tf_score(PY: np.ndarray, PX: np.ndarray, eps: float = 1e-10) -> float:
    """OM-LSA-inspired TF-domain gate. Lower = better.

    Identical logic to _omlsa_residual_score but accepts pre-computed power
    spectra PY [F, T] (noisy) and PX [F, T] (enhanced / latent) directly,
    so no waveform-to-spectrum conversion is needed.  This preserves the
    mid-gate property: xt_mean stays in TF domain and is never converted to
    a waveform.

    PY is used for IMCRA noise tracking (same as _omlsa_residual_score uses
    the noisy waveform y).  PX is scored against the derived noise floor
    (same as _omlsa_residual_score uses x_hat).  If T or F differ between PY
    and PX the shorter dimension is used.

    n_fft is inferred as 2*(F-1) to map bin indices to Hz.  Fs=16000 Hz
    is assumed (VoiceBank).
    """
    import numpy as np
    from scipy.special import exp1

    EPS = eps
    MAX_SCORE = 1e6

    # Align shapes
    M21 = min(PY.shape[0], PX.shape[0])
    n_frames = min(PY.shape[1], PX.shape[1])
    PY = np.asarray(PY[:M21, :n_frames], dtype=np.float32)
    PX = np.asarray(PX[:M21, :n_frames], dtype=np.float32)

    if n_frames < 2:
        return MAX_SCORE

    # Infer n_fft from F bins (assumes real-FFT: n_fft = 2*(F-1))
    M   = 2 * (M21 - 1)
    Fs  = 16000.0

    # ===== OM-LSA params =====
    w = 1                   # hann window len 2w+1 -> 3-bin freq-smoothing
    alpha_s_ref = 0.9       # time-smoothing of noise spectrum (higher = slower, Low fast but noisy)
    Nwin = 8                # Minima tracking: how often we update the buffer
    Vwin = 15               # Minima tracking: how many past segments we keep
    
    # IMCRA
    delta_s = 1.67          # smoothing constraint (stability)
    Bmin = 1.66             # bias correction factor for minimum statistics
    delta_y = 4.6           # thresholds for noise vs speech detection
    delta_yt = 3.0
    alpha_d_long = 0.99

    alpha_xi_ref = 0.7      # decision-directed smoothing (high = smooth, low- reactive+noisy)
    w_xi_local = 1          # Local smoothing of zeta
    w_xi_global = 15        # global smoothing of zeta
    f_u = 10000.0           # Frequency range of interest- high end
    f_l = 50.0              # Frequency range of interest - low end
    P_min = 0.005
    xi_lu_dB = -5.0         # upper decision threshold for local a-priori SNR (zeta)
    xi_ll_dB = -10.0        # lower decision threshold for local a-priori SNR (zeta)
    xi_gu_dB = -5.0         # upper decision threshold for global a-priori SNR (zeta)
    xi_gl_dB = -10.0        # lower decision threshold for global a-priori SNR (zeta)
    xi_fu_dB = -5.0         # upper decision threshold for fundamental frequency (zeta)
    xi_fl_dB = -10.0        # lower decision threshold for fundamental frequency (zeta)
    xi_mu_dB = 10.0
    xi_ml_dB = 0.0
    q_max = 0.998

    alpha_eta_ref = 0.95    # A posteriori / decision-directed SNR smoothing (high = smooth, low = reactive+noisy)
    eta_min_dB = -18.0

    # --- Dataset-dependent parameters ---
    # Switch the active block to match your evaluation domain.
    #
    # VoiceBank-DEMAND (in-domain, near-stationary additive noise):
    #   slow noise tracker is sufficient; tonal detection catches real speech
    #   harmonics; lower noise floor is appropriate for mild noise levels.
    # alpha_d_ref = 0.85   # slow tracker — noise is nearly stationary
    # broad_flag  = True
    # tone_flag   = True   # speech harmonics are real; tonal flag is meaningful
    # nonstat     = "medium"  # lambda_d = 1.4685 * lambda_dav
    # FRAME_BOOST    = 0.75
    # W_SPEECH_RESID = 0.70
    # W_NOISE_EXCESS = 1.00
    # W_SPEECH_HOLE  = 0.18
    # W_TONAL        = 0.35
    #
    # DNS-noreverb (cross-domain, non-stationary / music / babble noise):
    #   faster tracker needed because DNS noise changes rapidly; music/harmonic
    #   noise misfires the tonal detector so it is disabled; higher noise floor
    #   (nonstat="high") is more conservative and suits energetic DNS noise;
    #   upweight W_NOISE_EXCESS since leakage is the primary failure mode here.
    alpha_d_ref = 0.70   # faster tracker — DNS noise is non-stationary
    broad_flag  = True
    tone_flag   = False  # music/harmonic noise falsely triggers tonal detection
    nonstat     = "high" # lambda_d = 2.0 * lambda_dav — more conservative floor
    FRAME_BOOST    = 0.75
    W_SPEECH_RESID = 0.70
    W_NOISE_EXCESS = 1.30  # was 1.00 — upweight leakage, main failure mode in DNS
    W_SPEECH_HOLE  = 0.28  # was 0.18
    W_TONAL        = 0.00  # disabled (tone_flag=False)

    alpha_s   = alpha_s_ref
    alpha_d   = alpha_d_ref
    alpha_eta = alpha_eta_ref
    alpha_xi  = alpha_xi_ref


    # op3- demand optimized
    # ===== DEMAND-oriented preset =====
    w = 1
    alpha_s_ref = 0.82          # was 0.9
    Nwin = 6                    # was 8
    Vwin = 8                    # was 15
    delta_s = 1.45              # was 1.67
    Bmin = 1.66
    delta_y = 3.2               # was 4.6
    delta_yt = 2.2              # was 3.0
    alpha_d_ref = 0.72          # was 0.85
    alpha_d_long = 0.96         # was 0.99

    alpha_xi_ref = 0.55         # was 0.7
    w_xi_local = 1
    w_xi_global = 9             # was 15
    f_u = 8000.0                # was 10000.0
    f_l = 80.0                  # was 50.0
    P_min = 0.02                # was 0.005
    xi_lu_dB = -4.0             # was -5
    xi_ll_dB = -9.0             # was -10
    xi_gu_dB = -4.0             # was -5
    xi_gl_dB = -9.0             # was -10
    xi_fu_dB = -4.0             # was -5
    xi_fl_dB = -9.0             # was -10
    xi_mu_dB = 8.0              # was 10
    xi_ml_dB = -1.0             # was 0
    q_max = 0.992               # was 0.998

    alpha_eta_ref = 0.90        # was 0.95
    eta_min_dB = -16.0          # was -18.0

    broad_flag = True
    tone_flag = False           # was True
    nonstat = "high"            # was "medium"

    FRAME_BOOST    = 0.35       # was 0.75
    W_SPEECH_RESID = 1.10       # was 0.70
    W_NOISE_EXCESS = 1.70       # was 1.00
    W_SPEECH_HOLE  = 0.05       # was 0.18
    W_TONAL        = 0.00       # was 0.35
    # ===== DEMAND-oriented preset end =====


    eta_min = 10.0 ** (eta_min_dB / 10.0)
    G_f     = eta_min ** 0.5

    # ===== Smoothing kernels =====
    b = np.hanning(2 * w + 1).astype(np.float32)
    b /= np.sum(b)

    b_xi_local = np.hanning(2 * w_xi_local + 1).astype(np.float32)
    b_xi_local /= np.sum(b_xi_local)

    b_xi_global = np.hanning(2 * w_xi_global + 1).astype(np.float32)
    b_xi_global /= np.sum(b_xi_global)

    # ===== Frequency ranges =====
    k_u = int(round(f_u / Fs * M + 1))
    k_l = int(round(f_l / Fs * M + 1))
    k_u = min(k_u, M21)
    k_l = max(1, min(k_l, M21 - 1))

    k2_local = int(round(500.0 / Fs * M + 1))
    k3_local = int(round(3500.0 / Fs * M + 1))
    k2_local = max(1, min(k2_local, M21 - 1))
    k3_local = max(k2_local + 1, min(k3_local, M21 - 1))

    # ===== Helpers =====
    def _conv_same(x: np.ndarray, h: np.ndarray) -> np.ndarray:
        return np.convolve(x, h, mode="same")

    def _db_prob(x_db: np.ndarray, lo: float, hi: float, pmin: float) -> np.ndarray:
        out = np.ones_like(x_db, dtype=np.float32)
        out[x_db <= lo] = pmin
        mid = (x_db > lo) & (x_db < hi)
        out[mid] = pmin + (x_db[mid] - lo) / (hi - lo) * (1.0 - pmin)
        return out

    # ===== Initial state =====
    eta_2term = np.ones(M21, dtype=np.float32)
    xi = np.zeros(M21, dtype=np.float32)
    xi_frame = 0.0
    xi_m_dB = xi_ml_dB       # running P_frame reference level
    l_mod_lswitch = 0

    lambda_d = np.maximum(PY[:, 0].copy(), EPS)
    Sy = PY[:, 0].copy()
    Sf0 = _conv_same(PY[:, 0], b)
    S = Sf0.copy()
    St = Sf0.copy()
    lambda_dav = PY[:, 0].copy()
    lambda_dav_long = PY[:, 0].copy()
    Smin = S.copy()
    SMact = S.copy()
    Smint = St.copy()
    SMactt = St.copy()

    SW = np.tile(S[:, None], (1, Nwin))
    SWt = np.tile(St[:, None], (1, Nwin))

    score_num = 0.0
    score_den = 0.0

    for l in range(n_frames):
        Ya2 = np.maximum(PY[:, l], EPS)
        X2  = np.maximum(PX[:, l], 0.0)

        gamma = Ya2 / np.maximum(lambda_d, EPS)
        eta = alpha_eta * eta_2term + (1.0 - alpha_eta) * np.maximum(gamma - 1.0, 0.0)
        eta = np.maximum(eta, eta_min)
        v = gamma * eta / (1.0 + eta)

        # 2.1 frequency smoothing
        Sf = _conv_same(Ya2, b)

        if l == 0:
            S = Sf.copy()
            St = Sf.copy()
            Smin = S.copy()
            SMact = S.copy()
            Smint = St.copy()
            SMactt = St.copy()
            lambda_dav = Ya2.copy()
            lambda_dav_long = Ya2.copy()
            Sy = Ya2.copy()
        else:
            S = alpha_s * S + (1.0 - alpha_s) * Sf
            if l < 14:
                Smin = S.copy()
                SMact = S.copy()
            else:
                Smin = np.minimum(Smin, S)
                SMact = np.minimum(SMact, S)

        # IMCRA local minima logic
        I_f = ((Ya2 < delta_y * Bmin * Smin) & (S < delta_s * Bmin * Smin)).astype(np.float32)
        conv_I = _conv_same(I_f, b)
        Sft = St.copy()
        idx = conv_I > 0
        if np.any(idx):
            conv_Y = _conv_same(I_f * Ya2, b)
            Sft[idx] = conv_Y[idx] / np.maximum(conv_I[idx], EPS)

        if l < 14:
            St = S.copy()
            Smint = St.copy()
            SMactt = St.copy()
        else:
            St = alpha_s * St + (1.0 - alpha_s) * Sft
            Smint = np.minimum(Smint, St)
            SMactt = np.minimum(SMactt, St)

        qhat = np.ones(M21, dtype=np.float32)
        phat = np.zeros(M21, dtype=np.float32)

        if nonstat == "low":
            gamma_mint = Ya2 / (Bmin * np.maximum(Smin, EPS))
            zetat = S / (Bmin * np.maximum(Smin, EPS))
        else:
            gamma_mint = Ya2 / (Bmin * np.maximum(Smint, EPS))
            zetat = S / (Bmin * np.maximum(Smint, EPS))

        idx = (gamma_mint > 1.0) & (gamma_mint < delta_yt) & (zetat < delta_s)
        qhat[idx] = (delta_yt - gamma_mint[idx]) / (delta_yt - 1.0)
        phat[idx] = 1.0 / (
            1.0
            + qhat[idx] / np.maximum(1.0 - qhat[idx], EPS)
            * (1.0 + eta[idx])
            * np.exp(-v[idx])
        )
        phat[(gamma_mint >= delta_yt) | (zetat >= delta_s)] = 1.0

        alpha_dt = alpha_d + (1.0 - alpha_d) * phat
        lambda_dav = alpha_dt * lambda_dav + (1.0 - alpha_dt) * Ya2

        if l < 14:
            lambda_dav_long = lambda_dav.copy()
        else:
            alpha_dt_long = alpha_d_long + (1.0 - alpha_d_long) * phat
            lambda_dav_long = alpha_dt_long * lambda_dav_long + (1.0 - alpha_dt_long) * Ya2

        # sliding minima window
        l_mod_lswitch += 1
        if l_mod_lswitch == Vwin:
            l_mod_lswitch = 0
            if l == Vwin - 1:
                SW = np.tile(S[:, None], (1, Nwin))
                SWt = np.tile(St[:, None], (1, Nwin))
            else:
                SW = np.concatenate([SW[:, 1:], SMact[:, None]], axis=1)
                Smin = np.min(SW, axis=1)
                SMact = S.copy()

                SWt = np.concatenate([SWt[:, 1:], SMactt[:, None]], axis=1)
                Smint = np.min(SWt, axis=1)
                SMactt = St.copy()

        if nonstat == "high":
            lambda_d = 2.0 * lambda_dav
        else:
            lambda_d = 1.4685 * lambda_dav
        lambda_d = np.maximum(lambda_d, EPS)

        # ===== A priori probability of signal absence =====
        xi = alpha_xi * xi + (1.0 - alpha_xi) * eta
        xi_local  = _conv_same(xi, b_xi_local)
        xi_global = _conv_same(xi, b_xi_global)

        prev_xi_frame = xi_frame
        xi_frame  = float(np.mean(xi[k_l:k_u]))
        dxi_frame = xi_frame - prev_xi_frame

        xi_local_dB  = 10.0 * np.log10(np.maximum(xi_local,  1e-10))
        xi_global_dB = 10.0 * np.log10(np.maximum(xi_global, 1e-10))
        xi_frame_dB  = 10.0 * np.log10(max(xi_frame, 1e-10))

        P_local  = _db_prob(xi_local_dB,  xi_ll_dB, xi_lu_dB, P_min)
        P_global = _db_prob(xi_global_dB, xi_gl_dB, xi_gu_dB, P_min)

        lo = min(3, M21 - 1)
        hi = min(k2_local + k3_local - 3, M21)
        if hi > lo:
            m_P_local = float(np.mean(P_local[lo:hi]))
        else:
            m_P_local = float(np.mean(P_local))

        tonal_mask = np.zeros(M21, dtype=np.float32)

        if m_P_local < 0.25:
            P_local[k2_local:k3_local] = P_min

        if tone_flag and (m_P_local < 0.5) and (l > 120) and M21 > 16:
            lhs = lambda_dav_long[7:(M21 - 8)]
            rhs = 2.5 * (
                lambda_dav_long[9:(M21 - 6)] + lambda_dav_long[5:(M21 - 10)]
            )
            tonal_idx = np.where(lhs > rhs)[0] + 6
            tonal_idx = tonal_idx[(tonal_idx >= 0) & (tonal_idx < M21)]
            if tonal_idx.size > 0:
                P_local[tonal_idx] = P_min
                tonal_mask[tonal_idx] = 1.0

        if xi_frame_dB <= xi_fl_dB:
            P_frame = P_min
        elif dxi_frame >= 0:
            xi_m_dB = min(max(xi_frame_dB, xi_ml_dB), xi_mu_dB)
            P_frame = 1.0
        elif xi_frame_dB >= xi_m_dB + xi_fu_dB:
            P_frame = 1.0
        elif xi_frame_dB <= xi_m_dB + xi_fl_dB:
            P_frame = P_min
        else:
            P_frame = P_min + (
                (xi_frame_dB - xi_m_dB - xi_fl_dB) / (xi_fu_dB - xi_fl_dB)
            ) * (1.0 - P_min)

        if broad_flag:
            q = 1.0 - P_global * P_local * P_frame
        else:
            q = 1.0 - P_local * P_frame
        q = np.minimum(q, q_max)

        gamma = Ya2 / np.maximum(lambda_d, EPS)
        eta = alpha_eta * eta_2term + (1.0 - alpha_eta) * np.maximum(gamma - 1.0, 0.0)
        eta = np.maximum(eta, eta_min)
        v = gamma * eta / (1.0 + eta)

        PH1 = np.zeros(M21, dtype=np.float32)
        idx = q < 0.9
        PH1[idx] = 1.0 / (
            1.0
            + q[idx] / np.maximum(1.0 - q[idx], EPS)
            * (1.0 + eta[idx])
            * np.exp(-v[idx])
        )

        # ===== Spectral gains =====
        GH1 = np.ones(M21, dtype=np.float32)
        idx_hi = v > 5.0
        GH1[idx_hi] = eta[idx_hi] / (1.0 + eta[idx_hi])

        idx_mid = (v > 0.0) & (v <= 5.0)
        if np.any(idx_mid):
            vv = np.maximum(v[idx_mid], 1e-8)
            GH1[idx_mid] = (
                eta[idx_mid] / (1.0 + eta[idx_mid]) * np.exp(0.5 * exp1(vv))
            )

        if tone_flag:
            lambda_d_global = lambda_d.copy()
            if M21 > 6:
                tmp = lambda_d_global.copy()
                tmp[3:(M21 - 3)] = np.minimum.reduce([
                    lambda_d_global[3:(M21 - 3)],
                    lambda_d_global[0:(M21 - 6)],
                    lambda_d_global[6:M21],
                ])
                lambda_d_global = tmp

            Sy = 0.8 * Sy + 0.2 * Ya2
            GH0 = G_f * np.sqrt(lambda_d_global / np.maximum(Sy, EPS))
        else:
            GH0 = np.full(M21, G_f, dtype=np.float32)

        G = (GH1 ** PH1) * (GH0 ** (1.0 - PH1))  # noqa: F841  (unused but matches _omlsa_residual_score)
        eta_2term = (GH1 ** 2) * gamma

        # ===== Score =====
        frame_weight = 1.0 + FRAME_BOOST * (1.0 - float(P_frame))

        target_floor = np.maximum(lambda_d, EPS)

        speech_resid  = np.sum(PH1 * np.minimum(X2, target_floor))
        noise_excess  = np.sum((1.0 - PH1) * X2)
        speech_keep   = np.sum(PH1 * X2)
        speech_hole   = np.sum(PH1 * GH1 * np.maximum(target_floor - X2, 0.0))
        tonal_penalty = np.sum(tonal_mask * np.maximum(X2 - GH0 * target_floor, 0.0))

        frame_num = (
            W_SPEECH_RESID * speech_resid
            + W_NOISE_EXCESS * noise_excess
            + W_SPEECH_HOLE * speech_hole
            + W_TONAL * tonal_penalty
        )

        score_num += float(frame_weight * frame_num)
        score_den += float(speech_keep + EPS)

    score = score_num / (score_den + EPS)
    if not np.isfinite(score):
        return MAX_SCORE
    return min(float(score), MAX_SCORE)



def _omlsa_gating_score(
    PY: np.ndarray,
    PX: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """OMLSA/IMCRA gating score. Higher = better.

    IMCRA noise tracking on PY → PH1 (speech-presence probability per TF bin).
    Score = Σ PH1·X² / Σ (1-PH1)·X²  (speech energy / noise energy in PX).
    """
    import numpy as np
    from scipy.special import exp1

    EPS = eps
    MAX_SCORE = 1e6

    M21 = min(PY.shape[0], PX.shape[0])
    n_frames = min(PY.shape[1], PX.shape[1])
    PY = np.asarray(PY[:M21, :n_frames], dtype=np.float32)
    PX = np.asarray(PX[:M21, :n_frames], dtype=np.float32)

    if n_frames < 2:
        return MAX_SCORE

    M  = 2 * (M21 - 1)
    Fs = 16000.0

    # ===== Parameters (DEMAND-optimized preset) =====
    w           = 1
    alpha_s     = 0.9     # spectral smoothing — from original block, before DEMAND override
    alpha_d     = 0.70    # noise update rate  — from original block, before DEMAND override
    alpha_eta   = 0.95    # a priori SNR smoothing — from original block
    alpha_xi    = 0.7     # xi smoothing — from original block
    Nwin        = 6
    Vwin        = 8
    delta_s     = 1.45
    Bmin        = 1.66
    delta_y     = 3.2
    delta_yt    = 2.2
    w_xi_local  = 1
    w_xi_global = 9
    f_u         = 8000.0
    f_l         = 80.0
    P_min       = 0.02
    xi_lu_dB    = -4.0
    xi_ll_dB    = -9.0
    xi_gu_dB    = -4.0
    xi_gl_dB    = -9.0
    xi_fu_dB    = -4.0
    xi_fl_dB    = -9.0
    xi_mu_dB    = 8.0
    xi_ml_dB    = -1.0
    q_max       = 0.992
    eta_min     = 10.0 ** (-16.0 / 10.0)

    # ===== Smoothing kernels =====
    b = np.hanning(2 * w + 1).astype(np.float32)
    b /= np.sum(b)

    b_xi_local = np.hanning(2 * w_xi_local + 1).astype(np.float32)
    b_xi_local /= np.sum(b_xi_local)

    b_xi_global = np.hanning(2 * w_xi_global + 1).astype(np.float32)
    b_xi_global /= np.sum(b_xi_global)

    # ===== Frequency ranges =====
    k_u = min(int(round(f_u / Fs * M + 1)), M21)
    k_l = max(1, min(int(round(f_l / Fs * M + 1)), M21 - 1))

    k2_local = max(1, min(int(round(500.0 / Fs * M + 1)), M21 - 1))
    k3_local = max(k2_local + 1, min(int(round(3500.0 / Fs * M + 1)), M21 - 1))

    def _conv_same(x: np.ndarray, h: np.ndarray) -> np.ndarray:
        return np.convolve(x, h, mode="same")

    def _db_prob(x_db: np.ndarray, lo: float, hi: float, pmin: float) -> np.ndarray:
        out = np.ones_like(x_db, dtype=np.float32)
        out[x_db <= lo] = pmin
        mid = (x_db > lo) & (x_db < hi)
        out[mid] = pmin + (x_db[mid] - lo) / (hi - lo) * (1.0 - pmin)
        return out

    # ===== Initial state =====
    eta_2term = np.ones(M21, dtype=np.float32)
    xi = np.zeros(M21, dtype=np.float32)
    xi_frame = 0.0
    xi_m_dB = xi_ml_dB
    l_mod_lswitch = 0

    lambda_d = np.maximum(PY[:, 0].copy(), EPS)
    Sf0 = _conv_same(PY[:, 0], b)
    S = Sf0.copy()
    St = Sf0.copy()
    lambda_dav = PY[:, 0].copy()
    Smin = S.copy()
    SMact = S.copy()
    Smint = St.copy()
    SMactt = St.copy()

    SW  = np.tile(S[:, None],  (1, Nwin))
    SWt = np.tile(St[:, None], (1, Nwin))

    score_num = 0.0
    score_den = 0.0

    for l in range(n_frames):
        Ya2 = np.maximum(PY[:, l], EPS)
        X2  = np.maximum(PX[:, l], 0.0)

        gamma = Ya2 / np.maximum(lambda_d, EPS)
        eta = alpha_eta * eta_2term + (1.0 - alpha_eta) * np.maximum(gamma - 1.0, 0.0)
        eta = np.maximum(eta, eta_min)
        v = gamma * eta / (1.0 + eta)

        Sf = _conv_same(Ya2, b)

        if l == 0:
            S = Sf.copy()
            St = Sf.copy()
            Smin = S.copy()
            SMact = S.copy()
            Smint = St.copy()
            SMactt = St.copy()
            lambda_dav = Ya2.copy()
        else:
            S = alpha_s * S + (1.0 - alpha_s) * Sf
            if l < 14:
                Smin = S.copy()
                SMact = S.copy()
            else:
                Smin = np.minimum(Smin, S)
                SMact = np.minimum(SMact, S)

        I_f = ((Ya2 < delta_y * Bmin * Smin) & (S < delta_s * Bmin * Smin)).astype(np.float32)
        conv_I = _conv_same(I_f, b)
        Sft = St.copy()
        idx = conv_I > 0
        if np.any(idx):
            conv_Y = _conv_same(I_f * Ya2, b)
            Sft[idx] = conv_Y[idx] / np.maximum(conv_I[idx], EPS)

        if l < 14:
            St = S.copy()
            Smint = St.copy()
            SMactt = St.copy()
        else:
            St = alpha_s * St + (1.0 - alpha_s) * Sft
            Smint = np.minimum(Smint, St)
            SMactt = np.minimum(SMactt, St)

        qhat = np.ones(M21, dtype=np.float32)
        phat = np.zeros(M21, dtype=np.float32)

        # nonstat="high": use Smint for local-minima reference
        gamma_mint = Ya2 / (Bmin * np.maximum(Smint, EPS))
        zetat = S / (Bmin * np.maximum(Smint, EPS))

        idx = (gamma_mint > 1.0) & (gamma_mint < delta_yt) & (zetat < delta_s)
        qhat[idx] = (delta_yt - gamma_mint[idx]) / (delta_yt - 1.0)
        phat[idx] = 1.0 / (
            1.0
            + qhat[idx] / np.maximum(1.0 - qhat[idx], EPS)
            * (1.0 + eta[idx])
            * np.exp(-v[idx])
        )
        phat[(gamma_mint >= delta_yt) | (zetat >= delta_s)] = 1.0

        alpha_dt = alpha_d + (1.0 - alpha_d) * phat
        lambda_dav = alpha_dt * lambda_dav + (1.0 - alpha_dt) * Ya2

        l_mod_lswitch += 1
        if l_mod_lswitch == Vwin:
            l_mod_lswitch = 0
            if l == Vwin - 1:
                SW  = np.tile(S[:, None],  (1, Nwin))
                SWt = np.tile(St[:, None], (1, Nwin))
            else:
                SW = np.concatenate([SW[:, 1:], SMact[:, None]], axis=1)
                Smin = np.min(SW, axis=1)
                SMact = S.copy()

                SWt = np.concatenate([SWt[:, 1:], SMactt[:, None]], axis=1)
                Smint = np.min(SWt, axis=1)
                SMactt = St.copy()

        # nonstat="high": lambda_d = 2 * lambda_dav
        lambda_d = np.maximum(2.0 * lambda_dav, EPS)

        xi = alpha_xi * xi + (1.0 - alpha_xi) * eta
        xi_local  = _conv_same(xi, b_xi_local)
        xi_global = _conv_same(xi, b_xi_global)

        prev_xi_frame = xi_frame
        xi_frame  = float(np.mean(xi[k_l:k_u]))
        dxi_frame = xi_frame - prev_xi_frame

        xi_local_dB  = 10.0 * np.log10(np.maximum(xi_local,  1e-10))
        xi_global_dB = 10.0 * np.log10(np.maximum(xi_global, 1e-10))
        xi_frame_dB  = 10.0 * np.log10(max(xi_frame, 1e-10))

        P_local  = _db_prob(xi_local_dB,  xi_ll_dB, xi_lu_dB, P_min)
        P_global = _db_prob(xi_global_dB, xi_gl_dB, xi_gu_dB, P_min)

        lo = min(3, M21 - 1)
        hi = min(k2_local + k3_local - 3, M21)
        m_P_local = float(np.mean(P_local[lo:hi])) if hi > lo else float(np.mean(P_local))

        if m_P_local < 0.25:
            P_local[k2_local:k3_local] = P_min

        if xi_frame_dB <= xi_fl_dB:
            P_frame = P_min
        elif dxi_frame >= 0:
            xi_m_dB = min(max(xi_frame_dB, xi_ml_dB), xi_mu_dB)
            P_frame = 1.0
        elif xi_frame_dB >= xi_m_dB + xi_fu_dB:
            P_frame = 1.0
        elif xi_frame_dB <= xi_m_dB + xi_fl_dB:
            P_frame = P_min
        else:
            P_frame = P_min + (
                (xi_frame_dB - xi_m_dB - xi_fl_dB) / (xi_fu_dB - xi_fl_dB)
            ) * (1.0 - P_min)

        # broad_flag=True: q uses P_global * P_local * P_frame
        q = np.minimum(1.0 - P_global * P_local * P_frame, q_max)

        gamma = Ya2 / np.maximum(lambda_d, EPS)
        eta = alpha_eta * eta_2term + (1.0 - alpha_eta) * np.maximum(gamma - 1.0, 0.0)
        eta = np.maximum(eta, eta_min)
        v = gamma * eta / (1.0 + eta)

        PH1 = np.zeros(M21, dtype=np.float32)
        idx = q < 0.9
        PH1[idx] = 1.0 / (
            1.0
            + q[idx] / np.maximum(1.0 - q[idx], EPS)
            * (1.0 + eta[idx])
            * np.exp(-v[idx])
        )

        GH1 = np.ones(M21, dtype=np.float32)
        idx_hi = v > 5.0
        GH1[idx_hi] = eta[idx_hi] / (1.0 + eta[idx_hi])
        idx_mid = (v > 0.0) & (v <= 5.0)
        if np.any(idx_mid):
            vv = np.maximum(v[idx_mid], 1e-8)
            GH1[idx_mid] = eta[idx_mid] / (1.0 + eta[idx_mid]) * np.exp(0.5 * exp1(vv))

        eta_2term = (GH1 ** 2) * gamma

        score_num += float(np.sum(PH1 * X2))
        score_den += float(np.sum((1.0 - PH1) * X2))

    score = score_num / (score_den + EPS)
    if not np.isfinite(score):
        return 0.0
    return float(score)


def _omlsa_residual_tf_mix_score(
    PY: np.ndarray,
    PX: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """OMLSA mix score. Higher = better.

    # OMLSA mix score:
    # measures how much enhanced energy remains speech-like (via PH1_X)
    # in regions where noisy signal predicts speech (via PH1_Y)

    Score = Σ PH1_Y · PH1_X · X² / (Σ PH1_Y · X² + ε)

    PH1_Y: speech-presence probability from IMCRA run on PY (noisy).
    PH1_X: speech-presence probability from IMCRA run on PX (enhanced).
    X²   : enhanced power spectrum.
    """
    import numpy as np
    from scipy.special import exp1

    EPS = eps

    M21 = min(PY.shape[0], PX.shape[0])
    n_frames = min(PY.shape[1], PX.shape[1])
    PY = np.asarray(PY[:M21, :n_frames], dtype=np.float32)
    PX = np.asarray(PX[:M21, :n_frames], dtype=np.float32)

    if n_frames < 2:
        return 0.0

    M  = 2 * (M21 - 1)
    Fs = 16000.0

    # ===== Parameters — identical to _omlsa_gating_score =====
    w           = 1
    alpha_s     = 0.9
    alpha_d     = 0.70
    alpha_eta   = 0.95
    alpha_xi    = 0.7
    Nwin        = 6
    Vwin        = 8
    delta_s     = 1.45
    Bmin        = 1.66
    delta_y     = 3.2
    delta_yt    = 2.2
    w_xi_local  = 1
    w_xi_global = 9
    f_u         = 8000.0
    f_l         = 80.0
    P_min       = 0.02
    xi_lu_dB    = -4.0
    xi_ll_dB    = -9.0
    xi_gu_dB    = -4.0
    xi_gl_dB    = -9.0
    xi_fu_dB    = -4.0
    xi_fl_dB    = -9.0
    xi_mu_dB    = 8.0
    xi_ml_dB    = -1.0
    q_max       = 0.992
    eta_min     = 10.0 ** (-16.0 / 10.0)

    b = np.hanning(2 * w + 1).astype(np.float32)
    b /= np.sum(b)
    b_xi_local = np.hanning(2 * w_xi_local + 1).astype(np.float32)
    b_xi_local /= np.sum(b_xi_local)
    b_xi_global = np.hanning(2 * w_xi_global + 1).astype(np.float32)
    b_xi_global /= np.sum(b_xi_global)

    k_u = min(int(round(f_u / Fs * M + 1)), M21)
    k_l = max(1, min(int(round(f_l / Fs * M + 1)), M21 - 1))
    k2_local = max(1, min(int(round(500.0  / Fs * M + 1)), M21 - 1))
    k3_local = max(k2_local + 1, min(int(round(3500.0 / Fs * M + 1)), M21 - 1))

    def _conv_same(x: np.ndarray, h: np.ndarray) -> np.ndarray:
        return np.convolve(x, h, mode="same")

    def _db_prob(x_db: np.ndarray, lo: float, hi: float, pmin: float) -> np.ndarray:
        out = np.ones_like(x_db, dtype=np.float32)
        out[x_db <= lo] = pmin
        mid = (x_db > lo) & (x_db < hi)
        out[mid] = pmin + (x_db[mid] - lo) / (hi - lo) * (1.0 - pmin)
        return out

    def _run_imcra(P: np.ndarray) -> list:
        """Run IMCRA backbone on P; return per-frame PH1 arrays."""
        eta_2term = np.ones(M21, dtype=np.float32)
        xi = np.zeros(M21, dtype=np.float32)
        xi_frame = 0.0
        xi_m_dB = xi_ml_dB
        l_mod_lswitch = 0

        lambda_d   = np.maximum(P[:, 0].copy(), EPS)
        Sf0        = _conv_same(P[:, 0], b)
        S          = Sf0.copy()
        St         = Sf0.copy()
        lambda_dav = P[:, 0].copy()
        Smin       = S.copy()
        SMact      = S.copy()
        Smint      = St.copy()
        SMactt     = St.copy()
        SW         = np.tile(S[:,  None], (1, Nwin))
        SWt        = np.tile(St[:, None], (1, Nwin))

        PH1_frames = []

        for l in range(n_frames):
            Ya2 = np.maximum(P[:, l], EPS)

            gamma = Ya2 / np.maximum(lambda_d, EPS)
            eta   = alpha_eta * eta_2term + (1.0 - alpha_eta) * np.maximum(gamma - 1.0, 0.0)
            eta   = np.maximum(eta, eta_min)
            v     = gamma * eta / (1.0 + eta)

            Sf = _conv_same(Ya2, b)

            if l == 0:
                S = Sf.copy(); St = Sf.copy()
                Smin = S.copy(); SMact = S.copy()
                Smint = St.copy(); SMactt = St.copy()
                lambda_dav = Ya2.copy()
            else:
                S = alpha_s * S + (1.0 - alpha_s) * Sf
                if l < 14:
                    Smin = S.copy(); SMact = S.copy()
                else:
                    Smin  = np.minimum(Smin,  S)
                    SMact = np.minimum(SMact, S)

            I_f    = ((Ya2 < delta_y * Bmin * Smin) & (S < delta_s * Bmin * Smin)).astype(np.float32)
            conv_I = _conv_same(I_f, b)
            Sft    = St.copy()
            idx    = conv_I > 0
            if np.any(idx):
                conv_Y  = _conv_same(I_f * Ya2, b)
                Sft[idx] = conv_Y[idx] / np.maximum(conv_I[idx], EPS)

            if l < 14:
                St = S.copy(); Smint = St.copy(); SMactt = St.copy()
            else:
                St     = alpha_s * St + (1.0 - alpha_s) * Sft
                Smint  = np.minimum(Smint,  St)
                SMactt = np.minimum(SMactt, St)

            qhat = np.ones(M21, dtype=np.float32)
            phat = np.zeros(M21, dtype=np.float32)
            gamma_mint = Ya2 / (Bmin * np.maximum(Smint, EPS))
            zetat      = S   / (Bmin * np.maximum(Smint, EPS))
            idx = (gamma_mint > 1.0) & (gamma_mint < delta_yt) & (zetat < delta_s)
            qhat[idx] = (delta_yt - gamma_mint[idx]) / (delta_yt - 1.0)
            phat[idx] = 1.0 / (
                1.0 + qhat[idx] / np.maximum(1.0 - qhat[idx], EPS)
                * (1.0 + eta[idx]) * np.exp(-v[idx])
            )
            phat[(gamma_mint >= delta_yt) | (zetat >= delta_s)] = 1.0

            alpha_dt   = alpha_d + (1.0 - alpha_d) * phat
            lambda_dav = alpha_dt * lambda_dav + (1.0 - alpha_dt) * Ya2

            l_mod_lswitch += 1
            if l_mod_lswitch == Vwin:
                l_mod_lswitch = 0
                if l == Vwin - 1:
                    SW  = np.tile(S[:,  None], (1, Nwin))
                    SWt = np.tile(St[:, None], (1, Nwin))
                else:
                    SW   = np.concatenate([SW[:,  1:], SMact[:,  None]], axis=1)
                    Smin = np.min(SW, axis=1); SMact = S.copy()
                    SWt   = np.concatenate([SWt[:, 1:], SMactt[:, None]], axis=1)
                    Smint = np.min(SWt, axis=1); SMactt = St.copy()

            lambda_d = np.maximum(2.0 * lambda_dav, EPS)

            xi        = alpha_xi * xi + (1.0 - alpha_xi) * eta
            xi_local  = _conv_same(xi, b_xi_local)
            xi_global = _conv_same(xi, b_xi_global)

            prev_xi_frame = xi_frame
            xi_frame  = float(np.mean(xi[k_l:k_u]))
            dxi_frame = xi_frame - prev_xi_frame

            xi_local_dB  = 10.0 * np.log10(np.maximum(xi_local,  1e-10))
            xi_global_dB = 10.0 * np.log10(np.maximum(xi_global, 1e-10))
            xi_frame_dB  = 10.0 * np.log10(max(xi_frame, 1e-10))

            P_local  = _db_prob(xi_local_dB,  xi_ll_dB, xi_lu_dB, P_min)
            P_global = _db_prob(xi_global_dB, xi_gl_dB, xi_gu_dB, P_min)

            lo = min(3, M21 - 1)
            hi = min(k2_local + k3_local - 3, M21)
            m_P_local = float(np.mean(P_local[lo:hi])) if hi > lo else float(np.mean(P_local))
            if m_P_local < 0.25:
                P_local[k2_local:k3_local] = P_min

            if xi_frame_dB <= xi_fl_dB:
                P_frame = P_min
            elif dxi_frame >= 0:
                xi_m_dB = min(max(xi_frame_dB, xi_ml_dB), xi_mu_dB)
                P_frame = 1.0
            elif xi_frame_dB >= xi_m_dB + xi_fu_dB:
                P_frame = 1.0
            elif xi_frame_dB <= xi_m_dB + xi_fl_dB:
                P_frame = P_min
            else:
                P_frame = P_min + (
                    (xi_frame_dB - xi_m_dB - xi_fl_dB) / (xi_fu_dB - xi_fl_dB)
                ) * (1.0 - P_min)

            q = np.minimum(1.0 - P_global * P_local * P_frame, q_max)

            gamma = Ya2 / np.maximum(lambda_d, EPS)
            eta   = alpha_eta * eta_2term + (1.0 - alpha_eta) * np.maximum(gamma - 1.0, 0.0)
            eta   = np.maximum(eta, eta_min)
            v     = gamma * eta / (1.0 + eta)

            PH1 = np.zeros(M21, dtype=np.float32)
            idx = q < 0.9
            PH1[idx] = 1.0 / (
                1.0 + q[idx] / np.maximum(1.0 - q[idx], EPS)
                * (1.0 + eta[idx]) * np.exp(-v[idx])
            )

            GH1 = np.ones(M21, dtype=np.float32)
            idx_hi = v > 5.0
            GH1[idx_hi] = eta[idx_hi] / (1.0 + eta[idx_hi])
            idx_mid = (v > 0.0) & (v <= 5.0)
            if np.any(idx_mid):
                vv = np.maximum(v[idx_mid], 1e-8)
                GH1[idx_mid] = eta[idx_mid] / (1.0 + eta[idx_mid]) * np.exp(0.5 * exp1(vv))

            eta_2term = (GH1 ** 2) * gamma
            PH1_frames.append(PH1)

        return PH1_frames

    # Run IMCRA independently on noisy and enhanced spectra
    PH1_Y_frames = _run_imcra(PY)
    PH1_X_frames = _run_imcra(PX)

    score_num = 0.0
    score_den = 0.0
    for l in range(n_frames):
        X2    = np.maximum(PX[:, l], 0.0)
        PH1_Y = PH1_Y_frames[l]
        PH1_X = PH1_X_frames[l]
        score_num += float(np.sum(PH1_Y * PH1_X * X2))
        score_den += float(np.sum(PH1_Y * X2))

    score = score_num / (score_den + EPS)
    if not np.isfinite(score):
        return 0.0
    return float(score)


def _ph1_frames_from_spectrum(P: np.ndarray, eps: float = 1e-10) -> list:
    """Run IMCRA/OMLSA backbone on P [F, T]; return per-frame PH1 list.

    Parameters identical to _omlsa_residual_tf_mix_score._run_imcra.
    New functions should call this instead of duplicating the backbone.
    """
    import numpy as np
    from scipy.special import exp1

    EPS = eps
    M21 = P.shape[0]
    n_frames = P.shape[1]
    M  = 2 * (M21 - 1)
    Fs = 16000.0

    w           = 1
    alpha_s     = 0.9
    alpha_d     = 0.70
    alpha_eta   = 0.95
    alpha_xi    = 0.7
    Nwin        = 6
    Vwin        = 8
    delta_s     = 1.45
    Bmin        = 1.66
    delta_y     = 3.2
    delta_yt    = 2.2
    w_xi_local  = 1
    w_xi_global = 9
    f_u         = 8000.0
    f_l         = 80.0
    P_min       = 0.02
    xi_lu_dB    = -4.0
    xi_ll_dB    = -9.0
    xi_gu_dB    = -4.0
    xi_gl_dB    = -9.0
    xi_fu_dB    = -4.0
    xi_fl_dB    = -9.0
    xi_mu_dB    = 8.0
    xi_ml_dB    = -1.0
    q_max       = 0.992
    eta_min     = 10.0 ** (-16.0 / 10.0)

    b = np.hanning(2 * w + 1).astype(np.float32);           b /= np.sum(b)
    b_xi_local  = np.hanning(2 * w_xi_local  + 1).astype(np.float32); b_xi_local  /= np.sum(b_xi_local)
    b_xi_global = np.hanning(2 * w_xi_global + 1).astype(np.float32); b_xi_global /= np.sum(b_xi_global)

    k_u      = min(int(round(f_u    / Fs * M + 1)), M21)
    k_l      = max(1, min(int(round(f_l    / Fs * M + 1)), M21 - 1))
    k2_local = max(1, min(int(round(500.0  / Fs * M + 1)), M21 - 1))
    k3_local = max(k2_local + 1, min(int(round(3500.0 / Fs * M + 1)), M21 - 1))

    def _conv_same(x, h):
        return np.convolve(x, h, mode="same")

    def _db_prob(x_db, lo, hi, pmin):
        out = np.ones_like(x_db, dtype=np.float32)
        out[x_db <= lo] = pmin
        mid = (x_db > lo) & (x_db < hi)
        out[mid] = pmin + (x_db[mid] - lo) / (hi - lo) * (1.0 - pmin)
        return out

    eta_2term     = np.ones(M21, dtype=np.float32)
    xi            = np.zeros(M21, dtype=np.float32)
    xi_frame      = 0.0
    xi_m_dB       = xi_ml_dB
    l_mod_lswitch = 0

    lambda_d   = np.maximum(P[:, 0].copy(), EPS)
    Sf0        = _conv_same(P[:, 0], b)
    S          = Sf0.copy()
    St         = Sf0.copy()
    lambda_dav = P[:, 0].copy()
    Smin       = S.copy()
    SMact      = S.copy()
    Smint      = St.copy()
    SMactt     = St.copy()
    SW         = np.tile(S[:,  None], (1, Nwin))
    SWt        = np.tile(St[:, None], (1, Nwin))

    PH1_frames = []
    for l in range(n_frames):
        Ya2   = np.maximum(P[:, l], EPS)
        gamma = Ya2 / np.maximum(lambda_d, EPS)
        eta   = np.maximum(alpha_eta * eta_2term + (1.0 - alpha_eta) * np.maximum(gamma - 1.0, 0.0), eta_min)
        v     = gamma * eta / (1.0 + eta)
        Sf    = _conv_same(Ya2, b)

        if l == 0:
            S = Sf.copy(); St = Sf.copy()
            Smin = S.copy(); SMact = S.copy()
            Smint = St.copy(); SMactt = St.copy()
            lambda_dav = Ya2.copy()
        else:
            S = alpha_s * S + (1.0 - alpha_s) * Sf
            if l < 14:
                Smin = S.copy(); SMact = S.copy()
            else:
                Smin  = np.minimum(Smin,  S)
                SMact = np.minimum(SMact, S)

        I_f    = ((Ya2 < delta_y * Bmin * Smin) & (S < delta_s * Bmin * Smin)).astype(np.float32)
        conv_I = _conv_same(I_f, b)
        Sft    = St.copy()
        idx    = conv_I > 0
        if np.any(idx):
            conv_Y   = _conv_same(I_f * Ya2, b)
            Sft[idx] = conv_Y[idx] / np.maximum(conv_I[idx], EPS)

        if l < 14:
            St = S.copy(); Smint = St.copy(); SMactt = St.copy()
        else:
            St     = alpha_s * St + (1.0 - alpha_s) * Sft
            Smint  = np.minimum(Smint,  St)
            SMactt = np.minimum(SMactt, St)

        qhat = np.ones(M21, dtype=np.float32)
        phat = np.zeros(M21, dtype=np.float32)
        gamma_mint = Ya2 / (Bmin * np.maximum(Smint, EPS))
        zetat      = S   / (Bmin * np.maximum(Smint, EPS))
        idx = (gamma_mint > 1.0) & (gamma_mint < delta_yt) & (zetat < delta_s)
        qhat[idx] = (delta_yt - gamma_mint[idx]) / (delta_yt - 1.0)
        phat[idx] = 1.0 / (
            1.0 + qhat[idx] / np.maximum(1.0 - qhat[idx], EPS)
            * (1.0 + eta[idx]) * np.exp(-v[idx])
        )
        phat[(gamma_mint >= delta_yt) | (zetat >= delta_s)] = 1.0

        alpha_dt   = alpha_d + (1.0 - alpha_d) * phat
        lambda_dav = alpha_dt * lambda_dav + (1.0 - alpha_dt) * Ya2

        l_mod_lswitch += 1
        if l_mod_lswitch == Vwin:
            l_mod_lswitch = 0
            if l == Vwin - 1:
                SW  = np.tile(S[:,  None], (1, Nwin))
                SWt = np.tile(St[:, None], (1, Nwin))
            else:
                SW   = np.concatenate([SW[:,  1:], SMact[:,  None]], axis=1)
                Smin = np.min(SW, axis=1); SMact = S.copy()
                SWt   = np.concatenate([SWt[:, 1:], SMactt[:, None]], axis=1)
                Smint = np.min(SWt, axis=1); SMactt = St.copy()

        lambda_d = np.maximum(2.0 * lambda_dav, EPS)
        xi        = alpha_xi * xi + (1.0 - alpha_xi) * eta
        xi_local  = _conv_same(xi, b_xi_local)
        xi_global = _conv_same(xi, b_xi_global)

        prev_xi_frame = xi_frame
        xi_frame      = float(np.mean(xi[k_l:k_u]))
        dxi_frame     = xi_frame - prev_xi_frame

        xi_local_dB  = 10.0 * np.log10(np.maximum(xi_local,  1e-10))
        xi_global_dB = 10.0 * np.log10(np.maximum(xi_global, 1e-10))
        xi_frame_dB  = 10.0 * np.log10(max(xi_frame, 1e-10))

        P_local  = _db_prob(xi_local_dB,  xi_ll_dB, xi_lu_dB, P_min)
        P_global = _db_prob(xi_global_dB, xi_gl_dB, xi_gu_dB, P_min)

        lo = min(3, M21 - 1)
        hi = min(k2_local + k3_local - 3, M21)
        m_P_local = float(np.mean(P_local[lo:hi])) if hi > lo else float(np.mean(P_local))
        if m_P_local < 0.25:
            P_local[k2_local:k3_local] = P_min

        if xi_frame_dB <= xi_fl_dB:
            P_frame = P_min
        elif dxi_frame >= 0:
            xi_m_dB = min(max(xi_frame_dB, xi_ml_dB), xi_mu_dB)
            P_frame = 1.0
        elif xi_frame_dB >= xi_m_dB + xi_fu_dB:
            P_frame = 1.0
        elif xi_frame_dB <= xi_m_dB + xi_fl_dB:
            P_frame = P_min
        else:
            P_frame = P_min + (
                (xi_frame_dB - xi_m_dB - xi_fl_dB) / (xi_fu_dB - xi_fl_dB)
            ) * (1.0 - P_min)

        q = np.minimum(1.0 - P_global * P_local * P_frame, q_max)

        gamma = Ya2 / np.maximum(lambda_d, EPS)
        eta   = np.maximum(alpha_eta * eta_2term + (1.0 - alpha_eta) * np.maximum(gamma - 1.0, 0.0), eta_min)
        v     = gamma * eta / (1.0 + eta)

        PH1 = np.zeros(M21, dtype=np.float32)
        idx = q < 0.9
        PH1[idx] = 1.0 / (
            1.0 + q[idx] / np.maximum(1.0 - q[idx], EPS)
            * (1.0 + eta[idx]) * np.exp(-v[idx])
        )

        GH1 = np.ones(M21, dtype=np.float32)
        idx_hi  = v > 5.0
        GH1[idx_hi] = eta[idx_hi] / (1.0 + eta[idx_hi])
        idx_mid = (v > 0.0) & (v <= 5.0)
        if np.any(idx_mid):
            vv = np.maximum(v[idx_mid], 1e-8)
            GH1[idx_mid] = eta[idx_mid] / (1.0 + eta[idx_mid]) * np.exp(0.5 * exp1(vv))

        eta_2term = (GH1 ** 2) * gamma
        PH1_frames.append(PH1)

    return PH1_frames


def _omlsa_mask_agree_score(
    PY: np.ndarray,
    PX: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """OMLSA mask agreement score. Higher = better.

    # OMLSA mask agreement score:
    # penalizes mismatch between noisy and enhanced speech masks
    # higher score = better agreement / cleaner enhancement

    score = 1 - Σ PH1_Y * |PH1_Y - PH1_X| / (Σ PH1_Y + ε)

    PH1_Y: speech-presence probability from IMCRA run on PY (noisy).
    PH1_X: speech-presence probability from IMCRA run on PX (enhanced).
    No X² weighting — pure mask comparison.
    """
    import numpy as np

    M21      = min(PY.shape[0], PX.shape[0])
    n_frames = min(PY.shape[1], PX.shape[1])
    PY = np.asarray(PY[:M21, :n_frames], dtype=np.float32)
    PX = np.asarray(PX[:M21, :n_frames], dtype=np.float32)

    if n_frames < 2:
        return 0.0

    PH1_Y_frames = _ph1_frames_from_spectrum(PY, eps=eps)
    PH1_X_frames = _ph1_frames_from_spectrum(PX, eps=eps)

    score_num = 0.0
    score_den = 0.0
    for l in range(n_frames):
        PH1_Y = PH1_Y_frames[l]
        PH1_X = PH1_X_frames[l]
        diff = np.abs(PH1_Y - PH1_X)
        score_num += float(np.sum(PH1_Y * diff))
        score_den += float(np.sum(PH1_Y))

    score = 1.0 - score_num / (score_den + eps)
    return float(np.clip(score, 0.0, 1.0))


def _omlsa_enhanced_dominant_score(
    PY: np.ndarray,
    PX: np.ndarray,
    eps: float = 1e-10,
    lambda_noise_penalty: float = 0.25,
) -> float:
    """OM-LSA enhanced-dominant score. Higher = better.

    Runs IMCRA independently on PX and PY to obtain PH1_X and PH1_Y.
    Score rewards enhanced energy that looks speech-like according to PH1_X,
    and lightly penalises enhanced energy in regions where PH1_Y says speech
    is absent.

    s = Σ PH1_X · X²  / (Σ X² + ε)
      − λ · Σ (1−PH1_Y) · X²  / (Σ X² + ε)
    """
    import numpy as np
    EPS = eps

    M21      = min(PY.shape[0], PX.shape[0])
    n_frames = min(PY.shape[1], PX.shape[1])
    PY = np.asarray(PY[:M21, :n_frames], dtype=np.float32)
    PX = np.asarray(PX[:M21, :n_frames], dtype=np.float32)

    if n_frames < 2:
        return 0.0

    PH1_X_frames = _ph1_frames_from_spectrum(PX, eps=EPS)
    PH1_Y_frames = _ph1_frames_from_spectrum(PY, eps=EPS)

    total_energy  = 0.0
    speech_like_x = 0.0
    noise_leak_y  = 0.0

    for l in range(n_frames):
        X2    = np.maximum(PX[:, l], 0.0)
        PH1_X = PH1_X_frames[l]
        PH1_Y = PH1_Y_frames[l]
        total_energy  += float(np.sum(X2))
        speech_like_x += float(np.sum(PH1_X * X2))
        noise_leak_y  += float(np.sum((1.0 - PH1_Y) * X2))

    denom = total_energy + EPS
    score = speech_like_x / denom - lambda_noise_penalty * noise_leak_y / denom
    if not np.isfinite(score):
        return 0.0
    return float(score)


def _omlsa_residual_consistency_score(
    PY: np.ndarray,
    PX: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """Residual consistency score. Higher = better.

    # Residual consistency score:
    # penalizes removing speech (via residual energy in speech regions)
    # and leaving noise (via energy in non-speech regions)
    # approximates SI-SDR selection behavior

    Uses magnitude approximation R_mag = sqrt(PY) - sqrt(PX) since only
    power spectra are available (no complex phase).

    bad = speech_term + noise_term
        speech_term = Σ PH1_Y · R²  / (Σ PH1_Y · Y²  + ε)
        noise_term  = Σ (1-PH1_Y) · X²  / (Σ (1-PH1_Y) · Y²  + ε)
    score = -bad
    """
    import numpy as np

    M21      = min(PY.shape[0], PX.shape[0])
    n_frames = min(PY.shape[1], PX.shape[1])
    PY = np.asarray(PY[:M21, :n_frames], dtype=np.float32)
    PX = np.asarray(PX[:M21, :n_frames], dtype=np.float32)

    if n_frames < 2:
        return 0.0

    PH1_Y_frames = _ph1_frames_from_spectrum(PY, eps=eps)

    speech_num = 0.0
    speech_den = 0.0
    noise_num  = 0.0
    noise_den  = 0.0

    for l in range(n_frames):
        Y2    = PY[:, l]
        X2    = np.maximum(PX[:, l], 0.0)
        R2    = (np.sqrt(np.maximum(Y2, 0.0)) - np.sqrt(X2)) ** 2
        PH1_Y = PH1_Y_frames[l]
        w_s   = PH1_Y
        w_n   = 1.0 - PH1_Y

        speech_num += float(np.sum(w_s * R2))
        speech_den += float(np.sum(w_s * Y2))
        noise_num  += float(np.sum(w_n * X2))
        noise_den  += float(np.sum(w_n * Y2))

    speech_term = speech_num / (speech_den + eps)
    noise_term  = noise_num  / (noise_den  + eps)
    score = -(speech_term + noise_term)

    if not np.isfinite(score):
        return 0.0
    return float(score)


def gate_step_omlsa_residual_tf(step_info: dict, cache: dict) -> float:
    """Per-step OMLSA-TF gate — pure TF-domain, no xt_mean→waveform conversion.

    Uses the noisy waveform y_np (STFTed once and cached) for IMCRA noise
    tracking, and |xt_mean|² directly as the enhanced power spectrum.  This
    gives the same quality of noise estimate as the post-hoc _omlsa_residual_score
    while keeping xt_mean entirely in the TF domain.

    The noisy STFT is computed with the model's standard parameters:
        n_fft  = 2 * (F - 1)   inferred from xt_mean.shape[2]
        hop    = 128
        window = Hann, center=True
    These match the SGMSE/VoiceBank data module defaults (n_fft=510, hop=128).

    step_info keys (used):
        xt_mean : torch.Tensor [B, C, F, T_spec] — denoised latent estimate

    cache keys (required):
        y_np    : np.ndarray [T] — normalised noisy waveform (same as
                  gate_step_omlsa_residual; set by enhancement.py)

    cache keys (optional):
        eps              : float  (default 1e-10)
        _py_tf_cache     : np.ndarray [F, T_spec] — cached PY; computed on
                           first call and reused for subsequent steps of the
                           same utterance.

    Returns:
        float: gate score. Lower = better quality.
    """
    import librosa

    xt_mean = step_info["xt_mean"]                               # [B, C, F, T_spec]
    eps     = cache.get("eps", 1e-10)

    # --- PX: sum power over channels, stay in TF domain ---
    PX = (xt_mean[0].abs() ** 2).sum(dim=0).detach().cpu().numpy()  # [F, T_spec]
    F_model = PX.shape[0]

    # --- PY: prefer model-domain noisy power spectrum (zero mismatch) ---
    if "y_PY" in cache:
        # Option A (preferred): enhancement.py stored |Y[0]|².sum(channels) directly.
        # Exact same TF grid as xt_mean — no STFT convention mismatch possible.
        PY = cache["y_PY"]                                       # [F_model, T_spec]
    else:
        # Fallback: STFT y_np with model-matching parameters (cached after first call).
        if "_py_tf_cache" not in cache:
            y_np  = cache["y_np"]                               # [T] waveform
            n_fft = 2 * (F_model - 1)                          # e.g. 510 for F=256
            hop   = 128
            win   = np.hanning(n_fft).astype(np.float32)
            win   = win / (win ** 2).sum() ** 0.5              # unit-energy normalisation
            Y_stft = librosa.stft(
                np.asarray(y_np, dtype=np.float32),
                n_fft=n_fft, hop_length=hop, win_length=n_fft,
                window=win, center=True,
            )                                                   # [F_model, T_stft]
            cache["_py_tf_cache"] = np.abs(Y_stft) ** 2       # [F_model, T_stft]
        PY = cache["_py_tf_cache"]

    return _omlsa_residual_tf_score(PY, PX, eps=eps)



def gate_step_omlsa_gating(step_info: dict, cache: dict) -> float:
    """Per-step simplified-v2 OMLSA-TF gate. Same TF-domain setup as
    gate_step_omlsa_residual_tf; delegates scoring to
    _omlsa_gating_score.
    """
    import librosa

    xt_mean = step_info["xt_mean"]
    eps     = cache.get("eps", 1e-10)

    PX = (xt_mean[0].abs() ** 2).sum(dim=0).detach().cpu().numpy()
    F_model = PX.shape[0]

    if "y_PY" in cache:
        PY = cache["y_PY"]
    else:
        if "_py_tf_cache" not in cache:
            y_np  = cache["y_np"]
            n_fft = 2 * (F_model - 1)
            hop   = 128
            win   = np.hanning(n_fft).astype(np.float32)
            win   = win / (win ** 2).sum() ** 0.5
            Y_stft = librosa.stft(
                np.asarray(y_np, dtype=np.float32),
                n_fft=n_fft, hop_length=hop, win_length=n_fft,
                window=win, center=True,
            )
            cache["_py_tf_cache"] = np.abs(Y_stft) ** 2
        PY = cache["_py_tf_cache"]

    return _omlsa_gating_score(PY, PX, eps=eps)


def gate_step_omlsa_mix(step_info: dict, cache: dict) -> float:
    """Per-step OMLSA-mix gate. Same TF-domain setup as gate_step_omlsa_gating;
    delegates scoring to _omlsa_residual_tf_mix_score. Higher = better.
    """
    import librosa

    xt_mean = step_info["xt_mean"]
    eps     = cache.get("eps", 1e-10)

    PX = (xt_mean[0].abs() ** 2).sum(dim=0).detach().cpu().numpy()
    F_model = PX.shape[0]

    if "y_PY" in cache:
        PY = cache["y_PY"]
    else:
        if "_py_tf_cache" not in cache:
            y_np  = cache["y_np"]
            n_fft = 2 * (F_model - 1)
            hop   = 128
            win   = np.hanning(n_fft).astype(np.float32)
            win   = win / (win ** 2).sum() ** 0.5
            Y_stft = librosa.stft(
                np.asarray(y_np, dtype=np.float32),
                n_fft=n_fft, hop_length=hop, win_length=n_fft,
                window=win, center=True,
            )
            cache["_py_tf_cache"] = np.abs(Y_stft) ** 2
        PY = cache["_py_tf_cache"]

    return _omlsa_residual_tf_mix_score(PY, PX, eps=eps)


def gate_step_omlsa_mask_agree(step_info: dict, cache: dict) -> float:
    """Per-step OMLSA mask-agreement gate. Higher = better.
    Delegates scoring to _omlsa_mask_agree_score.
    """
    import librosa

    xt_mean = step_info["xt_mean"]
    eps     = cache.get("eps", 1e-10)

    PX = (xt_mean[0].abs() ** 2).sum(dim=0).detach().cpu().numpy()
    F_model = PX.shape[0]

    if "y_PY" in cache:
        PY = cache["y_PY"]
    else:
        if "_py_tf_cache" not in cache:
            y_np  = cache["y_np"]
            n_fft = 2 * (F_model - 1)
            hop   = 128
            win   = np.hanning(n_fft).astype(np.float32)
            win   = win / (win ** 2).sum() ** 0.5
            Y_stft = librosa.stft(
                np.asarray(y_np, dtype=np.float32),
                n_fft=n_fft, hop_length=hop, win_length=n_fft,
                window=win, center=True,
            )
            cache["_py_tf_cache"] = np.abs(Y_stft) ** 2
        PY = cache["_py_tf_cache"]

    return _omlsa_mask_agree_score(PY, PX, eps=eps)


def gate_step_omlsa_enhanced_dominant(step_info: dict, cache: dict) -> float:
    """Per-step enhanced-dominant OMLSA gate. Higher = better.
    Delegates scoring to _omlsa_enhanced_dominant_score.
    """
    import librosa

    xt_mean = step_info["xt_mean"]
    eps     = cache.get("eps", 1e-10)

    PX = (xt_mean[0].abs() ** 2).sum(dim=0).detach().cpu().numpy()
    F_model = PX.shape[0]

    if "y_PY" in cache:
        PY = cache["y_PY"]
    else:
        if "_py_tf_cache" not in cache:
            y_np  = cache["y_np"]
            n_fft = 2 * (F_model - 1)
            hop   = 128
            win   = np.hanning(n_fft).astype(np.float32)
            win   = win / (win ** 2).sum() ** 0.5
            Y_stft = librosa.stft(
                np.asarray(y_np, dtype=np.float32),
                n_fft=n_fft, hop_length=hop, win_length=n_fft,
                window=win, center=True,
            )
            cache["_py_tf_cache"] = np.abs(Y_stft) ** 2
        PY = cache["_py_tf_cache"]

    return _omlsa_enhanced_dominant_score(PY, PX, eps=eps)


def gate_step_omlsa_enhanced_total_dominant(step_info: dict, cache: dict) -> float:
    """Per-step enhanced-dominant OMLSA gate, lambda_noise_penalty=0. Higher = better.
    Identical to omlsa_enhanced_dominant but with no noise-region penalty term.
    """
    import librosa

    xt_mean = step_info["xt_mean"]
    eps     = cache.get("eps", 1e-10)

    PX = (xt_mean[0].abs() ** 2).sum(dim=0).detach().cpu().numpy()
    F_model = PX.shape[0]

    if "y_PY" in cache:
        PY = cache["y_PY"]
    else:
        if "_py_tf_cache" not in cache:
            y_np  = cache["y_np"]
            n_fft = 2 * (F_model - 1)
            hop   = 128
            win   = np.hanning(n_fft).astype(np.float32)
            win   = win / (win ** 2).sum() ** 0.5
            Y_stft = librosa.stft(
                np.asarray(y_np, dtype=np.float32),
                n_fft=n_fft, hop_length=hop, win_length=n_fft,
                window=win, center=True,
            )
            cache["_py_tf_cache"] = np.abs(Y_stft) ** 2
        PY = cache["_py_tf_cache"]

    return _omlsa_enhanced_dominant_score(PY, PX, eps=eps, lambda_noise_penalty=0.0)


def gate_step_omlsa_residual_consistency(step_info: dict, cache: dict) -> float:
    """Per-step residual-consistency gate. Higher = better.
    Delegates scoring to _omlsa_residual_consistency_score.
    """
    import librosa

    xt_mean = step_info["xt_mean"]
    eps     = cache.get("eps", 1e-10)

    PX = (xt_mean[0].abs() ** 2).sum(dim=0).detach().cpu().numpy()
    F_model = PX.shape[0]

    if "y_PY" in cache:
        PY = cache["y_PY"]
    else:
        if "_py_tf_cache" not in cache:
            y_np  = cache["y_np"]
            n_fft = 2 * (F_model - 1)
            hop   = 128
            win   = np.hanning(n_fft).astype(np.float32)
            win   = win / (win ** 2).sum() ** 0.5
            Y_stft = librosa.stft(
                np.asarray(y_np, dtype=np.float32),
                n_fft=n_fft, hop_length=hop, win_length=n_fft,
                window=win, center=True,
            )
            cache["_py_tf_cache"] = np.abs(Y_stft) ** 2
        PY = cache["_py_tf_cache"]

    return _omlsa_residual_consistency_score(PY, PX, eps=eps)


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


def gate_step_traj_jump(step_info: dict, cache: dict) -> float:
    """Per-step trajectory-instability gate — speech-agnostic.

    Measures the relative squared displacement of consecutive denoised latent
    estimates (xt_mean).  A large jump indicates an unstable diffusion trajectory.

    Formula (higher = worse, same convention as other per-step gates):

        score = ||cur - prev||^2 / (||prev||^2 + eps)

    where cur and prev are flattened float32 views of xt_mean[0] (batch item 0).

    Cache keys read:
        eps                        : float (default 1e-8)

    Cache keys written (internal state — do NOT set externally):
        _traj_jump_prev_xt_mean    : np.ndarray | None — previous step's xt_mean
        _traj_jump_prev_step_idx   : int | None        — previous step_idx

    Cache reset across attempts:
        The cache dict is shared across all attempts of the same utterance.
        A new attempt is detected when step_idx is non-monotonic
        (step_idx <= _traj_jump_prev_step_idx), in which case stored state is
        cleared and 0.0 is returned for that first step.

    step_info keys (used):
        step_idx : int           — diffusion step index
        xt_mean  : torch.Tensor  — denoised estimate  [B, C, F, T_spec]

    Returns:
        float: relative squared displacement (higher = more unstable = worse).
               0.0 on the first usable step of each attempt.
    """
    cur_tensor  = step_info["xt_mean"]          # [B, C, F, T_spec]
    step_idx    = step_info["step_idx"]
    eps         = cache.get("eps", 1e-8)

    prev_tensor  = cache.get("_traj_jump_prev_xt_mean", None)
    prev_step    = cache.get("_traj_jump_prev_step_idx", None)

    # Detect attempt reset: step_idx is non-monotonic → new attempt started.
    if prev_step is not None and step_idx <= prev_step:
        prev_tensor = None
        cache.pop("_traj_jump_max_score", None)
        cache.pop("_traj_jump_max_step",  None)

    # Keep xt_mean as complex: a single .ravel() gives a 1-D complex128/64 vector.
    # np.vdot(a, b) = Σ conj(a_i)*b_i, so vdot(x, x).real = Σ |x_i|^2,
    # which is the correct Hermitian (complex Euclidean) squared norm.
    cur = cur_tensor[0].detach().cpu().numpy().ravel()

    if prev_tensor is None:
        # First usable step of this attempt: seed the cache and return 0.0.
        cache["_traj_jump_prev_xt_mean"]  = cur
        cache["_traj_jump_prev_step_idx"] = step_idx
        return 0.0

    prev = prev_tensor

    diff  = cur - prev
    score = float(np.vdot(diff, diff).real) / (float(np.vdot(prev, prev).real) + eps)

    cache["_traj_jump_prev_xt_mean"]  = cur
    cache["_traj_jump_prev_step_idx"] = step_idx

    # Track running max for this attempt (read back by the caller after sampling).
    if score > cache.get("_traj_jump_max_score", -float("inf")):
        cache["_traj_jump_max_score"] = score
        cache["_traj_jump_max_step"]  = step_idx

    return score


def gate_step_traj_curvature(step_info: dict, cache: dict) -> float:
    """Per-step trajectory-curvature gate — speech-agnostic.

    Measures the second difference of consecutive denoised latent estimates
    (xt_mean), i.e. the discrete curvature of the diffusion trajectory.  A large
    curvature indicates the sampler is bending sharply, which is a sign of
    instability.

    Formula (higher = worse, same convention as other per-step gates):

        score = ||cur - 2*prev1 + prev2||^2 / (||prev1||^2 + eps)

    where cur, prev1, prev2 are flattened float32 views of xt_mean[0]:
        cur   — this step's denoised estimate
        prev1 — one step ago
        prev2 — two steps ago

    The first two usable steps of each attempt return 0.0 while the history
    is being seeded.

    Cache keys written (internal state — do NOT set externally):
        _traj_curv_prev1         : np.ndarray | None — xt_mean from one step ago
        _traj_curv_prev2         : np.ndarray | None — xt_mean from two steps ago
        _traj_curv_prev_step_idx : int | None        — previous step_idx

    Cache reset across attempts:
        Same monotonicity check as gate_step_traj_jump: if the incoming step_idx
        is <= the last recorded step_idx, both history buffers are cleared and
        the function seeds fresh state for the new attempt.

    step_info keys (used):
        step_idx : int           — diffusion step index (always present)
        xt_mean  : torch.Tensor  — denoised estimate  [B, C, F, T_spec]

    Returns:
        float: normalised squared second difference (higher = more curved = worse).
               0.0 on the first two usable steps of each attempt.
    """
    cur_tensor = step_info["xt_mean"]          # [B, C, F, T_spec]
    step_idx   = step_info["step_idx"]
    eps        = cache.get("eps", 1e-8)

    prev1     = cache.get("_traj_curv_prev1", None)          # one step ago
    prev2     = cache.get("_traj_curv_prev2", None)          # two steps ago
    prev_step = cache.get("_traj_curv_prev_step_idx", None)

    # Detect attempt reset: non-monotonic step_idx means a new attempt started.
    if prev_step is not None and step_idx <= prev_step:
        prev1 = None
        prev2 = None
        cache.pop("_traj_curv_max_score", None)
        cache.pop("_traj_curv_max_step",  None)

    # Keep xt_mean as complex; use np.vdot for correct Hermitian squared norms
    # (same pattern as gate_step_traj_jump).
    cur = cur_tensor[0].detach().cpu().numpy().ravel()

    if prev1 is None:
        # First step of this attempt: seed prev1, no history yet.
        cache["_traj_curv_prev1"]          = cur
        cache["_traj_curv_prev2"]          = None
        cache["_traj_curv_prev_step_idx"]  = step_idx
        return 0.0

    if prev2 is None:
        # Second step: shift prev1 → prev2, store cur as prev1.
        cache["_traj_curv_prev2"]          = prev1
        cache["_traj_curv_prev1"]          = cur
        cache["_traj_curv_prev_step_idx"]  = step_idx
        return 0.0

    # Third+ step: we have a full three-point history; compute curvature.
    second_diff = cur - 2.0 * prev1 + prev2
    numer = float(np.vdot(second_diff, second_diff).real)
    denom = float(np.vdot(prev1, prev1).real) + eps
    score = numer / denom

    # Shift history forward.
    cache["_traj_curv_prev2"]          = prev1
    cache["_traj_curv_prev1"]          = cur
    cache["_traj_curv_prev_step_idx"]  = step_idx

    # Track running max for this attempt (read back by the caller after sampling).
    if score > cache.get("_traj_curv_max_score", -float("inf")):
        cache["_traj_curv_max_score"] = score
        cache["_traj_curv_max_step"]  = step_idx

    return score


def gate_step_pred_jump(step_info: dict, cache: dict) -> float:
    """Per-step model-prediction-jump gate — speech-agnostic.

    Measures the relative squared displacement between the model prediction
    (score) at consecutive diffusion steps.  Uses step_info["model_pred"],
    the raw network score ∇_x log p(x_t | y) exposed by the PC sampler.

    Formula (higher = worse, same convention as other per-step gates):

        score = ||cur - prev||^2 / (||prev||^2 + eps)

    where cur and prev are flattened complex numpy arrays of model_pred[0]
    (batch item 0).  Norms are computed via np.vdot for the correct Hermitian
    (complex Euclidean) squared norm, identical to gate_step_traj_jump.

    Returns 0.0 if step_info["model_pred"] is None (e.g. NonePredictor).

    Cache keys read:
        eps                      : float (default 1e-8)

    Cache keys written (internal state — do NOT set externally):
        _pred_jump_prev_pred     : np.ndarray | None — previous step's model_pred
        _pred_jump_prev_step_idx : int | None        — previous step_idx

    Cache reset across attempts:
        A new attempt is detected when step_idx is non-monotonic
        (step_idx <= _pred_jump_prev_step_idx), in which case stored state is
        cleared and 0.0 is returned for that first step.

    step_info keys (used):
        step_idx   : int                    — diffusion step index
        model_pred : torch.Tensor | None    — network score [B, C, F, T_spec]

    Returns:
        float: relative squared displacement of model prediction (higher = worse).
               0.0 on the first usable step of each attempt, or if model_pred is None.
    """
    model_pred = step_info.get("model_pred", None)
    if model_pred is None:
        return 0.0

    step_idx = step_info["step_idx"]
    eps      = cache.get("eps", 1e-8)

    prev_pred = cache.get("_pred_jump_prev_pred", None)
    prev_step = cache.get("_pred_jump_prev_step_idx", None)

    # Detect attempt reset: step_idx is non-monotonic → new attempt started.
    if prev_step is not None and step_idx <= prev_step:
        prev_pred = None
        cache.pop("_pred_jump_max_score", None)
        cache.pop("_pred_jump_max_step",  None)

    # Flatten batch item 0 to a 1-D complex array.
    # np.vdot(a, b) = Σ conj(a_i)*b_i, so vdot(x, x).real = Σ |x_i|^2,
    # which is the correct Hermitian squared norm for complex tensors.
    cur = model_pred[0].detach().cpu().numpy().ravel()

    if prev_pred is None:
        # First usable step of this attempt: seed the cache and return 0.0.
        cache["_pred_jump_prev_pred"]     = cur
        cache["_pred_jump_prev_step_idx"] = step_idx
        return 0.0

    prev = prev_pred

    diff  = cur - prev
    score = float(np.vdot(diff, diff).real) / (float(np.vdot(prev, prev).real) + eps)

    cache["_pred_jump_prev_pred"]     = cur
    cache["_pred_jump_prev_step_idx"] = step_idx

    # Track running max for this attempt (read back by the caller after sampling).
    if score > cache.get("_pred_jump_max_score", -float("inf")):
        cache["_pred_jump_max_score"] = score
        cache["_pred_jump_max_step"]  = step_idx

    return score


def compute_gate_scores_per_step(step_info: dict, cache: dict, gates: list) -> dict:
    """Return {gate_name: score} for each requested gate at this diffusion step.

    Supported gates:
        "leakage"              — non-speech-to-speech frame-power ratio (reference-free).
        "wiener_residual"      — Wiener-like residual-to-speech-excess energy ratio over
                                 speech frames; reference-free SI-SDR proxy.
                                 Higher = more noise leaking through = worse.
        "omlsa_residual"       — OMLSA-inspired post-hoc gate converted to waveform via
                                 model.to_audio; requires model/T_orig/norm_factor/y_np
                                 in cache.  Higher = worse.
        "omlsa_residual_tf"    — Pure TF-domain OMLSA/IMCRA-inspired gate.  Operates
                                 directly on |xt_mean|^2; no waveform conversion needed.
                                 Higher = worse.
        "stft_leakage"         — Simple non-speech/speech frame-power ratio on the TF
                                 spectrogram (sum over F then mean per region).  Matches
                                 the post-hoc _stft_leakage_score exactly.
                                 Higher = more leakage = worse.
        "traj_jump"            — Relative squared displacement between consecutive
                                 xt_mean tensors.  Speech-agnostic trajectory instability.
                                 Higher = larger jump = worse.
        "traj_curvature"       — Normalised squared second difference of consecutive
                                 xt_mean tensors.  Discrete curvature of the trajectory.
                                 Higher = sharper bend = worse.
        "pred_jump"            — Relative squared displacement between consecutive
                                 model prediction (score) tensors.  Speech-agnostic.
                                 Higher = larger jump in score = worse.

    Adding a new gate requires registering it in the if/elif chain below.
    """
    scores = {}
    for gate in gates:
        if gate == "leakage":
            scores["leakage"] = gate_step_score(step_info, cache)
        elif gate == "wiener_residual":
            scores["wiener_residual"] = gate_step_wiener_residual(step_info, cache)
#         elif gate == "omlsa_residual":
#             scores["omlsa_residual"] = gate_step_omlsa_residual(step_info, cache)
        elif gate == "omlsa_residual_tf":
            scores["omlsa_residual_tf"] = gate_step_omlsa_residual_tf(step_info, cache)
        elif gate == "omlsa_gating":
            scores["omlsa_gating"] = gate_step_omlsa_gating(step_info, cache)
        elif gate == "stft_leakage":
            scores["stft_leakage"] = gate_step_stft_leakage(step_info, cache)
        elif gate == "traj_jump":
            scores["traj_jump"] = gate_step_traj_jump(step_info, cache)
        elif gate == "traj_curvature":
            scores["traj_curvature"] = gate_step_traj_curvature(step_info, cache)
        elif gate == "pred_jump":
            scores["pred_jump"] = gate_step_pred_jump(step_info, cache)
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
    gate_max_values: dict = field(default_factory=dict)         # gate_name -> max per-step score (accepted attempt)
    gate_max_steps:  dict = field(default_factory=dict)         # gate_name -> step_idx of that max

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
    resid_est  = np.minimum(S_speech, N_f[:, None]) # still on speech frames- checking how much speech still looks like noise

    E_speech = float(speech_est.sum())
    E_resid  = float(resid_est.sum())
    return min(E_resid / (E_speech + _EPS), 1e6)


# def _omlsa_residual_tf_posthoc_score(y: np.ndarray, x_hat: np.ndarray) -> float:
#     """Post-hoc wrapper: STFT both waveforms, call _omlsa_residual_tf_score(PY, PX).

#     Uses identical STFT parameters to _omlsa_residual_score (M=512, hop=128,
#     Hamming, center=False) so results are directly comparable.
#     """
#     import librosa
#     M, Mno = 512, 128
#     T = min(len(y), len(x_hat))
#     y_   = np.asarray(y[:T],     dtype=np.float32)
#     xh_  = np.asarray(x_hat[:T], dtype=np.float32)
#     win  = np.hamming(M).astype(np.float32)
#     M21  = M // 2 + 1
#     Y    = librosa.stft(y_,  n_fft=M, hop_length=Mno, win_length=M, window=win, center=False)
#     Xh   = librosa.stft(xh_, n_fft=M, hop_length=Mno, win_length=M, window=win, center=False)
#     PY   = np.abs(Y[:M21,  :]) ** 2
#     PX   = np.abs(Xh[:M21, :]) ** 2
#     return _omlsa_residual_tf_score(PY, PX)


# def _omlsa_residual_score(y: np.ndarray, x_hat: np.ndarray) -> float:
#     """OM-LSA-inspired post-hoc residual gate. Lower = better.

#     This version keeps the successful op2 ingredients:
#       - Cohen-style recursive noise tracking from noisy y
#       - PH1 soft speech presence probability
#       - local/global/frame decisions
#       - tonal-aware logic

#     It amplifies op2 by:
#       1) upweighting hard frames using P_frame
#       2) adding a small speech-hole term
#       3) adding an explicit tonal penalty on bins flagged by Cohen-style tonal logic

#     Still a gating score, not an enhancer.
#     """
#     import numpy as np
#     import librosa
#     from scipy.special import exp1

#     EPS = 1e-10
#     MAX_SCORE = 1e6

#     # ===== Cohen / OM-LSA params =====
#     Fs_ref = 16000.0
#     M_ref = 512
#     Mo_ref = int(0.75 * M_ref)

#     w = 1
#     alpha_s_ref = 0.9
#     Nwin = 8
#     Vwin = 15
#     delta_s = 1.67
#     Bmin = 1.66
#     delta_y = 4.6
#     delta_yt = 3.0
#     alpha_d_ref = 0.85
#     alpha_d_long = 0.99

#     alpha_xi_ref = 0.7
#     w_xi_local = 1
#     w_xi_global = 15
#     f_u = 10000.0
#     f_l = 50.0
#     P_min = 0.005
#     xi_lu_dB = -5.0
#     xi_ll_dB = -10.0
#     xi_gu_dB = -5.0
#     xi_gl_dB = -10.0
#     xi_fu_dB = -5.0
#     xi_fl_dB = -10.0
#     xi_mu_dB = 10.0
#     xi_ml_dB = 0.0
#     q_max = 0.998

#     alpha_eta_ref = 0.95
#     eta_min_dB = -18.0

#     broad_flag = True
#     tone_flag = True
#     nonstat = "medium"

#     # ===== New score weights =====
#     FRAME_BOOST = 0.75      # harder frames count more
#     W_SPEECH_RESID = 0.70   # keep op2 baseline
#     W_NOISE_EXCESS = 1.00   # keep op2 baseline
#     W_SPEECH_HOLE = 0.18    # small, not dominant
#     W_TONAL = 0.35          # explicit tonal penalty

#     # ===== Basic setup =====
#     T = min(len(y), len(x_hat))
#     if T <= 0:
#         return MAX_SCORE

#     y = np.asarray(y[:T], dtype=np.float32)
#     x_hat = np.asarray(x_hat[:T], dtype=np.float32)

#     Fs = Fs_ref
#     M = M_ref
#     Mo = Mo_ref
#     Mno = M - Mo
#     alpha_s = alpha_s_ref
#     alpha_d = alpha_d_ref
#     alpha_eta = alpha_eta_ref
#     alpha_xi = alpha_xi_ref

#     eta_min = 10.0 ** (eta_min_dB / 10.0)
#     G_f = eta_min ** 0.5

#     win = np.hamming(M).astype(np.float32)

#     Y = librosa.stft(
#         y, n_fft=M, hop_length=Mno, win_length=M, window=win, center=False
#     )
#     Xh = librosa.stft(
#         x_hat, n_fft=M, hop_length=Mno, win_length=M, window=win, center=False
#     )

#     M21 = M // 2 + 1
#     Y = Y[:M21, :]
#     Xh = Xh[:M21, :]
#     PY = np.abs(Y) ** 2
#     PX = np.abs(Xh) ** 2

#     n_frames = PY.shape[1]
#     if n_frames < 2:
#         return MAX_SCORE

#     # ===== Smoothing kernels =====
#     b = np.hanning(2 * w + 1).astype(np.float32)
#     b /= np.sum(b)

#     b_xi_local = np.hanning(2 * w_xi_local + 1).astype(np.float32)
#     b_xi_local /= np.sum(b_xi_local)

#     b_xi_global = np.hanning(2 * w_xi_global + 1).astype(np.float32)
#     b_xi_global /= np.sum(b_xi_global)

#     # ===== Frequency ranges =====
#     k_u = int(round(f_u / Fs * M + 1))
#     k_l = int(round(f_l / Fs * M + 1))
#     k_u = min(k_u, M21)
#     k_l = max(1, min(k_l, M21 - 1))

#     k2_local = int(round(500.0 / Fs * M + 1))
#     k3_local = int(round(3500.0 / Fs * M + 1))
#     k2_local = max(1, min(k2_local, M21 - 1))
#     k3_local = max(k2_local + 1, min(k3_local, M21 - 1))

#     # ===== Helpers =====
#     def _conv_same(x: np.ndarray, h: np.ndarray) -> np.ndarray:
#         return np.convolve(x, h, mode="same")

#     def _db_prob(x_db: np.ndarray, lo: float, hi: float, pmin: float) -> np.ndarray:
#         out = np.ones_like(x_db, dtype=np.float32)
#         out[x_db <= lo] = pmin
#         mid = (x_db > lo) & (x_db < hi)
#         out[mid] = pmin + (x_db[mid] - lo) / (hi - lo) * (1.0 - pmin)
#         return out

#     # ===== Initial state =====
#     eta_2term = np.ones(M21, dtype=np.float32)
#     xi = np.zeros(M21, dtype=np.float32)
#     xi_frame = 0.0
#     l_mod_lswitch = 0

#     lambda_d = np.maximum(PY[:, 0].copy(), EPS)
#     Sy = PY[:, 0].copy()
#     Sf0 = _conv_same(PY[:, 0], b)
#     S = Sf0.copy()
#     St = Sf0.copy()
#     lambda_dav = PY[:, 0].copy()
#     lambda_dav_long = PY[:, 0].copy()
#     Smin = S.copy()
#     SMact = S.copy()
#     Smint = St.copy()
#     SMactt = St.copy()

#     SW = np.tile(S[:, None], (1, Nwin))
#     SWt = np.tile(St[:, None], (1, Nwin))

#     score_num = 0.0
#     score_den = 0.0

#     for l in range(n_frames):
#         Ya2 = np.maximum(PY[:, l], EPS)
#         X2 = np.maximum(PX[:, l], 0.0)

#         gamma = Ya2 / np.maximum(lambda_d, EPS)
#         eta = alpha_eta * eta_2term + (1.0 - alpha_eta) * np.maximum(gamma - 1.0, 0.0)
#         eta = np.maximum(eta, eta_min)
#         v = gamma * eta / (1.0 + eta)

#         # 2.1 frequency smoothing
#         Sf = _conv_same(Ya2, b)

#         if l == 0:
#             S = Sf.copy()
#             St = Sf.copy()
#             Smin = S.copy()
#             SMact = S.copy()
#             Smint = St.copy()
#             SMactt = St.copy()
#             lambda_dav = Ya2.copy()
#             lambda_dav_long = Ya2.copy()
#             Sy = Ya2.copy()
#         else:
#             S = alpha_s * S + (1.0 - alpha_s) * Sf
#             if l < 14:
#                 Smin = S.copy()
#                 SMact = S.copy()
#             else:
#                 Smin = np.minimum(Smin, S)
#                 SMact = np.minimum(SMact, S)

#         # IMCRA local minima logic
#         I_f = ((Ya2 < delta_y * Bmin * Smin) & (S < delta_s * Bmin * Smin)).astype(np.float32)
#         conv_I = _conv_same(I_f, b)
#         Sft = St.copy()
#         idx = conv_I > 0
#         if np.any(idx):
#             conv_Y = _conv_same(I_f * Ya2, b)
#             Sft[idx] = conv_Y[idx] / np.maximum(conv_I[idx], EPS)

#         if l < 14:
#             St = S.copy()
#             Smint = St.copy()
#             SMactt = St.copy()
#         else:
#             St = alpha_s * St + (1.0 - alpha_s) * Sft
#             Smint = np.minimum(Smint, St)
#             SMactt = np.minimum(SMactt, St)

#         qhat = np.ones(M21, dtype=np.float32)
#         phat = np.zeros(M21, dtype=np.float32)

#         if nonstat == "low":
#             gamma_mint = Ya2 / (Bmin * np.maximum(Smin, EPS))
#             zetat = S / (Bmin * np.maximum(Smin, EPS))
#         else:
#             gamma_mint = Ya2 / (Bmin * np.maximum(Smint, EPS))
#             zetat = S / (Bmin * np.maximum(Smint, EPS))

#         idx = (gamma_mint > 1.0) & (gamma_mint < delta_yt) & (zetat < delta_s)
#         qhat[idx] = (delta_yt - gamma_mint[idx]) / (delta_yt - 1.0)
#         phat[idx] = 1.0 / (
#             1.0
#             + qhat[idx] / np.maximum(1.0 - qhat[idx], EPS)
#             * (1.0 + eta[idx])
#             * np.exp(-v[idx])
#         )
#         phat[(gamma_mint >= delta_yt) | (zetat >= delta_s)] = 1.0

#         alpha_dt = alpha_d + (1.0 - alpha_d) * phat
#         lambda_dav = alpha_dt * lambda_dav + (1.0 - alpha_dt) * Ya2

#         if l < 14:
#             lambda_dav_long = lambda_dav.copy()
#         else:
#             alpha_dt_long = alpha_d_long + (1.0 - alpha_d_long) * phat
#             lambda_dav_long = alpha_dt_long * lambda_dav_long + (1.0 - alpha_dt_long) * Ya2

#         # sliding minima window
#         l_mod_lswitch += 1
#         if l_mod_lswitch == Vwin:
#             l_mod_lswitch = 0
#             if l == Vwin - 1:
#                 SW = np.tile(S[:, None], (1, Nwin))
#                 SWt = np.tile(St[:, None], (1, Nwin))
#             else:
#                 SW = np.concatenate([SW[:, 1:], SMact[:, None]], axis=1)
#                 Smin = np.min(SW, axis=1)
#                 SMact = S.copy()

#                 SWt = np.concatenate([SWt[:, 1:], SMactt[:, None]], axis=1)
#                 Smint = np.min(SWt, axis=1)
#                 SMactt = St.copy()

#         if nonstat == "high":
#             lambda_d = 2.0 * lambda_dav
#         else:
#             lambda_d = 1.4685 * lambda_dav
#         lambda_d = np.maximum(lambda_d, EPS)

#         # ===== A priori probability of signal absence =====
#         xi = alpha_xi * xi + (1.0 - alpha_xi) * eta
#         xi_local = _conv_same(xi, b_xi_local)
#         xi_global = _conv_same(xi, b_xi_global)

#         prev_xi_frame = xi_frame
#         xi_frame = float(np.mean(xi[k_l:k_u]))
#         dxi_frame = xi_frame - prev_xi_frame

#         xi_local_dB = 10.0 * np.log10(np.maximum(xi_local, 1e-10))
#         xi_global_dB = 10.0 * np.log10(np.maximum(xi_global, 1e-10))
#         xi_frame_dB = 10.0 * np.log10(max(xi_frame, 1e-10))

#         P_local = _db_prob(xi_local_dB, xi_ll_dB, xi_lu_dB, P_min)
#         P_global = _db_prob(xi_global_dB, xi_gl_dB, xi_gu_dB, P_min)

#         lo = min(3, M21 - 1)
#         hi = min(k2_local + k3_local - 3, M21)
#         if hi > lo:
#             m_P_local = float(np.mean(P_local[lo:hi]))
#         else:
#             m_P_local = float(np.mean(P_local))

#         tonal_mask = np.zeros(M21, dtype=np.float32)

#         if m_P_local < 0.25:
#             P_local[k2_local:k3_local] = P_min

#         if tone_flag and (m_P_local < 0.5) and (l > 120) and M21 > 16:
#             lhs = lambda_dav_long[7:(M21 - 8)]
#             rhs = 2.5 * (
#                 lambda_dav_long[9:(M21 - 6)] + lambda_dav_long[5:(M21 - 10)]
#             )
#             tonal_idx = np.where(lhs > rhs)[0] + 6
#             tonal_idx = tonal_idx[(tonal_idx >= 0) & (tonal_idx < M21)]
#             if tonal_idx.size > 0:
#                 P_local[tonal_idx] = P_min
#                 tonal_mask[tonal_idx] = 1.0

#         if xi_frame_dB <= xi_fl_dB:
#             P_frame = P_min
#         elif dxi_frame >= 0:
#             xi_m_dB = min(max(xi_frame_dB, xi_ml_dB), xi_mu_dB)
#             P_frame = 1.0
#         elif xi_frame_dB >= xi_m_dB + xi_fu_dB:
#             P_frame = 1.0
#         elif xi_frame_dB <= xi_m_dB + xi_fl_dB:
#             P_frame = P_min
#         else:
#             P_frame = P_min + (
#                 (xi_frame_dB - xi_m_dB - xi_fl_dB) / (xi_fu_dB - xi_fl_dB)
#             ) * (1.0 - P_min)

#         if broad_flag:
#             q = 1.0 - P_global * P_local * P_frame
#         else:
#             q = 1.0 - P_local * P_frame
#         q = np.minimum(q, q_max)

#         gamma = Ya2 / np.maximum(lambda_d, EPS)
#         eta = alpha_eta * eta_2term + (1.0 - alpha_eta) * np.maximum(gamma - 1.0, 0.0)
#         eta = np.maximum(eta, eta_min)
#         v = gamma * eta / (1.0 + eta)

#         PH1 = np.zeros(M21, dtype=np.float32)
#         idx = q < 0.9
#         PH1[idx] = 1.0 / (
#             1.0
#             + q[idx] / np.maximum(1.0 - q[idx], EPS)
#             * (1.0 + eta[idx])
#             * np.exp(-v[idx])
#         )

#         # ===== Spectral gains =====
#         GH1 = np.ones(M21, dtype=np.float32)
#         idx_hi = v > 5.0
#         GH1[idx_hi] = eta[idx_hi] / (1.0 + eta[idx_hi])

#         idx_mid = (v > 0.0) & (v <= 5.0)
#         if np.any(idx_mid):
#             vv = np.maximum(v[idx_mid], 1e-8)
#             GH1[idx_mid] = (
#                 eta[idx_mid] / (1.0 + eta[idx_mid]) * np.exp(0.5 * exp1(vv))
#             )

#         if tone_flag:
#             lambda_d_global = lambda_d.copy()
#             if M21 > 6:
#                 tmp = lambda_d_global.copy()
#                 tmp[3:(M21 - 3)] = np.minimum.reduce([
#                     lambda_d_global[3:(M21 - 3)],
#                     lambda_d_global[0:(M21 - 6)],
#                     lambda_d_global[6:M21],
#                 ])
#                 lambda_d_global = tmp

#             Sy = 0.8 * Sy + 0.2 * Ya2
#             GH0 = G_f * np.sqrt(lambda_d_global / np.maximum(Sy, EPS))
#         else:
#             GH0 = np.full(M21, G_f, dtype=np.float32)

#         G = (GH1 ** PH1) * (GH0 ** (1.0 - PH1))
#         eta_2term = (GH1 ** 2) * gamma

#         # ===== op3 score =====
#         # harder / less certain frames get larger weight
#         frame_weight = 1.0 + FRAME_BOOST * (1.0 - float(P_frame))

#         target_floor = np.maximum(lambda_d, EPS)

#         # op2 core terms
#         speech_resid = np.sum(PH1 * np.minimum(X2, target_floor))
#         noise_excess = np.sum((1.0 - PH1) * X2)
#         speech_keep = np.sum(PH1 * X2)

#         # mild speech-hole term: only where speech is likely and GH1 is strong
#         speech_hole = np.sum(PH1 * GH1 * np.maximum(target_floor - X2, 0.0))

#         # explicit tonal leftover penalty
#         tonal_penalty = np.sum(tonal_mask * np.maximum(X2 - GH0 * target_floor, 0.0))

#         frame_num = (
#             W_SPEECH_RESID * speech_resid
#             + W_NOISE_EXCESS * noise_excess
#             + W_SPEECH_HOLE * speech_hole
#             + W_TONAL * tonal_penalty
#         )

#         score_num += float(frame_weight * frame_num)
#         score_den += float(speech_keep + EPS)

#     score = score_num / (score_den + EPS)
#     if not np.isfinite(score):
#         return MAX_SCORE
#     return min(float(score), MAX_SCORE)


# Gates that operate exclusively on per-step xt_mean inside the sampler callback.
# They have no meaningful post-hoc implementation on the final waveform, so
# compute_posthoc_gate_score returns None when all requested gates are in this set.
_PERSTEP_ONLY_GATES = frozenset({"traj_jump", "traj_curvature", "pred_jump"})


def compute_posthoc_gate_score(
    gates: list,
    y_np: np.ndarray,
    x_hat_np: np.ndarray,
    gate_combine: str = "max",
    sr: int = None,
) -> Optional[float]:
    """Post-hoc gate score on the final waveform.  Convention: lower = worse.

    gates       : list of gate names (same as --gates in enhancement.py)
    y_np        : noisy input waveform, 1-D numpy float array (normalized)
    x_hat_np    : enhanced waveform, same scale as y_np (normalized)
    gate_combine: "max" or "mean" — how to combine multiple gates
    sr          : sample rate of x_hat_np in Hz (required for "nisqa")

    Returns None when all requested gates are per-step-only (traj_jump,
    traj_curvature) and therefore have no post-hoc waveform score.

    Gate directions:
        leakage          lower = better (less noise leaked through)
        wiener_residual  lower = better (less residual noise)
        stft_leakage     lower = better (less power in silence frames)
        nisqa            higher = better → negated so convention stays lower = worse
        traj_jump        per-step only — skipped here, returns None if sole gate
        traj_curvature   per-step only — skipped here, returns None if sole gate
        pred_jump        per-step only — skipped here, returns None if sole gate
    """
    scores = {}
    for gate in gates:
        if gate in _PERSTEP_ONLY_GATES:
            continue  # per-step only; no post-hoc waveform score available
        if gate == "leakage":
            mask = _speech_mask(y_np)
            scores["leakage"] = compute_speech_gate_score(y_np, x_hat_np, mask)
        elif gate == "wiener_residual":
            scores["wiener_residual"] = _wiener_residual_score(y_np, x_hat_np)
#         elif gate == "omlsa_residual":
#             scores["omlsa_residual"] = _omlsa_residual_score(y_np, x_hat_np)
        elif gate == "stft_leakage":
            scores["stft_leakage"] = _stft_leakage_score(y_np, x_hat_np)
        elif gate == "nisqa":
            if sr is None:
                raise ValueError("sr is required for the 'nisqa' gate")
            from .nisqa_helper import compute_nisqa
            raw = compute_nisqa(x_hat_np, sr)
            # Negate: nisqa is higher = better; convention here is lower = worse
            scores["nisqa"] = -float(raw) if raw is not None else 0.0
    if not scores:
        return None  # all requested gates are per-step-only; no post-hoc score
    return combine_gate_scores(scores, gate_combine)
