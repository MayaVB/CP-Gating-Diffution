import math
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
