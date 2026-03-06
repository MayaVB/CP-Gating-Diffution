"""Thin wrapper around speechmos DNSMOS p.835.

Returns None gracefully when the package is unavailable so callers can skip
DNSMOS evaluation without crashing.

Install: pip install speechmos
"""
import numpy as np


def to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample *audio* to 16 kHz, collapse to mono, return float32.

    This is the ONE shared preprocessing function used everywhere DNSMOS is
    computed (oracle_best_of_k_dnsmos_eval.py, calc_metrics.py, …).
    Using different resamplers for selection vs. evaluation would mean the
    selected file maximises score under one resampler while the reported
    metric uses another — the rankings can disagree and selection appears
    broken.

    Algorithm preference: scipy.signal.resample_poly (polyphase, integer
    ratio, low aliasing) → librosa fallback (Kaiser-window Fourier).
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:           # [T, channels] soundfile layout → mono
        audio = audio.mean(axis=1)
    if sr == 16000:
        return audio
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(16000, sr)
        return resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
    except ImportError:
        import librosa
        return librosa.resample(audio, orig_sr=sr, target_sr=16000)


_CHECKED = False
_AVAILABLE = False


def is_available() -> bool:
    """Check once whether speechmos is importable."""
    global _CHECKED, _AVAILABLE
    if not _CHECKED:
        try:
            import speechmos  # noqa: F401
            _AVAILABLE = True
        except ImportError:
            _AVAILABLE = False
        _CHECKED = True
    return _AVAILABLE


def compute_dnsmos(wav_np, sr: int):
    """Compute DNSMOS overall MOS for a waveform.

    Args:
        wav_np : numpy array, shape [T] or [1, T], float32 or float64
        sr     : sample rate in Hz (speechmos expects 16000)

    Returns:
        float ovrl_mos, or None if speechmos is unavailable.
    """
    if not is_available():
        return None
    from speechmos import dnsmos
    audio = np.asarray(wav_np, dtype=np.float32).squeeze()
    result = dnsmos.run(audio, sr)
    return float(result["ovrl_mos"])
