"""Thin wrapper around speechmos DNSMOS p.835.

Returns None gracefully when the package is unavailable so callers can skip
DNSMOS evaluation without crashing.

Install: pip install speechmos
"""
import numpy as np

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
