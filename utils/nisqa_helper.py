"""Thin wrapper around the NISQA speech-quality model.

Returns None gracefully when the dependency is unavailable so callers can
skip NISQA evaluation without crashing.

Assumed package / API
---------------------
Targets the ``nisqa`` PyPI package (pip install nisqa — requires torch).
  https://github.com/gabrielmittag/NISQA

Import path used::

    from nisqa.NISQA_model import nisqaModel

    model = nisqaModel(
        mode="predict_file",
        pretrained_model="<path/to/nisqa.tar>",
        deg="<path/to/audio.wav>",
        output_dir=None,
        ms_channel=None,
    )
    model.predict()
    mos = float(model.df["mos_pred"].iloc[0])

NISQA expects 48 000 Hz mono audio.  Audio is resampled internally.

Checkpoint discovery
--------------------
Set the environment variable ``NISQA_CKPT_PATH`` to the .tar checkpoint, or
place it at ``<repo_root>/nisqa_weights/nisqa.tar`` (default).

Install::

    pip install nisqa
    # Download checkpoint, e.g.:
    # wget https://github.com/gabrielmittag/NISQA/releases/download/v1.0/nisqa.tar

If the exact installed API differs, adapt only the inference block inside
``compute_nisqa``; the public interface stays fixed.
"""

import os
import tempfile
import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NISQA_CKPT_PATH: str = os.environ.get(
    "NISQA_CKPT_PATH",
    os.path.join(os.path.dirname(__file__), "..", "nisqa_weights", "nisqa.tar"),
)

NISQA_SR: int = 48_000

# ---------------------------------------------------------------------------
# Lazy availability check
# ---------------------------------------------------------------------------
_CHECKED: bool = False
_AVAILABLE: bool = False
# Emit the checkpoint-missing warning at most once per process.
_CKPT_WARNED: bool = False


def is_available() -> bool:
    """Return True if nisqa.NISQA_model and soundfile are both importable.

    Does NOT require the checkpoint to exist — checkpoint presence is checked
    lazily inside ``compute_nisqa``.  Result is cached after the first call.
    """
    global _CHECKED, _AVAILABLE
    if not _CHECKED:
        _CHECKED = True
        try:
            from nisqa.NISQA_model import nisqaModel  # noqa: F401
            import soundfile  # noqa: F401
            _AVAILABLE = True
        except ImportError:
            _AVAILABLE = False
    return _AVAILABLE


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_mono_float32(audio: np.ndarray) -> "np.ndarray | None":
    """Convert *audio* to a 1-D float32 mono array.

    Accepts:
      - 1-D [T]
      - 2-D [T, C] or [C, T] — smaller dimension is treated as channels

    Returns None for empty input or unsupported rank.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        pass  # already mono
    elif audio.ndim == 2:
        # Collapse whichever axis is smaller (channels).
        if audio.shape[0] <= audio.shape[1]:
            # shape [C, T] — channels along axis 0
            audio = audio.mean(axis=0)
        else:
            # shape [T, C] — channels along axis 1
            audio = audio.mean(axis=1)
    else:
        return None  # unexpected rank

    if audio.size == 0:
        return None

    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    return audio


def _to_nisqa_sr(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample 1-D float32 *audio* to NISQA_SR (48 kHz).

    Algorithm preference: scipy.signal.resample_poly (polyphase, integer
    ratio, low aliasing) → librosa fallback (Kaiser-window Fourier).
    Always returns float32.
    """
    if sr == NISQA_SR:
        return audio
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(NISQA_SR, sr)
        return resample_poly(audio, NISQA_SR // g, sr // g).astype(np.float32)
    except ImportError:
        import librosa
        return librosa.resample(audio, orig_sr=sr, target_sr=NISQA_SR).astype(np.float32)


def _write_temp_wav(audio: np.ndarray, sr: int) -> str:
    """Write *audio* to a named temp WAV file and return its path."""
    import soundfile as sf
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, audio, sr)
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_nisqa(audio: np.ndarray, sr: int) -> "float | None":
    """Compute NISQA overall MOS score for a waveform.

    Args:
        audio: Waveform as a NumPy array.  Accepted shapes: [T], [T, C], [C, T].
               Collapsed to mono and resampled to 48 kHz internally.
        sr:    Sampling rate of *audio* in Hz.

    Returns:
        Float MOS-like score (1–5 scale, higher = better), or None on any
        failure (unavailable package, missing checkpoint, inference error, …).
        Never raises.
    """
    global _CKPT_WARNED

    if not is_available():
        return None

    # --- Input sanitation ---
    mono = _to_mono_float32(audio)
    if mono is None:
        return None

    # --- Checkpoint check (warn once) ---
    ckpt = NISQA_CKPT_PATH
    if not os.path.isfile(ckpt):
        if not _CKPT_WARNED:
            warnings.warn(
                f"NISQA checkpoint not found at '{ckpt}'.  "
                "Set the NISQA_CKPT_PATH environment variable.  "
                "Skipping NISQA scoring.",
                RuntimeWarning,
                stacklevel=2,
            )
            _CKPT_WARNED = True
        return None

    tmp_path = None
    try:
        audio_48k = _to_nisqa_sr(mono, sr)
        tmp_path = _write_temp_wav(audio_48k, NISQA_SR)

        # --- nisqaModel takes a single dict argument (not kwargs) ---
        from nisqa.NISQA_model import nisqaModel  # type: ignore[import]
        import contextlib, io

        _sink = io.StringIO()
        with contextlib.redirect_stdout(_sink):
            model = nisqaModel({
                "mode": "predict_file",
                "pretrained_model": ckpt,
                "deg": tmp_path,
                "output_dir": None,
                "ms_channel": None,
            })
            # predict() stores results in model.ds_val.df and also returns the df
            df = model.predict()

        # Overall MOS column is 'mos_pred'
        return float(df["mos_pred"].iloc[0])
        # -----------------------------------------------------------

    except Exception:
        return None
    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
