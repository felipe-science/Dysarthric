"""Extract the acoustic feature vectors used by the three experiments."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from .multifractal import delta_alpha

VARIANT_SIZES = {"normal": 33, "max_density": 34, "multifrac": 35}


def load_audio(path: str | Path) -> tuple[NDArray[np.floating], int]:
    """Decode a recording to mono while preserving its native sample rate."""
    import librosa

    return librosa.load(path, sr=None)


def extract_features(
    waveform: NDArray[np.floating], sample_rate: int, variant: str
) -> NDArray[np.floating]:
    """Return the legacy feature vector for a decoded mono recording."""
    if variant not in VARIANT_SIZES:
        variants = ", ".join(VARIANT_SIZES)
        raise ValueError(f"Unknown feature variant {variant!r}. Choose from {variants}.")

    import librosa

    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=waveform, sr=sample_rate)
    contrast = librosa.feature.spectral_contrast(y=waveform, sr=sample_rate)
    zero_crossing = librosa.feature.zero_crossing_rate(waveform)
    features = [
        np.mean(mfcc, axis=1),
        chroma.mean(axis=1),
        contrast.mean(axis=1),
        np.array([np.mean(zero_crossing)]),
    ]

    if variant in ("max_density", "multifrac"):
        peaks, _ = find_peaks(waveform, height=0)
        features.append(np.array([len(peaks) / len(waveform)]))

    if variant == "multifrac":
        features.append(np.array([delta_alpha(waveform)]))

    return np.concatenate(features)
