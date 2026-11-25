import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from dysarthric_speech import features
from dysarthric_speech.multifractal import delta_alpha


def test_feature_order_and_peak_density(monkeypatch):
    waveform = np.array([0, 2, -2, -1, -2, 3, 0], dtype=np.float32)
    mfcc = np.column_stack((np.arange(13), np.arange(13) + 2))
    chroma = np.full((12, 2), 20)
    contrast = np.full((7, 2), 30)
    acoustic_features = SimpleNamespace(
        mfcc=lambda **kwargs: mfcc,
        chroma_stft=lambda **kwargs: chroma,
        spectral_contrast=lambda **kwargs: contrast,
        zero_crossing_rate=lambda samples: np.array([[0.25, 0.75]]),
    )
    monkeypatch.setitem(sys.modules, "librosa", SimpleNamespace(feature=acoustic_features))
    monkeypatch.setattr(features, "delta_alpha", lambda samples: 0.125)

    normal = features.extract_features(waveform, 16000, "normal")
    max_density = features.extract_features(waveform, 16000, "max_density")
    multifrac = features.extract_features(waveform, 16000, "multifrac")

    np.testing.assert_array_equal(normal, [*range(1, 14), *([20] * 12), *([30] * 7), 0.5])
    np.testing.assert_array_equal(max_density, [*normal, 2 / 7])
    np.testing.assert_array_equal(multifrac, [*max_density, 0.125])


def test_delta_alpha_matches_legacy_reference():
    waveform = np.random.default_rng(42).normal(size=8192).astype(np.float32)
    assert delta_alpha(waveform) == pytest.approx(0.025699513422546838, abs=1e-10)


@pytest.mark.parametrize(
    "waveform",
    [np.random.default_rng(42).normal(size=8191), np.zeros(8192)],
    ids=["short recording", "silence"],
)
def test_delta_alpha_preserves_legacy_nonfinite_results(waveform):
    with pytest.warns(RuntimeWarning):
        value = delta_alpha(waveform)
    assert np.isnan(value)


def test_audio_features_from_wave_file(tmp_path):
    pytest.importorskip("librosa")
    waveform = np.random.default_rng(42).normal(scale=0.1, size=8192).astype(np.float32)
    path = tmp_path / "generated.wav"
    wavfile.write(path, 16000, waveform)

    decoded, sample_rate = features.load_audio(path)

    assert sample_rate == 16000
    np.testing.assert_array_equal(decoded, waveform)
    extracted = {
        name: features.extract_features(decoded, sample_rate, name)
        for name in features.VARIANT_SIZES
    }
    for name, size in features.VARIANT_SIZES.items():
        assert extracted[name].shape == (size,)
        assert np.isfinite(extracted[name]).all()
    np.testing.assert_array_equal(extracted["max_density"][:33], extracted["normal"])
    np.testing.assert_array_equal(extracted["multifrac"][:34], extracted["max_density"])
