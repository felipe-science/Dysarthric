import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from dysarthric_speech import data
from dysarthric_speech.__main__ import main


def test_preparation_matches_legacy_split(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    paths = [f"{i}.wav" for i in range(12)]
    manifest.write_text(
        "path,label\n" + "".join(f"{name},{i % 2}.0\n" for i, name in enumerate(paths)),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        data, "load_audio", lambda path: (np.asarray([int(path.stem)]), 16000)
    )
    monkeypatch.setattr(
        data, "extract_features", lambda signal, rate, variant: np.repeat(signal, 33)
    )

    archive = data.prepare_dataset(manifest, tmp_path, tmp_path / "prepared", "normal")
    train, test = train_test_split(np.arange(len(paths)), test_size=0.2, random_state=42)
    with np.load(archive, allow_pickle=False) as prepared:
        np.testing.assert_array_equal(prepared["paths_train"], np.asarray(paths)[train])
        np.testing.assert_array_equal(prepared["paths_test"], np.asarray(paths)[test])
    features, labels, variant = data.load_split(archive.parent, "test")
    np.testing.assert_array_equal(features[:, 0], test)
    np.testing.assert_array_equal(labels, test % 2)
    assert variant == "normal"


def test_invalid_delta_alpha_stops_preparation(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,label\nrecording.wav,1\n", encoding="utf-8")
    monkeypatch.setattr(data, "load_audio", lambda path: (np.zeros(8192), 16000))
    monkeypatch.setattr(
        data, "extract_features", lambda signal, rate, variant: np.r_[np.zeros(34), np.nan]
    )

    with pytest.raises(ValueError, match="Non-finite multifrac features at manifest row 2"):
        data.prepare_dataset(manifest, tmp_path, tmp_path / "prepared", "multifrac")
    assert not (tmp_path / "prepared" / "dataset.npz").exists()


def test_demo_trains_and_reloads_checkpoint(tmp_path):
    pytest.importorskip("librosa")
    pytest.importorskip("imblearn")
    pytest.importorskip("torch")

    assert main(["demo", "--output-dir", str(tmp_path)]) == 0

    def read_report(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    trained = read_report(tmp_path / "run" / "metrics.json")
    evaluated = read_report(tmp_path / "evaluation" / "metrics.json")
    assert trained == evaluated
    assert sum(map(sum, trained["confusion_matrix"])) == 6
    assert (tmp_path / "run" / "loss.png").is_file()
