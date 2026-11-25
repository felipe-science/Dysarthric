"""Prepare and load recording-level dataset splits."""

import csv
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from sklearn.model_selection import train_test_split

from .features import VARIANT_SIZES, extract_features, load_audio

LOGGER = logging.getLogger(__name__)
DATASET_FILENAME = "dataset.npz"
SPLIT_SEED = 42
TEST_FRACTION = 0.2


def prepare_dataset(
    manifest: Path, data_root: Path, output_dir: Path, variant: str
) -> Path:
    """Extract features and preserve the historical recording-level split."""
    if variant not in VARIANT_SIZES:
        raise ValueError(f"Unrecognized feature variant: {variant}.")
    data_root = data_root.resolve(strict=True)
    if not data_root.is_dir():
        raise ValueError("The data root must be a directory.")
    features = []
    labels = []
    recording_paths = []
    skipped = 0
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"path", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("The manifest must contain path and label columns.")

        for line_number, row in enumerate(reader, start=2):
            path_text = row["path"]
            try:
                label = float(row["label"] or "")
            except ValueError as exc:
                raise ValueError(f"Invalid binary label at manifest row {line_number}.") from exc
            if not path_text or label not in {0, 1}:
                raise ValueError(f"Invalid path or binary label at manifest row {line_number}.")
            relative_path = Path(path_text)
            audio_path = (data_root / relative_path).resolve()
            if relative_path.is_absolute() or not audio_path.is_relative_to(data_root):
                raise ValueError(f"Path at manifest row {line_number} must stay within data-root.")
            try:
                waveform, sample_rate = load_audio(audio_path)
                vector = extract_features(waveform, sample_rate, variant)
            except (OSError, RuntimeError, ValueError) as exc:
                LOGGER.warning("Skipping manifest row %d: %s", line_number, exc)
                skipped += 1
                continue
            if not np.isfinite(vector).all():
                raise ValueError(
                    f"Non-finite {variant} features at manifest row {line_number}. "
                    "A scientific policy for invalid features has not been selected."
                )
            features.append(vector)
            labels.append(label)
            recording_paths.append(relative_path.as_posix())

    if len(features) < 2:
        raise ValueError("At least two usable recordings are required to create a split.")
    x = np.asarray(features, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Extracted features exceed the supported float32 range.")
    y = np.asarray(labels, dtype=np.float32)
    ids = np.asarray(recording_paths)
    train, test = train_test_split(
        np.arange(len(y)), test_size=TEST_FRACTION, random_state=SPLIT_SEED
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / DATASET_FILENAME
    with NamedTemporaryFile(dir=output_dir, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                variant=variant,
                split_seed=SPLIT_SEED,
                test_fraction=TEST_FRACTION,
                x_train=x[train],
                y_train=y[train],
                paths_train=ids[train],
                x_test=x[test],
                y_test=y[test],
                paths_test=ids[test],
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    LOGGER.info(
        "Prepared %d training and %d test recordings (%d skipped).", len(train), len(test), skipped
    )
    return destination


def load_split(
    input_dir: Path, split: str
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load and validate a single split without allowing pickled objects."""
    if split not in {"train", "test"}:
        raise ValueError(f"Unrecognized dataset split: {split}.")
    with np.load(input_dir / DATASET_FILENAME, allow_pickle=False) as archive:
        try:
            variant = archive["variant"].item()
            features = archive[f"x_{split}"]
            labels = archive[f"y_{split}"]
        except (KeyError, ValueError) as exc:
            raise ValueError("The dataset archive does not match the expected format.") from exc

    if not isinstance(variant, str) or variant not in VARIANT_SIZES:
        raise ValueError("The dataset archive contains an unrecognized variant.")
    if (
        features.ndim != 2
        or features.shape[1] != VARIANT_SIZES[variant]
        or labels.shape != (len(features),)
        or not len(features)
        or features.dtype.kind not in "fiu"
        or labels.dtype.kind not in "fiu"
    ):
        raise ValueError(f"Invalid array shape or type in the {split} split.")
    features = features.astype(np.float32, copy=False)
    if not np.isfinite(features).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError(f"The {split} split has non-finite features or invalid binary labels.")
    return features, labels.astype(np.float32, copy=False), variant
