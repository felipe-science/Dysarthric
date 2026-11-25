"""Command-line workflows for preparation, training, and evaluation."""

import argparse
import csv
import json
import logging
import pickle
import platform
from importlib.metadata import version
from pathlib import Path
from zipfile import BadZipFile

from .features import VARIANT_SIZES

LOGGER = logging.getLogger(__name__)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Extract features and create the historical split."
    )
    prepare.add_argument("--manifest", required=True, type=Path)
    prepare.add_argument("--data-root", required=True, type=Path)
    prepare.add_argument("--variant", required=True, choices=VARIANT_SIZES)
    train = commands.add_parser("train", help="Train and evaluate the historical MLP.")
    train.add_argument("--input-dir", required=True, type=Path)
    train.add_argument("--variant", required=True, choices=VARIANT_SIZES)
    train.add_argument("--epochs", type=int, default=5000)
    train.add_argument("--seed", type=int)
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate a package checkpoint on a test split."
    )
    evaluate.add_argument("--input-dir", required=True, type=Path)
    evaluate.add_argument("--checkpoint", required=True, type=Path)
    demo = commands.add_parser("demo", help="Run a short workflow on synthetic audio.")
    for command in (prepare, train, evaluate, demo):
        command.add_argument("--output-dir", required=True, type=Path)
    return parser


def _train(input_dir: Path, output_dir: Path, variant: str, epochs: int, seed: int | None) -> None:
    from .data import SPLIT_SEED, TEST_FRACTION, load_split
    from .evaluate import write_evaluation, write_training_loss
    from .model import predict, save_checkpoint, train_model

    features, labels, train_variant = load_split(input_dir, "train")
    test_features, test_labels, test_variant = load_split(input_dir, "test")
    if variant != train_variant or variant != test_variant:
        raise ValueError("The requested variant does not match the prepared dataset.")
    model, losses = train_model(features, labels, epochs=epochs, seed=seed)
    probabilities = predict(model, test_features)
    metrics = write_evaluation(test_labels, probabilities, output_dir)
    save_checkpoint(model, variant, output_dir / "model.pt")
    write_training_loss(losses, output_dir)
    run = {
        "variant": variant,
        "epochs": epochs,
        "torch_seed": seed,
        "split_seed": SPLIT_SEED,
        "test_fraction": TEST_FRACTION,
        "training_recordings_before_smote": len(labels),
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in (
                "imbalanced-learn", "matplotlib", "numpy", "scikit-learn", "scipy", "torch"
            )
        },
    }
    (output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Test accuracy %.4f. Reports saved to %s.", metrics["accuracy"], output_dir)


def _evaluate(input_dir: Path, checkpoint: Path, output_dir: Path) -> None:
    from .data import load_split
    from .evaluate import write_evaluation
    from .model import load_checkpoint, predict

    features, labels, variant = load_split(input_dir, "test")
    model, model_variant = load_checkpoint(checkpoint)
    if variant != model_variant:
        raise ValueError("The checkpoint variant does not match the prepared dataset.")
    metrics = write_evaluation(labels, predict(model, features), output_dir)
    LOGGER.info("Test accuracy %.4f. Reports saved to %s.", metrics["accuracy"], output_dir)


def _demo(output_dir: Path) -> None:
    import numpy as np
    from scipy.io.wavfile import write

    from .data import prepare_dataset

    source = output_dir / "synthetic"
    source.mkdir(parents=True, exist_ok=True)
    manifest = source / "manifest.csv"
    sample_rate = 16000
    time = np.arange(sample_rate // 2) / sample_rate
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label"])
        for index in range(30):
            label = index % 2
            frequency = 225 + label * 200 + index * 3
            signal = 0.5 * np.sin(2 * np.pi * frequency * time)
            filename = f"tone-{index:02d}.wav"
            write(source / filename, sample_rate, (signal * 32767).astype(np.int16))
            writer.writerow([filename, label])

    prepared = output_dir / "prepared"
    run = output_dir / "run"
    prepare_dataset(manifest, source, prepared, "normal")
    _train(prepared, run, "normal", epochs=3, seed=42)
    _evaluate(prepared, run / "model.pt", output_dir / "evaluation")


def main(argv: list[str] | None = None) -> int:
    """Run a supported command and return a shell exit status."""
    parser = _make_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.command == "prepare":
            from .data import prepare_dataset

            path = prepare_dataset(args.manifest, args.data_root, args.output_dir, args.variant)
            LOGGER.info("Dataset saved to %s.", path)
        elif args.command == "train":
            _train(args.input_dir, args.output_dir, args.variant, args.epochs, args.seed)
        elif args.command == "evaluate":
            _evaluate(args.input_dir, args.checkpoint, args.output_dir)
        else:
            _demo(args.output_dir)
    except (OSError, RuntimeError, ValueError, BadZipFile, pickle.UnpicklingError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
