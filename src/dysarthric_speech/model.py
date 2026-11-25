"""Training and inference for the acoustic feature classifier."""

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn

from .features import VARIANT_SIZES

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1


class SimpleNN(nn.Module):
    """The two-hidden-layer network used in the original experiments."""

    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(feature_count, 64, device="cpu", dtype=torch.float32)
        self.layer2 = nn.Linear(64, 64, device="cpu", dtype=torch.float32)
        self.layer3 = nn.Linear(64, 1, device="cpu", dtype=torch.float32)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features: Tensor) -> Tensor:
        hidden = torch.relu(self.layer1(features))
        hidden = torch.relu(self.layer2(hidden))
        return self.sigmoid(self.layer3(hidden))


def _feature_matrix(features: ArrayLike, expected_size: int | None = None) -> NDArray[np.float32]:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("Features must be a nonempty two-dimensional matrix.")
    if expected_size is not None and matrix.shape[1] != expected_size:
        raise ValueError(f"Expected {expected_size} features per row, got {matrix.shape[1]}.")
    if not np.isfinite(matrix).all():
        raise ValueError("Features contain NaN or infinite values.")
    return matrix


def train_model(
    features: ArrayLike,
    labels: ArrayLike,
    epochs: int = 5000,
    seed: int | None = None,
) -> tuple[SimpleNN, list[float]]:
    """Fit on training rows using SMOTE and return the loss for each epoch."""
    if epochs < 1:
        raise ValueError("Epochs must be positive.")

    matrix = _feature_matrix(features)
    targets = np.asarray(labels)
    if targets.ndim != 1 or len(targets) != len(matrix):
        raise ValueError("Labels must be a one-dimensional array matching the feature rows.")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")

    smote = SMOTE(sampling_strategy="auto", random_state=42)
    matrix, targets = smote.fit_resample(matrix, targets.astype(np.float32))
    inputs = torch.tensor(matrix, device="cpu", dtype=torch.float32)
    outputs = torch.tensor(targets, device="cpu", dtype=torch.float32)

    if seed is None:
        model = SimpleNN(inputs.shape[1])
    else:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = SimpleNN(inputs.shape[1])

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        probabilities = model(inputs).squeeze(-1)
        loss = criterion(probabilities, outputs)
        loss.backward()
        optimizer.step()
        history.append(loss.item())

        if epoch % 10 == 0:
            logger.info("Epoch %d/%d, loss %.4f", epoch, epochs, history[-1])

    return model, history


def predict(model: SimpleNN, features: ArrayLike) -> NDArray[np.float32]:
    """Return one class-1 probability per row in the supplied feature matrix."""
    matrix = _feature_matrix(features, model.layer1.in_features)
    inputs = torch.tensor(
        matrix, device=model.layer1.weight.device, dtype=model.layer1.weight.dtype
    )
    model.eval()
    with torch.inference_mode():
        return model(inputs).squeeze(-1).cpu().numpy()


def save_checkpoint(model: SimpleNN, variant: str, path: str | Path) -> None:
    """Write a versioned checkpoint containing model weights and the feature variant."""
    if variant not in VARIANT_SIZES:
        raise ValueError(f"Unknown feature variant: {variant!r}.")
    if model.layer1.in_features != VARIANT_SIZES[variant]:
        raise ValueError(f"Model input size does not match feature variant {variant!r}.")

    state_dict = model.state_dict()
    if not all(torch.isfinite(value).all() for value in state_dict.values()):
        raise ValueError("Model parameters contain NaN or infinite values.")
    checkpoint = {"version": CHECKPOINT_VERSION, "variant": variant, "state_dict": state_dict}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary_path = Path(stream.name)
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(path: str | Path) -> tuple[SimpleNN, str]:
    """Restore a weights-only checkpoint as a CPU model in evaluation mode."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a versioned model checkpoint.")
    version = checkpoint.get("version")
    if type(version) is not int or version != CHECKPOINT_VERSION:
        raise ValueError("Unsupported model checkpoint version.")
    variant = checkpoint.get("variant")
    if not isinstance(variant, str) or variant not in VARIANT_SIZES:
        raise ValueError("Checkpoint has an unknown feature variant.")
    weights = checkpoint.get("state_dict")
    if not isinstance(weights, dict):
        raise ValueError("Checkpoint is missing its model weights.")

    with torch.random.fork_rng(devices=[]):
        model = SimpleNN(VARIANT_SIZES[variant])
    expected = model.state_dict()
    if weights.keys() != expected.keys():
        raise ValueError("Checkpoint parameter names do not match the network architecture.")
    for name, value in weights.items():
        if (
            not isinstance(value, Tensor)
            or value.shape != expected[name].shape
            or value.dtype != expected[name].dtype
            or value.layout != torch.strided
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Invalid checkpoint parameter: {name}.")
    model.load_state_dict(weights)
    return model.eval(), variant
