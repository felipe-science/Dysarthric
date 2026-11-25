"""Characterization tests for the MLP and its checkpoint format."""

import numpy as np
import pytest
from imblearn.over_sampling import SMOTE

torch = pytest.importorskip("torch")

from dysarthric_speech import model  # noqa: E402


def test_training_preserves_legacy_full_batch_updates():
    features = np.random.default_rng(5).normal(size=(18, 33)).astype(np.float32)
    labels = np.array([0] * 11 + [1] * 7, dtype=np.float32)
    seed = 42
    epochs = 3

    expected_features, expected_labels = SMOTE(
        sampling_strategy="auto", random_state=42
    ).fit_resample(features, labels)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        expected = torch.nn.Sequential(
            torch.nn.Linear(33, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
        )
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(expected.parameters(), lr=0.001)
    expected_history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = expected(torch.tensor(expected_features)).squeeze()
        loss = criterion(predictions, torch.tensor(expected_labels))
        loss.backward()
        optimizer.step()
        expected_history.append(loss.item())

    trained, history = model.train_model(features, labels, epochs, seed)

    assert history == pytest.approx(expected_history, rel=0, abs=0)
    for actual, reference in zip(trained.parameters(), expected.parameters(), strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize(("variant", "feature_count"), [("normal", 33), ("multifrac", 35)])
def test_checkpoint_round_trip_preserves_single_row_prediction(tmp_path, variant, feature_count):
    network = model.SimpleNN(feature_count)
    features = np.ones((1, feature_count), dtype=np.float32)
    path = tmp_path / "checkpoint.pth"

    model.save_checkpoint(network, variant, path)
    loaded, loaded_variant = model.load_checkpoint(path)

    assert loaded_variant == variant
    assert not loaded.training
    expected = model.predict(network, features)
    assert expected.shape == (1,)
    np.testing.assert_array_equal(model.predict(loaded, features), expected)


def test_checkpoint_rejects_weights_for_a_different_variant(tmp_path):
    path = tmp_path / "mismatched.pth"
    torch.save(
        {
            "version": model.CHECKPOINT_VERSION,
            "variant": "normal",
            "state_dict": model.SimpleNN(34).state_dict(),
        },
        path,
    )

    with pytest.raises(ValueError, match="Invalid checkpoint parameter: layer1.weight"):
        model.load_checkpoint(path)


def test_predict_rejects_wrong_dimensions():
    with pytest.raises(ValueError, match="Expected 33 features"):
        model.predict(model.SimpleNN(33), np.ones((1, 34), dtype=np.float32))
