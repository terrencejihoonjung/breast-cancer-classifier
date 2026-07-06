"""Sanity tests for the data pipeline.

These are deliberately small — the point is to build the habit of writing a test
that would CATCH a real bug. The leakage test below is the valuable one: it
fails loudly if you accidentally fit the scaler on the full dataset.

Run with:  pytest
"""

import numpy as np
import pytest

from config import CONFIG
from src.data import load_data, split_and_scale


@pytest.fixture
def data():
    X, y, _ = load_data()
    return split_and_scale(X, y, CONFIG)


def test_shapes_line_up(data):
    """Every X split has matching y length and the same feature count."""
    assert data.X_train.shape[0] == data.y_train.shape[0]
    assert data.X_val.shape[0] == data.y_val.shape[0]
    assert data.X_test.shape[0] == data.y_test.shape[0]
    assert data.X_train.shape[1] == data.X_test.shape[1] == data.X_val.shape[1]


def test_no_split_overlap(data):
    """Splits should be disjoint in size — all samples accounted for once."""
    total = data.X_train.shape[0] + data.X_val.shape[0] + data.X_test.shape[0]
    # Breast Cancer Wisconsin has 569 samples.
    assert total == 569


def test_labels_are_binary(data):
    """y should only contain 0 and 1."""
    for split in (data.y_train, data.y_val, data.y_test):
        assert set(np.unique(split)).issubset({0, 1})


def test_scaler_fit_on_train_only(data):
    """No leakage: training features are standardized (mean ~0, std ~1).

    Because the scaler is fit on the training set, X_train columns should have
    (approximately) zero mean and unit variance. The TEST set, scaled with the
    TRAINING statistics, should NOT be perfectly standardized — if X_test also
    has exactly mean 0 / std 1, you fit the scaler on the wrong data.
    """
    train_means = data.X_train.mean(axis=0)
    train_stds = data.X_train.std(axis=0)
    assert np.allclose(train_means, 0, atol=1e-6), "train features should be centered"
    assert np.allclose(train_stds, 1, atol=1e-6), "train features should be unit-variance"

    # Test set should be close-ish but not exactly standardized.
    test_means = np.abs(data.X_test.mean(axis=0))
    assert test_means.max() > 1e-6, (
        "Test set is perfectly centered — you likely fit the scaler on all data "
        "(data leakage). Fit on X_train only."
    )
