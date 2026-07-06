"""Data loading, splitting, and scaling.

Design rule you must respect: **fit the scaler on the training set ONLY**, then
apply that same fitted scaler to validation and test. Fitting on all the data
leaks information from the test set into training and inflates your scores. This
module is structured to make that mistake hard to commit.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
    """A container for the fully-prepared, scaled splits.

    Keeping the splits in one typed object (instead of returning a 6-tuple)
    means callers can't accidentally mix up the order of X_val and y_test.
    """

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list


def load_data():
    """Load the Breast Cancer Wisconsin dataset.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)   # 0 = malignant, 1 = benign
    feature_names : list[str]

    TODO:
      - Import `load_breast_cancer` from `sklearn.datasets`.
      - Call it with `return_X_y=False` so you also get `.feature_names`.
      - Return X, y, and the feature names as a list.
    """
    # TODO: implement
    raise NotImplementedError


def split_and_scale(X, y, config):
    """Split into train/val/test and standardize features (fit on train only).

    Parameters
    ----------
    X, y : arrays from `load_data`
    config : Config  (see config.py — use test_size, val_size, seed, stratify)

    Returns
    -------
    Dataset

    TODO:
      1. First split off the TEST set:
         train_test_split(X, y, test_size=config.test_size,
                          random_state=config.seed,
                          stratify=y if config.stratify else None)
      2. Split the remaining train portion again into train/val using
         config.val_size. (Remember: val_size is a fraction of what's LEFT.)
      3. Create ONE StandardScaler. Call .fit_transform on X_train, then
         .transform (NOT fit_transform) on X_val and X_test.
      4. Return a Dataset(...) with all six arrays + feature_names.

    Why this matters: step 3 is the classic data-leakage trap. The test in
    tests/test_data.py checks that your test set was scaled with the training
    statistics, not its own.
    """
    # TODO: implement
    raise NotImplementedError
