"""Training entry point.

Run with:
    python -m src.train --model baseline
    python -m src.train --model nn

This file WIRES the pipeline together: set seeds -> load data -> build model ->
fit -> save. The argument parsing and reproducibility setup are done for you;
the TODOs are the parts that exercise what you've learned.
"""

import argparse
import random

import numpy as np

from config import CONFIG
from src.data import load_data, split_and_scale
from src.model import build_baseline, build_nn


def set_seeds(seed):
    """Make runs reproducible. (TensorFlow is seeded lazily inside train_nn.)"""
    random.seed(seed)
    np.random.seed(seed)


def train_baseline(data, config):
    """Fit the logistic-regression baseline and return the fitted model.

    TODO:
      - model = build_baseline(config)
      - model.fit(data.X_train, data.y_train)
      - print train and validation accuracy (model.score(...))
      - return model
    """
    model = build_baseline(config)
    model.fit(data.X_train, data.y_train)
    print("Train accuracy:", model.score(data.X_train, data.y_train))
    print("Validation accuracy:", model.score(data.X_val, data.y_val))
    return model


def train_nn(data, config):
    """Build, train, and return the neural network + its Keras History.

    TODO:
      - import tensorflow as tf; tf.random.set_seed(config.seed)
      - model = build_nn(input_dim=data.X_train.shape[1], config=config)
      - history = model.fit(
            data.X_train, data.y_train,
            validation_data=(data.X_val, data.y_val),
            epochs=config.epochs, batch_size=config.batch_size, verbose=2)
      - save the model to config.artifacts_dir / 'nn.keras'
      - return model, history   # history feeds the learning-curve plot
    """
    # TODO: implement
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="Train a breast cancer classifier.")
    parser.add_argument(
        "--model",
        choices=["baseline", "nn"],
        default="baseline",
        help="Which model to train.",
    )
    args = parser.parse_args()

    set_seeds(CONFIG.seed)

    X, y, feature_names = load_data()
    data = split_and_scale(X, y, feature_names, CONFIG)

    if args.model == "baseline":
        train_baseline(data, CONFIG)
    else:
        train_nn(data, CONFIG)


if __name__ == "__main__":
    main()
