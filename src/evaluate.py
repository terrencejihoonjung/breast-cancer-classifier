"""Evaluation: don't stop at accuracy.

On this dataset a false negative (predicting benign for a malignant tumor) is
the dangerous error, so you care about **recall** and the **confusion matrix**,
not just overall accuracy.
"""

import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)


def predict_labels(model, X, threshold):
    """Turn a model's probability output into 0/1 labels.

    Works for both the sklearn baseline and the Keras net.

    TODO:
      - If the model has `predict_proba` (sklearn): take column 1.
        Otherwise (Keras): use `model.predict(X)` and flatten.
      - Return (proba >= threshold).astype(int).
    """
    
    if hasattr(model, 'predict_proba'):
      return (model.predict_proba(X)[:, 1] >= threshold).astype(int)
    else:
      return (model.predict(X).flatten() >= threshold).astype(int)


def print_metrics(y_true, y_pred):
    """Print accuracy, precision, recall, F1, and the confusion matrix.

    TODO:
      - Use sklearn.metrics: accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report.
      - Print them clearly labeled. Call out recall explicitly.
    """
    
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"Precision (malignant): {precision_score(y_true, y_pred, pos_label=0)}")
    print(f"Recall (malignant): {recall_score(y_true, y_pred, pos_label=0)}")
    print(f"F1 (malignant): {f1_score(y_true, y_pred, pos_label=0)}")

    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_true, y_pred))

    print("Classification report:")
    print(classification_report(y_true, y_pred, zero_division=0))



def plot_confusion_matrix(y_true, y_pred, save_path):
    """Save a confusion-matrix figure to save_path.

    TODO:
      - sklearn.metrics.ConfusionMatrixDisplay.from_predictions(...)
      - plt.savefig(save_path); plt.close()
    """
    # TODO: implement
    raise NotImplementedError


def plot_learning_curves(history, save_path):
    """Plot train vs. validation loss (and accuracy) over epochs.

    This is the plot that reveals overfitting — the gap between the train and
    val curves is exactly the high-variance picture from your under/overfitting
    notes.

    Parameters
    ----------
    history : the Keras History object returned by model.fit
    save_path : where to save the PNG

    TODO:
      - Read history.history['loss'] and ['val_loss'] (and accuracy).
      - Plot both curves on shared axes, label them, savefig, close.
      - Only applies to the neural net (the baseline has no epochs).
    """
    # TODO: implement
    raise NotImplementedError


def main():
    """Load a saved model and report metrics on the test set.

    TODO:
      - Reload the data (same split/seed as training so the test set matches).
      - Load the saved model from config.artifacts_dir.
      - preds = predict_labels(model, data.X_test, CONFIG.threshold)
      - print_metrics(data.y_test, preds)
      - plot_confusion_matrix(...) into artifacts/.
    """
    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    main()
