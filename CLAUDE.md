# CLAUDE.md

Context for Claude Code working in this repository.

## What this project is

An end-to-end binary classification pipeline on the Breast Cancer Wisconsin
dataset (predict malignant=0 / benign=1 from 30 numeric features). It is a
**learning scaffold**: the structure, function signatures, docstrings, config,
and tests are already written; the actual ML logic lives behind `TODO` markers.

## MOST IMPORTANT: this is a learning exercise

**Do not fill in the `TODO` blocks for me.** I am writing the ML code myself to
practice. When I'm working on a function:

- Explain concepts, point me at the right API, and describe the approach.
- Review code I've written and point out bugs, leakage, or bad practice.
- Help me debug errors and interpret results.
- **Do not** write the implementation inside a `TODO` unless I explicitly say
  "write it for me" or "just implement this."

If you think a `TODO` is unclear, ask me how I'd approach it rather than solving
it. Treat yourself as a pair-programming mentor, not an autocomplete.

## My background

I've completed supervised learning fundamentals: linear/logistic regression,
gradient descent, regularization, feature scaling/engineering, and basic neural
nets in TensorFlow/Keras (Sequential API, activations, loss/compile/fit). I have
NOT yet done: CNNs, RNNs, or PyTorch. Explain anything beyond that scope when it
comes up.

## Stack

- Python 3, NumPy, scikit-learn (dataset, splitting, scaling, metrics),
  TensorFlow/Keras (the neural net), matplotlib (plots), pytest (tests).
- Install: `pip install -r requirements.txt`

## Structure

```
config.py            # all hyperparameters & paths (single source of truth)
src/data.py          # load_data(), split_and_scale() -> Dataset
src/model.py         # build_baseline() [logistic reg], build_nn() [Keras]
src/train.py         # CLI entry: python -m src.train --model {baseline,nn}
src/evaluate.py      # metrics, confusion matrix, learning curves
tests/test_data.py   # shape + no-leakage sanity checks
artifacts/           # generated models/plots/metrics (gitignored)
```

## Commands

```bash
python -m src.train --model baseline
python -m src.train --model nn
python -m src.evaluate --model nn
pytest
```

## Conventions & design rules (enforce these in review)

- **No data leakage.** The scaler is fit on the training set ONLY, then applied
  to val/test. `tests/test_data.py::test_scaler_fit_on_train_only` guards this.
- **Baseline before neural net.** Logistic regression must be trained and beaten
  before trusting the net. If the net can't beat it, suspect a bug.
- **Recall matters more than accuracy** — a false negative (malignant called
  benign) is the costly error. Always report the confusion matrix and recall,
  not just accuracy.
- **All hyperparameters live in `config.py`** — no scattered magic numbers.
- **Reproducibility** — seeds are set in `train.py::set_seeds` (and
  `tf.random.set_seed` inside the NN training). Don't remove them.
- **`artifacts/` is gitignored** — commit code that produces outputs, never the
  outputs.
- ReLU in hidden layers, sigmoid on the output; binary cross-entropy loss.

## Suggested order of work

1. `src/data.py` → run `pytest` until green.
2. `src/model.py` (baseline first, then NN).
3. `src/train.py`.
4. `src/evaluate.py`.
5. Stretch: L2/dropout + train-vs-val curves, a small hyperparameter sweep,
   run logging to `artifacts/`.

## Progress

- [x] **Step 1 — `src/data.py`**: `load_data()` and `split_and_scale()` done;
      `pytest` green. Note: `split_and_scale` gained a `feature_names` parameter,
      so `train.py` and `tests/test_data.py` were updated to pass it through.
- [x] **Step 2 — `src/model.py`**: `build_baseline()` (logistic regression) and
      `build_nn()` (compiled Keras Sequential) done. Dropout layer deferred to the
      stretch goal (config `dropout`/`l2` default to 0).
- [x] **Step 3 — `src/train.py`**: `train_baseline()` done and verified (98.9%
      train / 97.8% val — the bar the NN must beat). `train_nn()` done: builds,
      fits with val data, saves to `artifacts/nn.keras`, returns `(model, history)`.
      Result: 99.7% train / 96.7% val — does NOT beat the baseline. Overfitting
      (train≫val, val loss creeping up past ~epoch 90) with `l2=0`/`dropout=0`.
      Regularization is the Step 5 fix.
- [ ] **Step 4 — `src/evaluate.py`**: confusion matrix + recall + learning
      curves, to *see* the overfitting rather than infer it from the epoch log.  ← next
- [ ] Step 5 — stretch goals (early stopping / L2 / dropout to close the gap)
