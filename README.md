# Breast Cancer Classifier

An end-to-end binary classification pipeline on the Breast Cancer Wisconsin
dataset, built to practice **professional ML engineering habits**: clean module
boundaries, a baseline before the neural net, leakage-free preprocessing,
reproducibility, and real evaluation metrics.

> This is a learning scaffold. The structure, signatures, and docstrings are
> written for you — the actual ML code lives behind `TODO` markers that you fill
> in yourself.

## The task

Predict whether a tumor is **malignant (0)** or **benign (1)** from 30 numeric
features computed from a digitized image of a breast mass. Because a false
negative (calling a malignant tumor benign) is costly, you'll care about
**recall**, not just accuracy.

## Project layout

```
breast-cancer-classifier/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py            # all hyperparameters & paths in one place
├── src/
│   ├── __init__.py
│   ├── data.py          # load, split, scale (fit scaler on TRAIN ONLY)
│   ├── model.py         # baseline (logistic reg) + neural net builders
│   ├── train.py         # orchestrates: data -> model -> fit -> save
│   └── evaluate.py      # metrics, confusion matrix, learning curves
├── tests/
│   └── test_data.py     # shape + no-leakage sanity checks
└── artifacts/           # saved models, plots, metrics (gitignored)
```

## Why it's structured this way

- **`config.py`** — one source of truth for every knob. When you run a
  hyperparameter sweep later, you change values here, not scattered literals.
- **`data.py` separate from `model.py`** — you can test data handling without
  touching the model, and swap datasets without rewriting training code.
- **Baseline in `model.py`** — always beat a simple model before trusting a
  complex one. If your neural net can't beat logistic regression, something's
  wrong.
- **`artifacts/` gitignored** — never commit generated models/plots; commit the
  *code that produces them*.

## Recommended order to fill it in

1. `data.py` — get `load_data()` and `split_and_scale()` working. Run the tests.
2. `model.py` — `build_baseline()` first, then `build_nn()`.
3. `train.py` — wire it together; train the baseline, then the net.
4. `evaluate.py` — confusion matrix, precision/recall, learning curves.
5. Compare the two models. Then try the stretch goals below.

## Usage (once implemented)

```bash
pip install -r requirements.txt
python -m src.train --model baseline
python -m src.train --model nn
python -m src.evaluate --model nn
pytest
```

## Stretch goals

- Add L2 regularization / dropout to the net; compare train vs. val curves for
  overfitting (ties back to your regularization notes).
- Run a small hyperparameter sweep (learning rate, layer sizes) driven by
  `config.py`.
- Log each run's config + metrics to a timestamped file in `artifacts/`.
- Add a `Makefile` with `make train`, `make eval`, `make test`.
