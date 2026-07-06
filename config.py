"""Central configuration.

Every hyperparameter and path lives here so experiments are reproducible and a
sweep means changing values in ONE place. Import `CONFIG` wherever you need it.
"""

from dataclasses import dataclass, field
from pathlib import Path

# Project root = the folder this file sits in.
ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"


@dataclass
class Config:
    # --- Reproducibility ---
    seed: int = 42

    # --- Data splitting ---
    test_size: float = 0.2          # fraction held out as the test set
    val_size: float = 0.2           # fraction of the *remaining* data used for validation
    stratify: bool = True           # keep class balance across splits

    # --- Neural network architecture ---
    hidden_units: tuple = (16, 8)   # one entry per hidden layer
    hidden_activation: str = "relu"
    output_activation: str = "sigmoid"

    # --- Training ---
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    l2: float = 0.0                 # L2 regularization strength (0 = off; turn on for stretch goal)
    dropout: float = 0.0            # dropout rate (0 = off)

    # --- Evaluation ---
    threshold: float = 0.5          # probability cutoff for the positive class

    # --- Paths ---
    artifacts_dir: Path = field(default_factory=lambda: ARTIFACTS_DIR)

    def __post_init__(self):
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
