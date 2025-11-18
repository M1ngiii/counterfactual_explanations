from typing import List
import numpy as np
import pandas as pd
import warnings
import tensorflow as tf

from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog

from carla.recourse_methods import ActionableRecourse
from carla.recourse_methods import GrowingSpheres
from carla.recourse_methods import Face
from carla.recourse_methods import Dice
from carla.recourse_methods import Clue

warnings.filterwarnings("ignore", category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class CarlaGenerator:
    """Counterfactual Generator using Carla"""
    def __init__(
        self,
        dataset_name: str = "adult",
        model_name: str = "ann",
        method: str = "FACE",
        oversample: int = 1,
    ):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.method = method.upper()
        self.oversample = oversample

        # Load dataset + model from CARLA
        self.dataset = OnlineCatalog(self.dataset_name)
        self.model = MLModelCatalog(self.dataset, self.model_name, backend="pytorch")
        print("CARLA model backend:", self.model.backend)
        print("Oversample Factor:", self.oversample)

        # Pick recourse method
        if self.method == "GS":
            self.recourse = GrowingSpheres(self.model, {})

        elif self.method == "AR":
            self.recourse = ActionableRecourse(self.model, {})

        elif self.method == "DICE":
            self.recourse = Dice(self.model, {})

        elif self.method == "FACE":
            hyperparams = {
                "mode": "knn"    # knn / epsilon
                }
            self.recourse = Face(self.model, hyperparams)

        elif self.method == "CLUE":
            hyperparams = {
                "data_name": self.dataset_name,

                "train_vae": True,

                # VAE Architecture
                "width": 20,            # neurons per hidden layer
                "depth": 2,             # number of hidden layers
                "latent_dim": 12,       # latent space dimension (10–20 typical)

                # Training parameters
                "batch_size": 32,       # common defaults: 32 or 64
                "epochs": 30,           # 30 is enough for Adult; 50 improves quality
                "lr": 0.001,            # learning rate for VAE (1e-3 default)
                "early_stop": 7,        # stop if no improvement (5 epochs)
            }
            self.recourse = Clue(self.dataset, self.model, hyperparams)
        else:
            raise ValueError(f"Unknown CARLA algorithm: '{method}'")

        # Determine feature columns (exclude target)
        target = getattr(self.dataset, "target", None)
        cols = list(self.dataset.df.columns)
        if target in cols:
            cols.remove(target)
        self.feature_names = cols

    def _format_value(self, val):
        """Ensures consistent numeric formatting."""
        if isinstance(val, (int, float)):
            v = float(val)
            return str(int(v)) if v.is_integer() else f"{v:.3f}"
        return str(val)

    def _row_to_text(self, row: pd.Series) -> str:
        """Converts a tabular row into a simple readable text description."""
        parts = []
        for col, val in row.items():
            if isinstance(val, (int, float)):
                v = float(val)
                if v.is_integer():
                    val_str = str(int(v))
                else:
                    val_str = f"{v:.3f}"
            else:
                val_str = str(val)
            parts.append(f"{col}={val_str}")
        return ", ".join(parts)

    def generate(self, n_factuals: int = 50) -> pd.DataFrame:
        """Generate up to n_factuals counterfactuals."""
        df_all = self.dataset.df.copy()

        if self.method == "DICE":
            # Random, since DICE seems to hang 
            factuals = df_all.sample(n_factuals).reset_index(drop=True)
        else:
            # Normal path
            sample_size = min(len(df_all), n_factuals * self.oversample)
            factuals = df_all.sample(sample_size, random_state=0).reset_index(drop=True)

        # Get counterfactuals
        cfs = self.recourse.get_counterfactuals(factuals)

        # Normalize output format
        if not isinstance(cfs, pd.DataFrame):
            cfs = pd.DataFrame(cfs, columns=self.feature_names)

        # Align rows
        min_len = min(len(factuals), len(cfs))
        factuals = factuals.iloc[:min_len]
        cfs = cfs.iloc[:min_len]

        records: List[dict] = []

        for idx in range(min_len):
            cf_row = cfs.iloc[idx]

            if cf_row.isna().all():
                continue

            orig_row = factuals.iloc[idx]

            orig_text = self._row_to_text(orig_row[self.feature_names])
            cf_text = self._row_to_text(cf_row[self.feature_names])

            if orig_text == cf_text:
                continue

            records.append(
                {
                    "orig_text": orig_text,
                    "cf_text": cf_text,
                    "algorithm": self.method,
                    "dataset": self.dataset_name,
                }
            )

            if len(records) >= n_factuals:
                break

        return pd.DataFrame(records)