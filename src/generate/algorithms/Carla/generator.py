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
    """ Counterfactual Generator using Carla """
    def __init__(
        self,
        dataset,
        model,
        dataset_name: str = "adult",
        model_name: str = "ann",
        method: str = "FACE",
        oversample: int = 1,
    ):

        if dataset is None or model is None:
            raise ValueError(
                "CarlaGenerator requires explicit dataset and model"
            )

        # Load dataset + model from CARLA
        self.dataset = dataset
        self.model = model

        self.method = method.upper()
        self.oversample = oversample
        self.dataset_name = dataset_name
        self.model_name = model_name

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

                "train_vae": False,

                # VAE Architecture
                "width": 20,            # neurons per hidden layer
                "depth": 2,             # number of hidden layers
                "latent_dim": 12,       # latent space dimension (10–20 typical)

                # Training parameters
                "batch_size": 32,       # common defaults: 32 or 64
                "epochs": 30,           # 30 is enough
                "lr": 0.001,            # learning rate for VAE
                "early_stop": 7,        # stop if no improvement
            }
            self.recourse = Clue(data=self.dataset, mlmodel=self.model, hyperparams=hyperparams)

        else:
            raise ValueError(f"Unknown CARLA algorithm: '{method}'")

        # Determine feature columns (exclude target)
        target = getattr(self.dataset, "target", None)
        cols = list(self.dataset.df.columns)
        if target in cols:
            cols.remove(target)
        self.feature_names = cols

    def _format_value(self, val):
        """ Ensures consistent numeric formatting """
        if isinstance(val, (int, float)):
            v = float(val)
            return str(int(v)) if v.is_integer() else f"{v:.3f}"
        return str(val)

    def _row_to_text(self, row: pd.Series) -> str:
        """ Converts a tabular row into a simple readable text description """
        parts = []
        for col, val in row.items():
            if isinstance(val, (int, np.integer)):
                val_str = str(val)

            elif isinstance(val, (float, np.floating)):
                # tolerate tiny floating-point noise
                if abs(val - round(val)) < 1e-4:
                    val_str = str(int(round(val)))
                else:
                    val_str = f"{val:.3f}"

            else:
                val_str = str(val)

            parts.append(f"{col}={val_str}")

        return ", ".join(parts)

    def generate(self, n_factuals: int = 50) -> pd.DataFrame:
        """Generate up to n_factuals counterfactuals"""

        # df is encoded
        df_all_enc = self.dataset.df.copy()

        # get feature cols
        feature_cols = self.dataset.encoded_feature_names

        # Randomly choose low income rows
        y_pred = self.model.predict(df_all_enc[feature_cols])
        df_low_income = df_all_enc[y_pred == 0]

        # pick factuals in encoded space
        sample_size = min(len(df_low_income), n_factuals * self.oversample)
        factuals_enc = df_low_income.sample(sample_size, random_state=0).reset_index(drop=True)

        X_factuals = factuals_enc[feature_cols]

        # get counterfactuals in encoded space
        cfs_enc = self.recourse.get_counterfactuals(X_factuals)

        if not isinstance(cfs_enc, pd.DataFrame):
            cfs_enc = pd.DataFrame(cfs_enc, columns=feature_cols)
        else:
            cfs_enc = cfs_enc.copy()

        # align lengths
        min_len = min(len(X_factuals), len(cfs_enc))
        X_factuals = X_factuals.iloc[:min_len].reset_index(drop=True)
        cfs_enc = cfs_enc.iloc[:min_len].reset_index(drop=True)

        # drop invalid CF rows
        valid_mask = ~cfs_enc.isna().all(axis=1)
        X_factuals = X_factuals[valid_mask].reset_index(drop=True)
        cfs_enc = cfs_enc[valid_mask].reset_index(drop=True)

        # map both factuals + CFs back to RAW for LLM / human
        orig_raw = self.dataset.inverse_transform(X_factuals)
        cf_raw = self.dataset.inverse_transform(cfs_enc)

        raw_feature_names = self.dataset.raw_continuous + self.dataset.raw_categorical

        records = []
        for i in range(min_len):
            orig_row = orig_raw.iloc[i]
            cf_row = cf_raw.iloc[i]

            if cf_row.isna().all():
                continue

            orig_text = self._row_to_text(orig_row[raw_feature_names])
            cf_text = self._row_to_text(cf_row[raw_feature_names])

            if orig_text == cf_text:
                continue

            records.append(
                {
                    "orig_text": orig_text,
                    "cf_text": cf_text,
                    "dataset": self.dataset_name,
                    "model": self.model_name,
                    "algorithm": self.method,
                }
            )

            if len(records) >= n_factuals:
                break

        return pd.DataFrame(records)
