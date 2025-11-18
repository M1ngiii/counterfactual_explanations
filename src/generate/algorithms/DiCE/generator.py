from typing import List, Optional
import pandas as pd
import dice_ml
from dice_ml import Dice


class DiCEGenerator:
    """Generates counterfactuals on Adult using dice-ml."""
    def __init__(
        self,
        bundle,
        num_cf: int = 1,
        desired_class: str = "opposite",
    ):
        # bundle comes from main_generate.py
        self.bundle = bundle

        self.X = bundle.X
        self.y = bundle.y
        self.X_train = bundle.X_train
        self.X_test = bundle.X_test
        self.y_train = bundle.y_train
        self.y_test = bundle.y_test
        self.model = bundle.model
        self.feature_names = bundle.feature_names

        self.num_cf = num_cf
        self.desired_class = desired_class

        # Prepare data for DiCE
        df_train = self.X_train.copy()
        df_train["target"] = self.y_train.values

        self.dice_data = dice_ml.Data(
            dataframe=df_train,
            continuous_features=self.feature_names,
            outcome_name="target",
        )

        self.dice_model = dice_ml.Model(
            model=self.model,
            backend="sklearn",
        )

        self.dice = Dice(
            self.dice_data,
            self.dice_model,
            method="genetic",
        )

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
        """Generate counterfactuals for up to n_factuals factual instances."""
        y_pred = self.model.predict(self.X_test)

        # Generate CFs for class 0 (negative)
        mask_neg = y_pred == 0
        X_candidates = self.X_test[mask_neg]

        if len(X_candidates) == 0:
            raise RuntimeError("No negative predictions found to generate CFs for.")

        factuals = X_candidates.sample(
            min(n_factuals, len(X_candidates)),
            random_state=0,
        )

        records: List[dict] = []

        for idx, row in factuals.iterrows():
            factual_df = row.to_frame().T

            cf_expl = self.dice.generate_counterfactuals(
                factual_df,
                total_CFs=self.num_cf,
                desired_class=self.desired_class,
            )

            cf_df = cf_expl.cf_examples_list[0].final_cfs_df

            if cf_df.empty:
                continue

            if "target" in cf_df.columns:
                cf_df = cf_df.drop(columns=["target"])

            # Take first CF for now
            cf_row = cf_df.iloc[0]

            orig_text = self._row_to_text(row)
            cf_text = self._row_to_text(cf_row)

            records.append(
                {
                    "orig_text": orig_text,
                    "cf_text": cf_text,
                    "algorithm": "DiCE",
                    "dataset": "adult",
                }
            )

        return pd.DataFrame(records)
