from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import joblib

from carla import Data

ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_PATH = ROOT / "data" / "adult" / "adult_raw.csv"
PREPROCESSOR_PATH = ROOT / "data" / "adult" / "adult_preprocessor.joblib"

RAW_CONTINUOUS: List[str] = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

RAW_CATEGORICAL: List[str] = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

RAW_IMMUTABLES: List[str] = [
    "race", 
    "sex", 
    "native-country",
]

TARGET = "income"


class AdultData(Data):
    def __init__(self):
        self._preprocessor = joblib.load(PREPROCESSOR_PATH)
        self.encoder = self._preprocessor.named_transformers_["cat"]  # Careful

        self.raw_categorical = RAW_CATEGORICAL
        self.raw_continuous = RAW_CONTINUOUS
        self.raw_immutables = RAW_IMMUTABLES

        # encoded feature names
        cat_encoder = self._preprocessor.named_transformers_["cat"]
        cat_features = list(cat_encoder.get_feature_names(self.raw_categorical))
        self._encoded_feature_names: List[str] = self.raw_continuous + cat_features

        # continuous columns
        self._continuous = [
            col for col in self._encoded_feature_names
            if col in self.raw_continuous
        ]

        # categoricals
        self._categorical = [
            col for col in self._encoded_feature_names
            if any(col.startswith(f"{raw}_") for raw in self.raw_categorical)
        ]

        # immutables
        self._immutables = [
            col for col in self._encoded_feature_names
            if any(col.startswith(f"{raw}_") for raw in self.raw_immutables)
        ]

        # load RAW data
        df_raw = pd.read_csv(RAW_DATA_PATH)
        df_raw = df_raw.replace("?", np.nan).dropna().reset_index(drop=True)
        df_raw[TARGET] = (df_raw[TARGET] == ">50K").astype(int)

        # keep raw around if you want it
        self._df_raw = df_raw

        from sklearn.model_selection import train_test_split
        df_train_raw, df_test_raw = train_test_split(
            df_raw,
            test_size=0.2,
            stratify=df_raw[TARGET],
            random_state=0,
        )

        self._df_train_raw = df_train_raw
        self._df_test_raw = df_test_raw

        # encode
        X_train_enc = self.transform(df_train_raw)
        X_test_enc = self.transform(df_test_raw)

        # attach target back
        self._df_train = X_train_enc.assign(**{TARGET: df_train_raw[TARGET].values})
        self._df_test = X_test_enc.assign(**{TARGET: df_test_raw[TARGET].values})

        # full encoded df
        self._df = pd.concat([self._df_train, self._df_test], axis=0).reset_index(drop=True)

    @property
    def continuous(self) -> List[str]:
        return self._continuous

    @property
    def categorical(self) -> List[str]:
        return self._categorical

    @property
    def immutables(self) -> List[str]:
        return self._immutables

    @property
    def target(self) -> str:
        return TARGET

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def df_train(self) -> pd.DataFrame:
        return self._df_train

    @property
    def df_test(self) -> pd.DataFrame:
        return self._df_test

    @property
    def df_raw(self) -> pd.DataFrame:
        return self._df_raw

    @property
    def encoded_feature_names(self) -> List[str]:
        return self._encoded_feature_names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X_raw = df[self.raw_continuous + self.raw_categorical]
        X_enc = self._preprocessor.transform(X_raw)
        return pd.DataFrame(
            X_enc,
            columns=self._encoded_feature_names,
            index=df.index,
        )

    def inverse_transform(self, df_proc: pd.DataFrame) -> pd.DataFrame:
        X = df_proc[self._encoded_feature_names].values

        num_len = len(self.raw_continuous)
        X_num = X[:, :num_len]
        X_cat = X[:, num_len:]

        num_scaler = self._preprocessor.named_transformers_["num"]
        cat_encoder = self._preprocessor.named_transformers_["cat"]

        X_num_raw = num_scaler.inverse_transform(X_num)
        X_cat_raw = cat_encoder.inverse_transform(X_cat)

        data = np.concatenate([X_num_raw, X_cat_raw], axis=1)
        cols = self.raw_continuous + self.raw_categorical
        return pd.DataFrame(data, columns=cols, index=df_proc.index)
