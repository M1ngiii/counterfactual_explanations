from dataclasses import dataclass
from typing import Any, List

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


@dataclass
class AdultModelBundle:
    X: pd.DataFrame          # one-hot encoded features (numeric)
    y: pd.Series             # 0 / 1 labels
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    model: Any               # trained sklearn model
    feature_names: List[str]


def load_adult_model(
    test_size: float = 0.2,
    random_state: int = 0,
) -> AdultModelBundle:
    """Load Adult from OpenML, one-hot encode categoricals, and train an RF classifier."""
    adult = fetch_openml("adult", version=2, as_frame=True)
    df = adult.frame

    # Labels
    y = (df["class"] == ">50K").astype(int)
    X = df.drop(columns=["class"])

    # One-hot encode all non-numeric columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X_encoded = X_encoded.astype(float)

    feature_names = list(X_encoded.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    return AdultModelBundle(
        X=X_encoded,
        y=y,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model=model,
        feature_names=feature_names,
    )
