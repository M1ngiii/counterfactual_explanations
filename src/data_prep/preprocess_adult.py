import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib

ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = ROOT / "data" / "adult" / "adult_raw.csv"
PROCESSED_DATA_PATH = ROOT / "data" / "adult" / "adult.csv"
PREPROCESSOR_PATH = ROOT / "data" / "adult" / "adult_preprocessor.joblib"

CONTINUOUS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

TARGET = "income"

def make_preprocessor():
    ohe = OneHotEncoder(drop="first", handle_unknown="error", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), CONTINUOUS),
            ("cat", ohe, CATEGORICAL),
        ]
    )
    return preprocessor

def preprocess_adult():
    print("[info] Loading raw Adult dataset...")
    df = pd.read_csv(RAW_DATA_PATH)

    # Clean values
    df = df.replace("?", np.nan)
    df = df.dropna().reset_index(drop=True)

    # Binary target
    df[TARGET] = (df[TARGET] == ">50K").astype(int)

    X = df[CONTINUOUS + CATEGORICAL]
    y = df[TARGET].values

    # Preprocessing
    preprocessor = make_preprocessor()
    X_proc = preprocessor.fit_transform(X)

    # Train/test split in processed space
    X_train_proc, X_test_proc, y_train, y_test = train_test_split(
        X_proc,
        y,
        test_size=0.2,
        random_state=0,
        stratify=y,
    )

    # Feature names after processing
    num_features = CONTINUOUS
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_features = list(cat_encoder.get_feature_names(CATEGORICAL))
    feature_names = list(num_features) + cat_features

    # Build processed DataFrames
    df_train_proc = pd.DataFrame(X_train_proc, columns=feature_names)
    df_train_proc[TARGET] = y_train

    df_test_proc = pd.DataFrame(X_test_proc, columns=feature_names)
    df_test_proc[TARGET] = y_test

    # Save
    df_all_proc = pd.concat([df_train_proc, df_test_proc], ignore_index=True)
    df_all_proc.to_csv(PROCESSED_DATA_PATH, index=False)
    joblib.dump(preprocessor, PREPROCESSOR_PATH, protocol=4)

    print(f"[ok] Saved processed dataset to {PROCESSED_DATA_PATH}")
    print(f"[ok] Saved preprocessor to {PREPROCESSOR_PATH}")

    print(f"[info] Final feature count: {len(feature_names)}")
    print(f"[info] Rows: {len(df_all_proc)}")


if __name__ == "__main__":
    preprocess_adult()
