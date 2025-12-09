import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = ROOT / "data" / "adult" / "adult.csv"
MODEL_SAVE_PATH = ROOT / "data" / "models" / "black-box" / "adult_ann.pt"

TARGET = "income"


class SimpleANN(nn.Module):
    """ANN with 2 hidden layers + ReLU, binary output"""

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)  # raw logits


def load_data():
    print(f"[info] Loading processed data from {PROCESSED_DATA_PATH}")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df.drop(columns=[TARGET]).values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=0,
        stratify=y,
    )

    print(f"[info] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"[info] Input dim:  {X_train.shape[1]}")
    return X_train, X_test, y_train, y_test


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")

    X_train, X_test, y_train, y_test = load_data()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

    input_dim = X_train.shape[1]
    model = SimpleANN(input_dim).to(device)

    # Use this because model outputs raw logits
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    n_epochs = 30

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        scheduler.step(avg_loss)

        # Quick eval
        model.eval()
        all_probs = []
        all_true = []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy().ravel()

                all_probs.append(probs)
                all_true.append(yb.numpy().ravel())

        y_prob = np.concatenate(all_probs)
        y_true = np.concatenate(all_true)

        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")  # in case of degenerate splits

        print(
            f"Epoch {epoch:02d} | "
            f"loss={avg_loss:.4f} | "
            f"acc={acc:.3f} | "
            f"auc={auc:.3f}"
        )

    # Save model
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
        },
        MODEL_SAVE_PATH,
    )
    print(f"[ok] Saved model to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
