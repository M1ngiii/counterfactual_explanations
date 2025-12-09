from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import nn

from carla import MLModel 
from ..datasets.adult import AdultData

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "data" / "models" / "black-box" / "adult_ann.pt"


class SimpleANN(nn.Module):
    """ Must match the architecture used in train_adult_ann.py """

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
        return self.net(x)


class AdultANNModel(MLModel):
    """ CARLA MLModel wrapper around our PyTorch ANN. Expects inputs in encoded feature space """

    def __init__(self, data: AdultData):
        super().__init__(data)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # feature order
        self._feature_input_order: List[str] = data.encoded_feature_names

        # load checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=self.device)
        input_dim = checkpoint["input_dim"]

        self._model = SimpleANN(input_dim)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(self.device)
        self._model.eval()

    # properties required by CARLA
    @property
    def feature_input_order(self) -> List[str]:
        return self._feature_input_order

    @property
    def backend(self) -> str:
        return "pytorch"

    @property
    def raw_model(self) -> nn.Module:
        return self._model

    # prediction API
    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.detach().to(self.device)

        if isinstance(x, pd.DataFrame):
            x_ordered = self.get_ordered_features(x)
            X = x_ordered.values.astype(np.float32)
            return torch.tensor(X, device=self.device)

        if isinstance(x, np.ndarray):
            return torch.tensor(x.astype(np.float32), device=self.device)

        # fallback
        X = np.asarray(x, dtype=np.float32)
        return torch.tensor(X, device=self.device)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """ Return hard class predictions 0/1 """
        X_tensor = self._to_tensor(x)
        with torch.no_grad():
            logits = self._model(X_tensor).view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
        return (probs >= 0.5).astype(int)

    def predict_proba(self, x) -> np.ndarray:
        """
        Return probabilities for both classes.
        If called with a torch.Tensor, return a torch.Tensor.
        Otherwise, return a numpy array.
        """
        import torch

        # detect original input type BEFORE _to_tensor
        input_is_tensor = isinstance(x, torch.Tensor)

        X_tensor = self._to_tensor(x)

        with torch.no_grad():
            logits = self._model(X_tensor).view(-1)
            probs_pos = torch.sigmoid(logits)  # this is a tensor

        if input_is_tensor:
            # CLUE path: keep everything as tensors
            probs_neg = 1.0 - probs_pos
            return torch.stack([probs_neg, probs_pos], dim=1)

        # normal path: return numpy
        probs_pos_np = probs_pos.cpu().numpy()
        probs = np.vstack([1.0 - probs_pos_np, probs_pos_np]).T
        return probs
