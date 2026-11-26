"""EEGNet: Compact Convolutional Neural Network for EEG motor imagery decoding."""

from __future__ import annotations

import copy
from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class EEGNet(nn.Module):
    """Compact 2D Temporal-Spatial Convolutional Neural Network for EEG."""

    def __init__(
        self,
        n_classes: int = 2,
        n_channels: int = 64,
        n_times: int = 481,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_times = n_times

        # Block 1: Temporal Convolution
        self.conv1 = nn.Conv2d(
            1,
            F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 2: Spatial Depthwise Convolution
        self.depthwise = nn.Conv2d(
            F1,
            F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(p=dropout_rate)

        # Block 3: Separable Convolution
        self.sep_depthwise = nn.Conv2d(
            F1 * D,
            F1 * D,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=F1 * D,
            bias=False,
        )
        self.sep_pointwise = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(p=dropout_rate)

        # Dynamically determine linear layer input dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            features = self._forward_features(dummy)
            in_features = features.view(1, -1).size(1)

        self.classifier = nn.Linear(in_features, n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv1(x))
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._forward_features(x)
        flattened = features.view(features.size(0), -1)
        return self.classifier(flattened)


class EEGNetClassifier(BaseEstimator, ClassifierMixin):
    """scikit-learn compatible estimator wrapper for EEGNet."""

    def __init__(
        self,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout_rate: float = 0.5,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 16,
        patience: int = 15,
        val_split: float = 0.15,
        random_state: int = 42,
        device: str = "cpu",
    ) -> None:
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_split = val_split
        self.random_state = random_state
        self.device = device

    def _prepare_input(self, X: np.ndarray) -> torch.Tensor:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 3:
            # (n_trials, n_channels, n_times) -> (n_trials, 1, n_channels, n_times)
            arr = arr[:, None, :, :]
        elif arr.ndim != 4:
            raise ValueError(f"Expected 3D or 4D array, got shape {arr.shape}.")
        return torch.from_numpy(arr)

    def fit(self, X: Any, y: Any) -> EEGNetClassifier:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        classes, y_encoded = np.unique(y, return_inverse=True)
        self.classes_ = classes
        n_classes = len(classes)

        X_tensor = self._prepare_input(X)
        y_tensor = torch.from_numpy(y_encoded.astype(np.int64))

        n_samples = len(X_tensor)
        n_channels = X_tensor.shape[2]
        n_times = X_tensor.shape[3]

        self.model_ = EEGNet(
            n_classes=n_classes,
            n_channels=n_channels,
            n_times=n_times,
            F1=self.F1,
            D=self.D,
            F2=self.F2,
            kernel_length=self.kernel_length,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        # Train/Validation split for early stopping
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        val_size = max(1, int(n_samples * self.val_split)) if self.val_split > 0 and n_samples >= 10 else 0

        if val_size > 0:
            val_idx, train_idx = indices[:val_size], indices[val_size:]
            X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
            X_val, y_val = X_tensor[val_idx].to(self.device), y_tensor[val_idx].to(self.device)
        else:
            X_train, y_train = X_tensor, y_tensor
            X_val, y_val = None, None

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        best_val_loss = float("inf")
        best_weights = copy.deepcopy(self.model_.state_dict())
        patience_counter = 0

        self.model_.train()
        for epoch in range(self.epochs):
            self.model_.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model_(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            if X_val is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_out = self.model_(X_val)
                    val_loss = criterion(val_out, y_val).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = copy.deepcopy(self.model_.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        if X_val is not None:
            self.model_.load_state_dict(best_weights)

        self.model_.eval()
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise ValueError("Model is not fitted yet.")

        self.model_.eval()
        X_tensor = self._prepare_input(X).to(self.device)
        with torch.no_grad():
            logits = self.model_(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: Any) -> np.ndarray:
        probs = self.predict_proba(X)
        class_indices = np.argmax(probs, axis=1)
        return self.classes_[class_indices]


def make_eegnet_classifier(
    epochs: int = 80,
    batch_size: int = 16,
    lr: float = 1e-3,
    dropout_rate: float = 0.5,
    random_state: int = 42,
    device: str = "cpu",
) -> EEGNetClassifier:
    """Helper to construct an EEGNetClassifier with standardized defaults."""
    return EEGNetClassifier(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        dropout_rate=dropout_rate,
        random_state=random_state,
        device=device,
    )
