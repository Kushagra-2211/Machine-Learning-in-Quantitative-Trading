"""
ML momentum strategy #3: feedforward Neural Network (PyTorch).

Same setup as ml_random_forest.py / ml_svm.py (predicts tomorrow's return
from the features in ml_common.py, same trading rule) -- only the model
differs.

Architecture and training procedure mirror pairs_trading/ml_neural_net.py
exactly (3 linear layers, ReLU + dropout, Adam with L2 weight decay, a small
hyperparameter search over hidden sizes/learning rate/regularization using a
chronological validation split for early stopping) -- see that module's
docstring for the full rationale (an earlier fixed-epoch-count version
showed training loss improving well past the point where trading-period
performance stopped improving; this version adds dropout, weight decay, and
real model selection instead of a single hand-picked architecture).

NOTE: requires PyTorch (`pip install torch`).
"""

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
import ml_common

EPOCHS = 500          # generous ceiling for the FINAL model -- early stopping should trigger well before this
SEARCH_EPOCHS = 150    # reduced budget used only to COMPARE hyperparameter configs
LEARNING_RATE = 0.001  # default / fallback if train_model is called with a single config directly
VALIDATION_FRACTION = 0.2  # last 20% of the formation period, held out for early stopping + model selection
PATIENCE = 15              # stop after this many epochs with no validation improvement

HYPERPARAM_GRID = [
    {"hidden1": 64, "hidden2": 32, "lr": 0.001,  "weight_decay": 0.0,  "dropout": 0.0},
    {"hidden1": 64, "hidden2": 32, "lr": 0.001,  "weight_decay": 1e-4, "dropout": 0.1},
    {"hidden1": 32, "hidden2": 16, "lr": 0.001,  "weight_decay": 1e-4, "dropout": 0.1},
    {"hidden1": 64, "hidden2": 32, "lr": 0.0005, "weight_decay": 1e-3, "dropout": 0.2},
]


class FeedforwardNet(nn.Module):
    def __init__(self, n_features, hidden1=64, hidden2=32, dropout_p=0.0):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def _train_single_config(X_tr_tensor, y_tr_tensor, X_val_tensor, y_val_tensor, n_features,
                          hidden1, hidden2, dropout, lr, weight_decay, epochs, patience,
                          random_state, label="", verbose=True):
    torch.manual_seed(random_state)
    model = FeedforwardNet(n_features=n_features, hidden1=hidden1, hidden2=hidden2, dropout_p=dropout)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_predictions = model(X_tr_tensor)
        train_loss = criterion(train_predictions, y_tr_tensor)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_tensor), y_val_tensor).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch + 1) % 25 == 0:
            print(f"  NN{label} epoch {epoch + 1}/{epochs} -- train MSE: {train_loss.item():.6f}, "
                  f"val MSE: {val_loss:.6f} (best: {best_val_loss:.6f} @ epoch {best_epoch})")

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"  NN{label} early stopping at epoch {epoch + 1}: no validation improvement "
                      f"for {patience} epochs. Best val MSE {best_val_loss:.6f} @ epoch {best_epoch}.")
            break
    else:
        if verbose:
            print(f"  NN{label} trained the full {epochs} epochs without early stopping triggering. "
                  f"Best val MSE {best_val_loss:.6f} @ epoch {best_epoch}.")

    model.load_state_dict(best_state)
    return model, best_val_loss, best_epoch


def train_model(train_X, train_y, epochs=EPOCHS, search_epochs=SEARCH_EPOCHS, random_state=None,
                 validation_fraction=VALIDATION_FRACTION, patience=PATIENCE, grid=None):
    random_state = random_state if random_state is not None else config.RANDOM_SEED
    grid = grid if grid is not None else HYPERPARAM_GRID

    X_tr, y_tr, X_val, y_val = ml_common.train_validation_split_flat(train_X, train_y, validation_fraction)
    X_tr_tensor = torch.tensor(X_tr.values, dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_tr.values, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
    n_features = train_X.shape[1]

    print(f"  NN hyperparameter search over {len(grid)} configs ({search_epochs}-epoch budget each)...")
    best_config, best_search_val_loss = None, float("inf")
    for i, cfg in enumerate(grid):
        print(f"  -- config {i + 1}/{len(grid)}: {cfg}")
        _, val_loss, _ = _train_single_config(
            X_tr_tensor, y_tr_tensor, X_val_tensor, y_val_tensor, n_features,
            hidden1=cfg["hidden1"], hidden2=cfg["hidden2"], dropout=cfg["dropout"],
            lr=cfg["lr"], weight_decay=cfg["weight_decay"], epochs=search_epochs,
            patience=patience, random_state=random_state, label=f" [search {i + 1}]",
        )
        if val_loss < best_search_val_loss:
            best_search_val_loss = val_loss
            best_config = cfg

    print(f"  NN best config from search: {best_config} (val MSE {best_search_val_loss:.6f}). "
          f"Retraining with full {epochs}-epoch budget...")

    model, best_val_loss, best_epoch = _train_single_config(
        X_tr_tensor, y_tr_tensor, X_val_tensor, y_val_tensor, n_features,
        hidden1=best_config["hidden1"], hidden2=best_config["hidden2"], dropout=best_config["dropout"],
        lr=best_config["lr"], weight_decay=best_config["weight_decay"], epochs=epochs,
        patience=patience, random_state=random_state, label=" [final]",
    )
    print(f"  NN final model: best val MSE {best_val_loss:.6f} @ epoch {best_epoch}.")
    return model


def predict(model, features):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(features.values, dtype=torch.float32)
        predictions = model(X_tensor).numpy().flatten()
    return predictions


def run_strategy(full_price, formation_end=None, trading_start=None, trading_end=None,
                  include_regime_features=False):
    data = ml_common.prepare_data(full_price, formation_end, trading_start, trading_end,
                                   include_regime_features=include_regime_features)

    trading_X_raw = data["trading_features"].dropna()
    train_X_scaled, (trading_X_scaled,), _ = ml_common.standardize_features(data["train_X"], [trading_X_raw])

    model = train_model(train_X_scaled, data["train_y"])
    predicted_return = pd.Series(predict(model, trading_X_scaled), index=trading_X_scaled.index)

    result = ml_common.finish_strategy_auto(
        predicted_return, data["full_return"], data["trading_start"], data["trading_end"], data["train_y"],
    )
    result["model"] = model
    return result


if __name__ == "__main__":
    import data_utils
    prices = data_utils.load_wide_prices()
    coin = config.COINS[0]
    result = run_strategy(prices[coin])
    cumret = (1 + result["daily_returns"]).cumprod()
    print(f"Neural Network strategy smoke test on {coin}: "
          f"terminal cumulative return = {cumret.iloc[-1]:.4f}")
