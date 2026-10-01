"""
ML pairs trading strategy #3: feedforward Neural Network (PyTorch).

Same setup as ml_random_forest.py / ml_svm.py (predicts tomorrow's spread
from the features in ml_common.py, same trading rule) -- only the model
differs.

Architecture originally followed Peng Liu, "Quantitative Trading Strategies
Using Python" (2023), Ch.10: 3 linear layers (input -> 64 -> 32 -> 1) with
ReLU activations, no activation on the output (linear output for
regression), MSE loss, Adam optimizer, full-batch (no mini-batching -- the
formation set here is much larger than the book's single-year toy example).

NOTE: requires PyTorch (`pip install torch`). I could not test-run this file
myself -- my sandbox couldn't install PyTorch's full dependency chain over
its restricted network. Please run this one yourself first and let me know
if anything errors.

CHANGE FROM THE FIRST DRAFT: that version trained for a fixed EPOCHS count
and only checked whether the TRAINING loss had plateaued. Running it showed
why that's the wrong criterion -- going from 100 to 150 epochs pushed
training MSE down into the single digits on several pairs, but actually made
the TRADING-period Sharpe worse on most pairs.

Fix #1: hold out the last VALIDATION_FRACTION of the formation period as a
validation slice, track validation loss each epoch, and use early stopping.
Running THAT showed something more interesting: validation loss was still
falling at epoch 500 on every single pair -- no overfitting signature at all
within the formation period. So the gap wasn't training-time overfitting,
it's that formation-period patterns don't fully carry over to the trading
period (a regime-shift / non-stationarity issue -- see walk_forward.py for
the fix aimed at THAT).

Fix #2 (this version): the architecture and optimizer settings were still
just hand-picked from the book with no regularization at all, which is a bit
reckless for ~1,000-1,300 formation rows and a few thousand parameters. This
version adds:
  - Dropout (applied after each hidden layer's ReLU).
  - L2 weight decay (via Adam's weight_decay).
  - A small hyperparameter search over hidden layer sizes / learning rate /
    weight decay / dropout, using the SAME validation split as early
    stopping -- each candidate config gets a reduced-epoch budget
    (SEARCH_EPOCHS) to compare validation MSE, and the winner is retrained
    with the full EPOCHS ceiling for the final model. This is real model
    selection rather than a guessed architecture, and it's what a
    dissertation reviewer would expect to see before concluding the NN
    "doesn't work" here.
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

# ===================== HYPERPARAMETER GRID =====================
# Deliberately small (4 configs) to keep runtime reasonable -- each is
# evaluated with SEARCH_EPOCHS epochs before the winner gets the full
# EPOCHS budget. Spans: architecture size (64/32 vs a smaller 32/16),
# learning rate, and regularization strength (weight_decay = L2 penalty,
# dropout = fraction of hidden units zeroed each forward pass in training).
# =================================================================
HYPERPARAM_GRID = [
    {"hidden1": 64, "hidden2": 32, "lr": 0.001,  "weight_decay": 0.0,  "dropout": 0.0},
    {"hidden1": 64, "hidden2": 32, "lr": 0.001,  "weight_decay": 1e-4, "dropout": 0.1},
    {"hidden1": 32, "hidden2": 16, "lr": 0.001,  "weight_decay": 1e-4, "dropout": 0.1},
    {"hidden1": 64, "hidden2": 32, "lr": 0.0005, "weight_decay": 1e-3, "dropout": 0.2},
]


# ===================== NEURAL NETWORK ARCHITECTURE =====================
# 3 linear layers (input -> hidden1 -> hidden2 -> 1) with ReLU + dropout in
# between, no activation on the output (this is regression, predicting a
# continuous spread value, not a 0/1 classification). hidden1/hidden2/
# dropout_p are now configurable so the hyperparameter search can vary them.
# =========================================================================
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
    """One early-stopped training run for a single hyperparameter config.
    Returns (model, best_val_loss, best_epoch) with the model's weights
    already restored to whichever epoch had the lowest validation MSE."""
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
    """Hyperparameter search (see HYPERPARAM_GRID) + early-stopped final
    training, all using the same chronological train/validation split
    within the formation period -- see module docstring for why this
    replaced a single hand-picked architecture."""
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
    model.eval()  # IMPORTANT now that dropout exists -- eval() disables it at inference time;
                  # forgetting this would randomly zero hidden units on every prediction.
    with torch.no_grad():
        X_tensor = torch.tensor(features.values, dtype=torch.float32)
        predictions = model(X_tensor).numpy().flatten()
    return predictions


def run_strategy(full_price1, full_price2, formation_end=None, trading_start=None, trading_end=None):
    data = ml_common.prepare_data(full_price1, full_price2, formation_end, trading_start, trading_end)

    # Guard against a walk-forward window where this pair's trading-period
    # features are entirely NaN (data gap in one leg) -- see
    # ml_common.trading_features_usable()'s docstring. Must be checked
    # BEFORE standardize_features(), since StandardScaler.transform() also
    # raises on a zero-row input.
    if not ml_common.trading_features_usable(data["trading_features"]):
        predicted_spread = ml_common.no_signal_result("Neural Net")
        result = ml_common.finish_strategy(
            full_price1, full_price2, predicted_spread, data["full_spread"],
            data["trading_start"], data["trading_end"],
        )
        result["model"] = None
        return result

    # Gradient-trained models are sensitive to input scale -- standardize
    # using formation-period statistics only (see ml_common.standardize_features).
    trading_X_raw = data["trading_features"].dropna()
    train_X_scaled, (trading_X_scaled,), _ = ml_common.standardize_features(data["train_X"], [trading_X_raw])

    model = train_model(train_X_scaled, data["train_y"])
    predicted_spread = pd.Series(predict(model, trading_X_scaled), index=trading_X_scaled.index)

    result = ml_common.finish_strategy(
        full_price1, full_price2, predicted_spread, data["full_spread"],
        data["trading_start"], data["trading_end"],
    )
    result["model"] = model
    return result


if __name__ == "__main__":
    import data_utils
    prices = data_utils.load_wide_prices()
    permnos = data_utils.get_sector_permnos()
    p1, p2 = permnos[0], permnos[1]
    result = run_strategy(prices[p1], prices[p2])
    cumret = (1 + result["daily_returns"]).cumprod()
    print(f"Neural Network strategy smoke test on PERMNO {p1}/{p2}: "
          f"terminal cumulative return = {cumret.iloc[-1]:.4f}")
