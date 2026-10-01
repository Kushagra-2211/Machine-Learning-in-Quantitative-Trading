"""
ML momentum strategy #4: LSTM (PyTorch).

Not from Peng Liu's book -- same rationale as pairs_trading/ml_lstm.py
(which cites Sarmento & Horta (2020) for using an LSTM to forecast a
financial time series from a sequence rather than a single day's snapshot):
here the LSTM sees a SEQUENCE of the past SEQ_LEN days' momentum features
rather than a flat single-day vector, potentially picking up on short-term
dynamics a flat feature vector discards.

Same features as the other models (ml_common.py), same trading rule
(signals.py via ml_common.finish_strategy) once a predicted return is
produced -- only the model architecture and how the data is shaped
(sequences vs flat rows) differs. Architecture and training procedure
mirror pairs_trading/ml_lstm.py exactly.

NOTE: requires PyTorch (`pip install torch`).
"""

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
import ml_common

SEQ_LEN = 20  # how many past days of features the LSTM sees before predicting tomorrow's return
EPOCHS = 500          # generous ceiling for the FINAL model -- early stopping should trigger well before this
SEARCH_EPOCHS = 150    # reduced budget used only to COMPARE hyperparameter configs
LEARNING_RATE = 0.001  # default / fallback if train_model is called with a single config directly
HIDDEN_SIZE = 32       # default / fallback
VALIDATION_FRACTION = 0.2  # last 20% of the formation period, held out for early stopping + model selection
PATIENCE = 15              # stop after this many epochs with no validation improvement

HYPERPARAM_GRID = [
    {"hidden_size": 32, "lr": 0.001,  "weight_decay": 0.0,  "dropout": 0.0},
    {"hidden_size": 32, "lr": 0.001,  "weight_decay": 1e-4, "dropout": 0.1},
    {"hidden_size": 16, "lr": 0.001,  "weight_decay": 1e-4, "dropout": 0.1},
    {"hidden_size": 32, "lr": 0.0005, "weight_decay": 1e-3, "dropout": 0.2},
]


class ReturnLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=HIDDEN_SIZE, dropout_p=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, n_features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = self.dropout(h_n[-1])  # (batch, hidden_size) -- final hidden state
        return self.fc(last_hidden)          # (batch, 1)


def _train_single_config(X_tr_tensor, y_tr_tensor, X_val_tensor, y_val_tensor, n_features,
                          hidden_size, dropout, lr, weight_decay, epochs, patience,
                          random_state, label="", verbose=True):
    torch.manual_seed(random_state)
    model = ReturnLSTM(n_features=n_features, hidden_size=hidden_size, dropout_p=dropout)
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

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  LSTM{label} epoch {epoch + 1}/{epochs} -- train MSE: {train_loss.item():.6f}, "
                  f"val MSE: {val_loss:.6f} (best: {best_val_loss:.6f} @ epoch {best_epoch})")

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"  LSTM{label} early stopping at epoch {epoch + 1}: no validation improvement "
                      f"for {patience} epochs. Best val MSE {best_val_loss:.6f} @ epoch {best_epoch}.")
            break
    else:
        if verbose:
            print(f"  LSTM{label} trained the full {epochs} epochs without early stopping triggering. "
                  f"Best val MSE {best_val_loss:.6f} @ epoch {best_epoch}.")

    model.load_state_dict(best_state)
    return model, best_val_loss, best_epoch


def train_model(X_seq_train, y_seq_train, epochs=EPOCHS, search_epochs=SEARCH_EPOCHS, random_state=None,
                 validation_fraction=VALIDATION_FRACTION, patience=PATIENCE, grid=None):
    random_state = random_state if random_state is not None else config.RANDOM_SEED
    grid = grid if grid is not None else HYPERPARAM_GRID

    X_tr, y_tr, X_val, y_val = ml_common.train_validation_split_sequences(
        X_seq_train, y_seq_train, validation_fraction
    )
    X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    n_features = X_seq_train.shape[2]

    print(f"  LSTM hyperparameter search over {len(grid)} configs ({search_epochs}-epoch budget each)...")
    best_config, best_search_val_loss = None, float("inf")
    for i, cfg in enumerate(grid):
        print(f"  -- config {i + 1}/{len(grid)}: {cfg}")
        _, val_loss, _ = _train_single_config(
            X_tr_tensor, y_tr_tensor, X_val_tensor, y_val_tensor, n_features,
            hidden_size=cfg["hidden_size"], dropout=cfg["dropout"],
            lr=cfg["lr"], weight_decay=cfg["weight_decay"], epochs=search_epochs,
            patience=patience, random_state=random_state, label=f" [search {i + 1}]",
        )
        if val_loss < best_search_val_loss:
            best_search_val_loss = val_loss
            best_config = cfg

    print(f"  LSTM best config from search: {best_config} (val MSE {best_search_val_loss:.6f}). "
          f"Retraining with full {epochs}-epoch budget...")

    model, best_val_loss, best_epoch = _train_single_config(
        X_tr_tensor, y_tr_tensor, X_val_tensor, y_val_tensor, n_features,
        hidden_size=best_config["hidden_size"], dropout=best_config["dropout"],
        lr=best_config["lr"], weight_decay=best_config["weight_decay"], epochs=epochs,
        patience=patience, random_state=random_state, label=" [final]",
    )
    print(f"  LSTM final model: best val MSE {best_val_loss:.6f} @ epoch {best_epoch}.")
    return model


def predict(model, X_seq):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        predictions = model(X_tensor).numpy().flatten()
    return predictions


def run_strategy(full_price, formation_end=None, trading_start=None, trading_end=None, seq_len=SEQ_LEN,
                  include_regime_features=False):
    formation_end = formation_end or config.FORMATION_END
    trading_start = trading_start or config.TRADING_START
    trading_end = trading_end or config.TRADING_END

    full_return = full_price.pct_change()
    full_features = ml_common.build_features(full_price, include_regime_features=include_regime_features)
    sequence_target = ml_common.build_sequence_target(full_price)

    X_seq, y_seq, target_dates = ml_common.build_sequences(full_features, sequence_target, seq_len)

    is_formation = target_dates <= pd.Timestamp(formation_end)
    is_trading = (target_dates >= pd.Timestamp(trading_start)) & (target_dates <= pd.Timestamp(trading_end))

    print(f"LSTM: {is_formation.sum()} formation-period sequences, {is_trading.sum()} trading-period sequences.")

    X_seq_train_scaled, (X_seq_trading_scaled,), _ = ml_common.standardize_sequences(
        X_seq[is_formation], [X_seq[is_trading]]
    )

    model = train_model(X_seq_train_scaled, y_seq[is_formation])
    predictions = predict(model, X_seq_trading_scaled)
    predicted_return = pd.Series(predictions, index=target_dates[is_trading]).sort_index()

    result = ml_common.finish_strategy_auto(
        predicted_return, full_return, trading_start, trading_end, y_seq[is_formation],
    )
    result["model"] = model
    return result


if __name__ == "__main__":
    import data_utils
    prices = data_utils.load_wide_prices()
    coin = config.COINS[0]
    result = run_strategy(prices[coin])
    cumret = (1 + result["daily_returns"]).cumprod()
    print(f"LSTM strategy smoke test on {coin}: "
          f"terminal cumulative return = {cumret.iloc[-1]:.4f}")
