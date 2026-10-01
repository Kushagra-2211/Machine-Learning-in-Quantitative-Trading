"""
ML pairs trading strategy #4: LSTM (PyTorch).

Not from Peng Liu's book (which doesn't cover sequence models) -- this one
follows Sarmento & Horta (2020), "Enhancing a Pairs Trading strategy with
the application of Machine Learning," who use an LSTM (among other models)
to forecast the spread. Unlike the flat-feature models (RF/SVM/NN), an LSTM
sees a SEQUENCE of the past SEQ_LEN days' features rather than a single
day's snapshot -- it can pick up on short-term dynamics/momentum in the
spread that a flat feature vector discards.

Same features as the other models (ml_common.py), same trading rule
(signals.py) once a predicted spread is produced -- only the model
architecture and how the data is shaped (sequences vs flat rows) differs.

NOTE: requires PyTorch (`pip install torch`). I could not test-run this file
myself -- my sandbox couldn't install PyTorch's full dependency chain over
its restricted network. Please run this one yourself first and let me know
if anything errors -- sequence-shaped code is the easiest place for a subtle
indexing bug to hide, so a second pair of eyes on the actual output (does
the predicted series line up with the right dates?) is worth doing before
trusting the results.

CHANGE FROM THE FIRST DRAFT: same fix applied to ml_neural_net.py, for
consistency across both gradient-trained models. The original version
trained a fixed EPOCHS count and only checked whether TRAINING loss had
plateaued -- at epoch 200 it hadn't, and this model's trading-period results
were the worst of all strategies. Adding a validation split + early stopping
showed early stopping DOES trigger for some pairs (unlike the NN, where
validation loss never plateaus even at epoch 500) -- so the LSTM's problem
isn't purely an epoch-count issue either.

CHANGE (this version): added dropout (on the LSTM's final hidden state,
before the output layer) and L2 weight decay, plus a small hyperparameter
search over hidden size / learning rate / regularization using the same
validation split -- same rationale as ml_neural_net.py: give the LSTM a fair,
tuned shot before concluding it doesn't add value here.
"""

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
import classical_strategy
import ml_common

SEQ_LEN = 20  # matches config.ZSCORE_WINDOW by default -- how many past days
              # of features the LSTM sees before predicting tomorrow's spread
EPOCHS = 500          # generous ceiling for the FINAL model -- early stopping should trigger well before this
SEARCH_EPOCHS = 150    # reduced budget used only to COMPARE hyperparameter configs
LEARNING_RATE = 0.001  # default / fallback if train_model is called with a single config directly
HIDDEN_SIZE = 32       # default / fallback
VALIDATION_FRACTION = 0.2  # last 20% of the formation period, held out for early stopping + model selection
PATIENCE = 15              # stop after this many epochs with no validation improvement

# ===================== HYPERPARAMETER GRID =====================
# Same idea as ml_neural_net.py's grid: small (4 configs) to keep runtime
# reasonable, each evaluated with SEARCH_EPOCHS before the winner gets the
# full EPOCHS budget. Spans hidden size, learning rate, weight decay (L2),
# and dropout applied to the LSTM's final hidden state.
# =================================================================
HYPERPARAM_GRID = [
    {"hidden_size": 32, "lr": 0.001,  "weight_decay": 0.0,  "dropout": 0.0},
    {"hidden_size": 32, "lr": 0.001,  "weight_decay": 1e-4, "dropout": 0.1},
    {"hidden_size": 16, "lr": 0.001,  "weight_decay": 1e-4, "dropout": 0.1},
    {"hidden_size": 32, "lr": 0.0005, "weight_decay": 1e-3, "dropout": 0.2},
]


# ===================== LSTM ARCHITECTURE =====================
# One LSTM layer reads a SEQUENCE of the past SEQ_LEN days' features (not
# just a single day like RF/SVM/NN), then dropout is applied to its final
# hidden state, then a linear layer maps that to a single predicted spread
# value. hidden_size/dropout_p are configurable for the hyperparameter search.
# ================================================================
class SpreadLSTM(nn.Module):
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
    """One early-stopped training run for a single hyperparameter config.
    Returns (model, best_val_loss, best_epoch) with the model's weights
    already restored to whichever epoch had the lowest validation MSE."""
    torch.manual_seed(random_state)
    model = SpreadLSTM(n_features=n_features, hidden_size=hidden_size, dropout_p=dropout)
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
    """Hyperparameter search (see HYPERPARAM_GRID) + early-stopped final
    training, all using the same chronological train/validation split
    within the formation period -- see module docstring."""
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
    model.eval()  # IMPORTANT now that dropout exists -- eval() disables it at inference time.
    with torch.no_grad():
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        predictions = model(X_tensor).numpy().flatten()
    return predictions


def run_strategy(full_price1, full_price2, formation_end=None, trading_start=None, trading_end=None, seq_len=SEQ_LEN):
    formation_end = formation_end or config.FORMATION_END
    trading_start = trading_start or config.TRADING_START
    trading_end = trading_end or config.TRADING_END

    formation_p1 = full_price1.loc[:formation_end]
    formation_p2 = full_price2.loc[:formation_end]
    const, beta = classical_strategy.fit_spread_model(formation_p1, formation_p2)

    full_spread = classical_strategy.compute_spread(full_price1, full_price2, const, beta)
    full_features = ml_common.build_features(full_price1, full_price2, full_spread)

    X_seq, y_seq, target_dates = ml_common.build_sequences(full_features, full_spread, seq_len)

    is_formation = target_dates <= pd.Timestamp(formation_end)
    is_trading = (target_dates >= pd.Timestamp(trading_start)) & (target_dates <= pd.Timestamp(trading_end))

    print(f"LSTM: {is_formation.sum()} formation-period sequences, {is_trading.sum()} trading-period sequences.")

    # Guard against a walk-forward window where this pair has ZERO usable
    # trading-period sequences (a data gap in one leg over this specific
    # date range means build_sequences()'s dropna+intersection drops every
    # trading-period date entirely) -- see
    # ml_common.trading_features_usable()'s docstring for the flat-model
    # equivalent of this same failure mode. Must be checked before
    # standardize_sequences(), since scaling a 0-row array raises too.
    if is_trading.sum() == 0:
        predicted_spread = ml_common.no_signal_result("LSTM")
        result = ml_common.finish_strategy(
            full_price1, full_price2, predicted_spread, full_spread,
            trading_start, trading_end,
        )
        result["model"] = None
        return result

    # LSTM training is gradient-based, so (like the feedforward net) it's
    # sensitive to input scale -- standardize using formation-sequence
    # statistics only (see ml_common.standardize_sequences).
    X_seq_train_scaled, (X_seq_trading_scaled,), _ = ml_common.standardize_sequences(
        X_seq[is_formation], [X_seq[is_trading]]
    )

    model = train_model(X_seq_train_scaled, y_seq[is_formation])
    predictions = predict(model, X_seq_trading_scaled)
    predicted_spread = pd.Series(predictions, index=target_dates[is_trading]).sort_index()

    result = ml_common.finish_strategy(
        full_price1, full_price2, predicted_spread, full_spread,
        trading_start, trading_end,
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
    print(f"LSTM strategy smoke test on PERMNO {p1}/{p2}: "
          f"terminal cumulative return = {cumret.iloc[-1]:.4f}")
