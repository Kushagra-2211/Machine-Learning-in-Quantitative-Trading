"""
Shared plumbing used by EVERY ML pairs-trading variant (Random Forest, SVM,
Neural Network, LSTM -- see ml_random_forest.py, ml_svm.py, ml_neural_net.py,
ml_lstm.py). Keeping this in one place means the comparison across models
isolates just the choice of model: spread definition, features, the
formation/trading split, and the trading rule are all identical across the
four, so any performance difference is attributable to the model itself.

See ml_random_forest.py's module docstring (or DISSERTATION_PLAN.md) for the
full rationale behind the design choices here (lagged features predicting
TOMORROW's spread, same OLS-residual spread as the classical strategy, etc).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
import classical_strategy
import signals


def rolling_half_life(spread, window=60):
    """===================== ROLLING HALF-LIFE OF MEAN REVERSION =====================
    For each day, fits a simple Ornstein-Uhlenbeck-style regression of the
    spread's day-over-day CHANGE on its own lagged LEVEL over the trailing
    `window` days: delta_spread_t = alpha + theta * spread_{t-1} + noise.
    theta < 0 means the spread is behaving as mean-reverting over that
    window (the more negative it is below zero, the faster it pulls back);
    half_life = -ln(2)/theta converts that into "how many days until half
    the current deviation reverts", the standard OU half-life used in the
    pairs-trading literature.

    This is only ever computed from information available up to and
    including day t (a trailing window), so it's safe to use as a feature --
    it just wasn't in the original six-feature set, and gives the model a
    direct read on how strong/fast the mean-reversion currently looks,
    something none of the other features capture explicitly.

    Where theta >= 0 (spread looks non-mean-reverting over that window) we
    leave the value as NaN for that day rather than reporting a nonsensical
    half-life; forward-filling afterwards (see build_features) carries the
    last valid estimate forward instead of creating gaps mid-series.
    ==================================================================================="""
    delta = spread.diff()
    lagged = spread.shift(1)
    half_life = pd.Series(np.nan, index=spread.index)

    delta_vals = delta.values
    lagged_vals = lagged.values

    for i in range(window, len(spread)):
        y = delta_vals[i - window:i]
        x = lagged_vals[i - window:i]
        mask = ~(np.isnan(y) | np.isnan(x))
        if mask.sum() < window // 2:
            continue
        x_m, y_m = x[mask], y[mask]
        x_mean = x_m.mean()
        var_x = ((x_m - x_mean) ** 2).sum()
        if var_x == 0:
            continue
        theta = ((x_m - x_mean) * (y_m - y_m.mean())).sum() / var_x
        if theta >= 0:
            continue  # not mean-reverting over this window -- leave as NaN
        half_life.iloc[i] = -np.log(2) / theta

    return half_life


def build_features(price1, price2, spread, zscore_window=None, halflife_window=60):
    """===================== FEATURE ENGINEERING (all ML models) =====================
    Peng Liu Ch.10's original five (Asset1Returns, Asset2Returns, SpreadMA5,
    Asset1Volatility, Asset2Volatility) plus a richer set of spread-specific
    features added after the first backtest run showed the ML models barely
    beating a naive persistence forecast: SpreadLag1 alone gives a flat model
    almost nothing beyond "tomorrow looks like today", which is close to what
    the classical z-score strategy already exploits. The additions below give
    the model more than one lag to work with, a direct read on the current
    z-score and mean-reversion speed, and a measure of the spread's OWN
    volatility (distinct from each stock's return volatility):

      - SpreadLag1/2/3/5: multiple lags, not just one day back.
      - SpreadZScore: the same rolling z-score the classical strategy trades
        on, computed the identical way (see classical_strategy.compute_zscore)
        -- lets the model either lean into or deliberately deviate from the
        classical signal, rather than having to reconstruct it implicitly.
      - SpreadVolatility: rolling std of the spread's own day-over-day
        change -- a regime-volatility signal, separate from Asset1/2Volatility
        which only look at each stock's raw returns.
      - SpreadHalfLife: rolling OU half-life of mean reversion (see
        rolling_half_life() above).

    All computed using only information available up to and including day t
    -- this is the input every ML model (Random Forest, SVM, Neural Net,
    LSTM, Gradient Boosting) is trained on, so the SAME features feed all of
    them.
    ==================================================================================="""
    zscore_window = zscore_window if zscore_window is not None else config.ZSCORE_WINDOW

    ret1 = price1.pct_change()
    ret2 = price2.pct_change()

    spread_mean = spread.rolling(zscore_window).mean()
    spread_std = spread.rolling(zscore_window).std()

    features = pd.DataFrame({
        "Asset1Returns": ret1,
        "Asset2Returns": ret2,
        "SpreadMA5": spread.rolling(5).mean(),
        "Asset1Volatility": ret1.rolling(20).std(),
        "Asset2Volatility": ret2.rolling(20).std(),
        "SpreadLag1": spread.shift(1),
        "SpreadLag2": spread.shift(2),
        "SpreadLag3": spread.shift(3),
        "SpreadLag5": spread.shift(5),
        "SpreadZScore": (spread - spread_mean) / spread_std,
        "SpreadVolatility": spread.diff().rolling(20).std(),
        "SpreadHalfLife": rolling_half_life(spread, window=halflife_window).ffill(),
    })
    return features


def build_target(spread):
    """Target is TOMORROW's spread: shift(-1) so row t's features (known at
    end of day t) line up with the value being predicted (day t+1)."""
    return spread.shift(-1)


def prepare_data(full_price1, full_price2, formation_end=None, trading_start=None, trading_end=None):
    """Shared setup for all flat-feature models (RF, SVM, NN): fit the OLS
    spread model on the formation period only, build features/target over
    the full history, and split into formation (train) vs trading (predict)."""
    formation_end = formation_end or config.FORMATION_END
    trading_start = trading_start or config.TRADING_START
    trading_end = trading_end or config.TRADING_END

    formation_p1 = full_price1.loc[:formation_end]
    formation_p2 = full_price2.loc[:formation_end]
    const, beta = classical_strategy.fit_spread_model(formation_p1, formation_p2)

    full_spread = classical_strategy.compute_spread(full_price1, full_price2, const, beta)
    full_features = build_features(full_price1, full_price2, full_spread)
    full_target = build_target(full_spread)

    formation_features = full_features.loc[:formation_end]
    formation_target = full_target.loc[:formation_end]
    trading_features = full_features.loc[trading_start:trading_end]

    # Drop rows where rolling windows / lag haven't filled yet (only affects
    # the very start of the formation period in practice).
    train_data = pd.concat([formation_features, formation_target.rename("target")], axis=1).dropna()

    return {
        "const": const, "beta": beta,
        "full_spread": full_spread,
        "train_X": train_data.drop(columns="target"),
        "train_y": train_data["target"],
        "trading_features": trading_features,
        "trading_start": trading_start, "trading_end": trading_end,
    }


def trading_features_usable(trading_features):
    """===================== EMPTY-WINDOW GUARD =====================
    True if at least one row of the trading-period features survives
    .dropna() -- i.e. there's something to actually predict on.

    Added after a combined-sector run crashed on Information Technology:
    IBM-GEN's first walk-forward window (2022): every row of that pair's
    trading-period features was NaN (a data gap in one leg during that
    specific date range), so trading_features.dropna() left a (0, 12)
    array, and RandomForestRegressor.predict() -- like SVR.predict() and
    StandardScaler.transform() -- raises ValueError on a zero-row input
    rather than returning an empty result. This should be checked BEFORE
    any model-specific work (scaling, fitting/predicting), not just
    immediately before the final .predict() call, since scaling a 0-row
    array fails the same way.
    =================================================================="""
    return not trading_features.dropna().empty


def no_signal_result(label=""):
    """===================== 'SIT THIS WINDOW OUT' FALLBACK =====================
    Used by every ml_*.py strategy when trading_features_usable() (or the
    LSTM's equivalent all-empty-sequences check) is False: returns an empty
    predicted_spread Series instead of attempting to fit/predict on no data.

    finish_strategy() reindexes this onto the full trading-period index,
    producing an all-NaN z-score, and signals.generate_positions() already
    treats a NaN z-score as "hold the previous position" (which starts at
    0) -- so the net effect is a flat 0% return for this walk-forward
    window rather than a crash. This keeps a single bad window from taking
    down an entire combined-sector run; the rest of the pairs/periods are
    unaffected.
    ==========================================================================="""
    print(f"  WARNING{f' [{label}]' if label else ''}: trading-period features are entirely "
          f"NaN/unusable for this walk-forward window (likely a data gap in one leg of the "
          f"pair over this specific date range) -- skipping model fit/prediction, strategy "
          f"stays flat (0 return) for this period.")
    return pd.Series(dtype=float)


def compute_ml_zscore(predicted_spread, actual_spread, window=None):
    """===================== ML PREDICTION -> TRADING SIGNAL =====================
    This is where a model's forecast actually becomes a tradeable signal.
    Same rolling mean/std reference as classical_strategy.compute_zscore,
    but the numerator is the model's PREDICTED next-day spread instead of
    the actual current spread -- the ONLY difference between "classical"
    and "ML" in this whole codebase is this one substitution.
    =============================================================================="""
    window = window if window is not None else config.ZSCORE_WINDOW
    mean = actual_spread.rolling(window=window).mean()
    std = actual_spread.rolling(window=window).std()
    return (predicted_spread - mean) / std


def finish_strategy(full_price1, full_price2, predicted_spread, full_spread, trading_start, trading_end):
    """Common tail end for every ML strategy: turn a predicted-spread series
    into a z-score, then positions, then daily returns -- identical trading
    rule to the classical strategy (see signals.py)."""
    actual_spread = full_spread.loc[trading_start:trading_end]
    predicted_spread = predicted_spread.reindex(actual_spread.index)
    zscore = compute_ml_zscore(predicted_spread, actual_spread)

    pos1, pos2 = signals.generate_positions(zscore)
    trading_p1 = full_price1.loc[trading_start:trading_end]
    trading_p2 = full_price2.loc[trading_start:trading_end]
    daily_returns = signals.strategy_returns(trading_p1, trading_p2, pos1, pos2)

    return {
        "predicted_spread": predicted_spread,
        "actual_spread": actual_spread,  # exposed so callers can compute RMSE / compare to a naive baseline
        "zscore": zscore,
        "position1": pos1,
        "position2": pos2,
        "daily_returns": daily_returns,
    }


def naive_persistence_forecast(full_spread, trading_start, trading_end):
    """===================== NAIVE BASELINE (persistence forecast) =====================
    The simplest possible forecast: tomorrow's spread = today's spread. Any
    ML model that can't beat this on RMSE isn't adding real predictive
    signal over doing nothing -- worth reporting alongside every model's own
    RMSE (see prediction_rmse()) so "the model has low training MSE" claims
    can be checked against a trivial reference point instead of read in
    isolation.
    ======================================================================================"""
    predicted = full_spread.shift(1).loc[trading_start:trading_end]
    actual = full_spread.loc[trading_start:trading_end]
    return predicted, actual


def run_naive_baseline(full_price1, full_price2, formation_end=None, trading_start=None, trading_end=None):
    """The naive persistence baseline (see naive_persistence_forecast()),
    framed with the SAME (price1, price2, formation_end, trading_start,
    trading_end) signature every other strategy here uses, purely so it can
    be run through walk_forward.run_walk_forward() the same way and stay
    period-consistent with whatever config.WALK_FORWARD is set to. Not a
    tradeable strategy -- has no daily_returns, just predicted_spread /
    actual_spread -- used only to give every real model's own RMSE an
    honest, trivial reference point to beat."""
    formation_end = formation_end or config.FORMATION_END
    trading_start = trading_start or config.TRADING_START
    trading_end = trading_end or config.TRADING_END

    formation_p1 = full_price1.loc[:formation_end]
    formation_p2 = full_price2.loc[:formation_end]
    const, beta = classical_strategy.fit_spread_model(formation_p1, formation_p2)
    full_spread = classical_strategy.compute_spread(full_price1, full_price2, const, beta)

    predicted, actual = naive_persistence_forecast(full_spread, trading_start, trading_end)
    return {"predicted_spread": predicted, "actual_spread": actual}


def prediction_rmse(predicted_spread, actual_spread):
    """RMSE between any predicted-spread series and the actual spread,
    aligned on shared (non-NaN) dates. Used for both each ML model's own
    predictions and the naive persistence baseline above, so they're
    directly comparable."""
    aligned = pd.concat(
        [predicted_spread.rename("pred"), actual_spread.rename("actual")], axis=1
    ).dropna()
    if aligned.empty:
        return np.nan
    return float(np.sqrt(((aligned["pred"] - aligned["actual"]) ** 2).mean()))


def standardize_features(train_X, other_X_list):
    """===================== FEATURE STANDARDIZATION (flat models) =====================
    Fits a StandardScaler on the FORMATION period only (no look-ahead --
    the trading period never influences the mean/std used to rescale it),
    then applies that same fitted transform to every other DataFrame passed
    in (typically the trading-period features).

    Why this matters: Random Forest splits on raw thresholds so scale is
    irrelevant to it, but SVM (margin/regularization) and any gradient-
    trained model (Neural Net, LSTM) are sensitive to input scale -- a
    feature like SpreadLag1 (raw price-difference units, can be in the
    tens) sitting next to Asset1Returns (typically +/-0.02) will otherwise
    dominate the fit simply because of its units, not because it's more
    informative.
    ===================================================================================="""
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_X), index=train_X.index, columns=train_X.columns
    )
    others_scaled = [
        pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
        for X in other_X_list
    ]
    return train_scaled, others_scaled, scaler


def standardize_sequences(X_seq_train, X_seq_other_list):
    """Same idea as standardize_features(), but for the 3D sequence arrays
    (n_samples, seq_len, n_features) used by the LSTM. The scaler is fit on
    the formation sequences only, flattened across samples and time steps,
    then applied to every array in X_seq_other_list."""
    n_features = X_seq_train.shape[2]
    scaler = StandardScaler()
    scaler.fit(X_seq_train.reshape(-1, n_features))

    def _transform(X_seq):
        shape = X_seq.shape
        flat_scaled = scaler.transform(X_seq.reshape(-1, n_features))
        return flat_scaled.reshape(shape)

    train_scaled = _transform(X_seq_train)
    others_scaled = [_transform(X) for X in X_seq_other_list]
    return train_scaled, others_scaled, scaler


def train_validation_split_flat(X, y, validation_fraction=0.2):
    """Chronological (NOT shuffled -- this is time series) split of the
    formation period into a training slice and a validation slice, used for
    early stopping in ml_neural_net.py. The validation slice is the LAST
    validation_fraction of the formation period -- i.e. still entirely
    before the trading period, so this doesn't leak trading-period
    information, it just holds out the end of formation to monitor
    generalization during training."""
    n = len(X)
    split_idx = int(n * (1 - validation_fraction))
    return X.iloc[:split_idx], y.iloc[:split_idx], X.iloc[split_idx:], y.iloc[split_idx:]


def train_validation_split_sequences(X_seq, y_seq, validation_fraction=0.2):
    """Same idea as train_validation_split_flat(), for the 3D sequence
    arrays used by ml_lstm.py."""
    n = len(X_seq)
    split_idx = int(n * (1 - validation_fraction))
    return X_seq[:split_idx], y_seq[:split_idx], X_seq[split_idx:], y_seq[split_idx:]


def check_convergence(loss_history, model_name, window=10, tolerance=0.01):
    """Shared by ml_neural_net.py and ml_lstm.py: prints a warning if the
    training loss is still improving by more than `tolerance` (relative)
    over the last `window` epochs -- a quick, honest diagnostic rather than
    silently trusting a fixed epoch count."""
    if len(loss_history) < window + 1:
        return
    recent_start, recent_end = loss_history[-window - 1], loss_history[-1]
    if recent_start == 0:
        return
    relative_change = abs(recent_start - recent_end) / abs(recent_start)
    if relative_change > tolerance:
        print(f"  WARNING: {model_name} training loss changed by {relative_change:.1%} "
              f"over the last {window} epochs -- likely hasn't converged. Consider raising EPOCHS.")
    else:
        print(f"  {model_name} training loss changed by only {relative_change:.1%} "
              f"over the last {window} epochs -- looks converged.")


def build_sequences(features_df, spread_series, seq_len):
    """For sequence models (LSTM): turn a flat (n_rows, n_features) frame
    into overlapping windows of shape (n_samples, seq_len, n_features).

    features_df and spread_series should share the same date index (pass the
    FULL history, formation+trading concatenated -- NOT the pre-shifted
    'target' used by the flat models). Window [i-seq_len+1, ..., i] (using
    only information known through day i) predicts spread_series at i+1
    (tomorrow) -- this avoids the off-by-one ambiguity of trying to recover
    dates from an already-shifted target series.

    Returns (X_seq, y_seq, target_dates) as numpy arrays / a DatetimeIndex,
    where target_dates[k] is the actual calendar date the k-th prediction
    corresponds to -- safe to reindex predictions back onto later."""
    common_idx = features_df.dropna().index.intersection(spread_series.dropna().index)
    features_df = features_df.loc[common_idx]
    spread_series = spread_series.loc[common_idx]

    values = features_df.values
    dates = features_df.index

    X_seq, y_seq, target_dates = [], [], []
    for i in range(seq_len - 1, len(dates) - 1):  # need i+1 to exist
        window = values[i - seq_len + 1: i + 1]      # seq_len rows ending at position i
        X_seq.append(window)
        y_seq.append(spread_series.iloc[i + 1])       # tomorrow's actual spread
        target_dates.append(dates[i + 1])             # tomorrow's date

    return np.array(X_seq), np.array(y_seq), pd.DatetimeIndex(target_dates)
