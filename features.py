from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple


def _make_lags(series: pd.Series, lags: list[int]) -> pd.DataFrame:
    lagged = {f"lag_{k}": series.shift(k) for k in lags}
    return pd.DataFrame(lagged, index=series.index)


def _rolling_stats(series: pd.Series, windows: list[int]) -> pd.DataFrame:
    feats = {}
    for w in windows:
        roll = series.rolling(window=w, min_periods=max(2, w // 2))
        feats[f"roll_mean_{w}"] = roll.mean()
        feats[f"roll_std_{w}"] = roll.std()
        feats[f"roll_min_{w}"] = roll.min()
        feats[f"roll_max_{w}"] = roll.max()
    return pd.DataFrame(feats, index=series.index)


def _ewm_features(series: pd.Series, spans: list[int]) -> pd.DataFrame:
    feats = {}
    for s in spans:
        e = series.ewm(span=s, adjust=False)
        feats[f"ewm_mean_{s}"] = e.mean()
        feats[f"ewm_std_{s}"] = e.std()
    return pd.DataFrame(feats, index=series.index)


def build_features(df_lr: pd.DataFrame, variant: str = "base") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and target y given a daily dataframe with columns:
    ['open','high','low','close','log_return'] indexed by date.

    Variants:
    - base: OHLC only
    - lags_small: OHLC + log_return + lags 1..3 + roll stats (3,7)
    - lags_medium: OHLC + log_return + lags 1..7 + roll stats (3,7,14) + EWM (5,10)
    - lags_large: OHLC + log_return + lags 1..14 + roll stats (3,7,14,28) + EWM (5,10,20)
    """
    variant = (variant or "base").lower()

    base = df_lr[["open", "high", "low", "close"]].copy()
    # Normalize OHLC by close to reduce scale issues
    norm = base.div(df_lr["close"].replace(0, np.nan), axis=0)
    norm.columns = [f"norm_{c}" for c in norm.columns]

    X = pd.concat([base, norm], axis=1)

    if variant == "base":
        feats = X
    else:
        r = df_lr["log_return"].copy()
        if variant == "lags_small":
            lags = _make_lags(r, [1, 2, 3])
            rolls = _rolling_stats(r, [3, 7])
            feats = pd.concat([X, r.rename("r1"), lags, rolls], axis=1)
        elif variant == "lags_medium":
            lags = _make_lags(r, list(range(1, 8)))
            rolls = _rolling_stats(r, [3, 7, 14])
            ewms = _ewm_features(r, [5, 10])
            feats = pd.concat([X, r.rename("r1"), lags, rolls, ewms], axis=1)
        elif variant == "lags_large":
            lags = _make_lags(r, list(range(1, 15)))
            rolls = _rolling_stats(r, [3, 7, 14, 28])
            ewms = _ewm_features(r, [5, 10, 20])
            feats = pd.concat([X, r.rename("r1"), lags, rolls, ewms], axis=1)
        else:
            feats = X

    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    # Align target y to features index
    y = df_lr.loc[feats.index, "log_return"]
    return feats, y
