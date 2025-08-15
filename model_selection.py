import json
import os
from typing import Tuple
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from scipy.stats import zscore

CONFIG_PATH = os.path.join(os.getcwd(), "config.json")


def zptae_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Z-transform with reference mean = 0 and rolling std estimation proxy
    # Here we standardize y_true using its own std, avoiding leakage by TSSplit
    abs_err = np.abs(y_true - y_pred)
    # smooth power-tanh: tanh(x) + x^p tail; use p=1.3
    p = 1.3
    transformed = np.tanh(abs_err) + np.power(abs_err, p) * 0.05
    return float(np.mean(transformed))


def read_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def build_model(name: str, params: dict | None):
    if name == "LinearRegression":
        return LinearRegression()
    if name == "BayesianRidge":
        return BayesianRidge(**(params or {}))
    if name == "KernelRidge":
        return KernelRidge(**(params or {}))
    if name == "SVR":
        return SVR(**(params or {}))
    raise ValueError(f"Unknown model {name}")


def tune_param_grid(model_name: str, param_grid: dict | None) -> list[dict]:
    if not param_grid:
        return [None]
    # expand grid
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    # Allow nulls
    def normalize(v):
        return [None if x is None else x for x in v]
    values = [normalize(v) for v in values]
    grids = [{}]
    for k, vs in zip(keys, values):
        grids = [dict(g, **{k: v}) for g in grids for v in vs]
    return grids


def time_series_cv_score(model, X: np.ndarray, y: np.ndarray, splits: int = 5) -> float:
    tscv = TimeSeriesSplit(n_splits=min(splits, len(X) - 1))
    scores: list[float] = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[test_idx]
        y_train, y_val = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_hat = model.predict(X_val)
        scores.append(zptae_loss(y_val, y_hat))
    return float(np.mean(scores))


def select_best_model(X: np.ndarray, y: np.ndarray):
    cfg = read_config()
    ms = cfg.get("model_selection", {})
    if not ms.get("enabled", True):
        return LinearRegression()
    candidates = ms.get("candidate_models", [
        {"name": "LinearRegression"}
    ])
    best_score = float("inf")
    best_model = None
    for c in candidates:
        name = c.get("name")
        grid = tune_param_grid(name, c.get("params"))
        for params in grid:
            model = build_model(name, params)
            score = time_series_cv_score(model, X, y, splits=ms.get("cv_folds", 5))
            if score < best_score:
                best_score = score
                best_model = build_model(name, params)
    return best_model
