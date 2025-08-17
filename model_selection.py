import json
import os
from typing import Tuple
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import inspect
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error
from datetime import datetime
from config import data_base_path

CONFIG_PATH = os.path.join(os.getcwd(), "mlconfig.json")


def zptae_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Z-transform with reference mean = 0 and rolling std estimation proxy
    # Here we standardize y_true using its own std, avoiding leakage by TSSplit
    abs_err = np.abs(y_true - y_pred)
    # smooth power-tanh: tanh(x) + x^p tail; use p=1.3
    p = 1.3
    transformed = np.tanh(abs_err) + np.power(abs_err, p) * 0.05
    return float(np.mean(transformed))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    s_true = np.sign(y_true)
    s_pred = np.sign(y_pred)
    return float(np.mean((s_true == s_pred).astype(np.float32)))


def mztae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Modified Z-Transformed Absolute Error variant
    # Standardize true values; then power-tanh of abs error
    y_std = (y_true - 0.0) / (np.std(y_true) + 1e-8)
    abs_err = np.abs(y_std - y_pred)
    return float(np.mean(np.tanh(abs_err) + np.power(abs_err, 1.2) * 0.05))


def read_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def _filter_supported_kwargs(klass, params: dict | None) -> dict:
    if not params:
        return {}
    sig = inspect.signature(klass.__init__)
    allowed = set(sig.parameters.keys())
    # Drop unknown and None values
    return {k: v for k, v in params.items() if k in allowed and v is not None}


def build_model(name: str, params: dict | None):
    if name == "LinearRegression":
        return LinearRegression(**_filter_supported_kwargs(LinearRegression, params))
    if name == "BayesianRidge":
        return BayesianRidge(**_filter_supported_kwargs(BayesianRidge, params))
    if name == "KernelRidge":
        return KernelRidge(**_filter_supported_kwargs(KernelRidge, params))
    if name == "SVR":
        return SVR(**_filter_supported_kwargs(SVR, params))
    raise ValueError(f"Unknown model {name}")


def tune_param_grid(model_name: str, param_grid: dict | None) -> list[dict]:
    if not param_grid:
        return [None]
    # Expand grid. Accept scalars or iterables for each param.
    keys = list(param_grid.keys())
    raw_values = [param_grid[k] for k in keys]

    def to_list(value):
        if isinstance(value, (list, tuple, set)):
            return list(value)
        # wrap scalars
        return [value]

    def normalize(seq):
        return [None if x is None else x for x in seq]

    values = [normalize(to_list(v)) for v in raw_values]

    grids = [{}]
    for key, candidates in zip(keys, values):
        grids = [dict(g, **{key: cand}) for g in grids for cand in candidates]
    return grids


def time_series_cv_score(model, X: np.ndarray, y: np.ndarray, splits: int = 5) -> float:
    tscv = TimeSeriesSplit(n_splits=min(splits, len(X) - 1))
    scores: list[float] = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[test_idx]
        y_train, y_val = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_hat = model.predict(X_val)
        # Composite scoring emphasizing ZPTAE but checking RMSE and direction
        z = zptae_loss(y_val, y_hat)
        r = rmse(y_val, y_hat)
        d = direction_accuracy(y_val, y_hat)
        # Lower is better; reward higher direction by subtracting
        composite = z + 0.2 * r - 0.05 * d
        scores.append(composite)
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
    best_name: str | None = None

    # Collect comparison logs
    comparison: dict[str, dict] = {}

    print("Model selection: evaluating candidates...")
    for c in candidates:
        name = c.get("name")
        best_for_model = float("inf")
        best_params_for_model = None
        grid = tune_param_grid(name, c.get("params"))
        for params in grid:
            model = build_model(name, params)
            score = time_series_cv_score(model, X, y, splits=ms.get("cv_folds", 5))
            if score < best_for_model:
                best_for_model = score
                best_params_for_model = params
        if best_for_model < float("inf"):
            comparison[name] = {
                "cv_score": best_for_model,
                "best_params": best_params_for_model,
            }
            print(f" - {name}: best CV score={best_for_model:.6f} params={best_params_for_model}")
            if best_for_model < best_score:
                best_score = best_for_model
                best_model = build_model(name, best_params_for_model)
                best_name = name

    # Persist comparison log
    try:
        os.makedirs(data_base_path, exist_ok=True)
        out_path = os.path.join(data_base_path, "model_selection.json")
        with open(out_path, "w") as f:
            json.dump({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "results": comparison,
                "best": {"name": best_name, "cv_score": best_score}
            }, f, indent=2)
    except Exception as e:
        print(f"Warning: could not write model_selection.json: {e}")

    if best_name is not None:
        print(f"Selected model: {best_name} with CV score={best_score:.6f}")
    return best_model
