import json
import os
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Any
from config import data_base_path, model_file_path
from features import build_features
from model_selection import zptae_loss, rmse, direction_accuracy

METRICS_PATH = os.path.join(data_base_path, "metrics.json")
MLCONFIG_PATH = os.path.join(os.getcwd(), "mlconfig.json")


def _safe_read_metrics() -> dict[str, Any]:
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}


def evaluate_current_model(X: np.ndarray, y: np.ndarray) -> float:
    if not os.path.exists(model_file_path):
        return float("inf")
    with open(model_file_path, "rb") as f:
        model = pickle.load(f)
    y_hat = model.predict(X)
    return zptae_loss(y, y_hat)


def evaluate_and_maybe_optimize():
    # Load training set built by model.py
    train_csv = os.path.join(data_base_path, "price_data.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError("Training data not found")
    import model as mdl

    df = pd.read_csv(train_csv)
    df_daily = mdl.load_frame(df, "1D")
    df_lr = mdl.compute_daily_log_return(df_daily)
    # Use same feature engineering as training
    variant = os.environ.get("FEATURES_VARIANT", "lags_medium")
    X_df, y_series = build_features(df_lr, variant=variant)
    y = y_series.values
    X = X_df.values

    current_score = evaluate_current_model(X, y)

    # Retrain via AUTO selection and compare
    prev_model_bytes = None
    if os.path.exists(model_file_path):
        with open(model_file_path, "rb") as f:
            prev_model_bytes = f.read()

    mdl.train_model("1D")

    with open(model_file_path, "rb") as f:
        new_model = pickle.load(f)
    new_pred = new_model.predict(X)
    # Reference std for ZPTAE based on last 100 ground-truths
    ref_std = float(np.std(y[-min(100, len(y)):]) + 1e-8)
    # Component metrics for current
    try:
        with open(model_file_path, "rb") as f:
            cur_model = pickle.load(f)
        cur_pred = cur_model.predict(X)
        cur_z = zptae_loss(y, cur_pred, ref_std=ref_std)
        cur_r = rmse(y, cur_pred)
        cur_d = direction_accuracy(y, cur_pred)
    except Exception:
        cur_z = current_score
        cur_r = float("nan")
        cur_d = float("nan")
    # Component metrics for new
    new_score = zptae_loss(y, new_pred, ref_std=ref_std)
    new_r = rmse(y, new_pred)
    new_d = direction_accuracy(y, new_pred)

    improved = new_score <= current_score

    # Write metrics
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    metrics = _safe_read_metrics()
    metrics.update({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "current_score": current_score,
        "new_score": new_score,
        "improved": improved,
        "components": {
            "reference_std_last100": ref_std,
            "current": {"zptae": cur_z, "rmse": cur_r, "direction_accuracy": cur_d},
            "new": {"zptae": new_score, "rmse": new_r, "direction_accuracy": new_d}
        }
    })
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # If not improved, revert model
    if not improved and prev_model_bytes is not None:
        with open(model_file_path, "wb") as f:
            f.write(prev_model_bytes)

    # If not improved, widen the hyperparameter search space in mlconfig.json automatically
    if not improved:
        try:
            if os.path.exists(MLCONFIG_PATH):
                with open(MLCONFIG_PATH, "r") as f:
                    mlcfg = json.load(f)
            else:
                mlcfg = {}

            ms = mlcfg.setdefault("model_selection", {})
            candidates = ms.setdefault("candidate_models", [])

            def ensure_model(name: str) -> dict:
                for c in candidates:
                    if c.get("name") == name:
                        return c
                c = {"name": name}
                candidates.append(c)
                return c

            # Expand SVR grid
            svr = ensure_model("SVR")
            svr_params = svr.setdefault("params", {})
            def extend(key, values):
                cur = svr_params.get(key, [])
                if not isinstance(cur, list):
                    cur = [cur]
                merged = sorted(set(cur + values), key=lambda x: (str(type(x)), x))
                svr_params[key] = merged
            extend("C", [0.1, 0.5, 1.0, 5.0, 10.0])
            extend("epsilon", [0.0, 1e-4, 1e-3, 1e-2, 0.1])
            extend("kernel", ["rbf", "linear", "poly"]) 

            # Expand KernelRidge grid
            kr = ensure_model("KernelRidge")
            kr_params = kr.setdefault("params", {})
            def extend_kr(key, values):
                cur = kr_params.get(key, [])
                if not isinstance(cur, list):
                    cur = [cur]
                merged = sorted(set(cur + values), key=lambda x: (str(type(x)), str(x)))
                kr_params[key] = merged
            extend_kr("alpha", [0.01, 0.1, 1.0, 10.0, 100.0])
            extend_kr("kernel", ["rbf", "linear"]) 
            extend_kr("gamma", [None, 1e-4, 1e-3, 1e-2, 1e-1])

            # Ensure LinearRegression and BayesianRidge are present
            ensure_model("LinearRegression")
            br = ensure_model("BayesianRidge")
            # remove unsupported params if present
            if isinstance(br.get("params"), dict):
                br["params"] = {k: v for k, v in br["params"].items() if k in []}

            # Persist updated mlconfig
            with open(MLCONFIG_PATH, "w") as f:
                json.dump(mlcfg, f, indent=2)
        except Exception as e:
            # Non-fatal: log and continue
            print(f"Auto-widen mlconfig failed: {e}")

    # Auto-commit and push code/data changes (model, metrics)
    try:
        repo = os.environ.get("REPO_NAME")
        token = os.environ.get("GITHUB_TOKEN")
        if repo and token:
            # Force add tracked artifacts even if ignored and include mlconfig updates
            os.system("git add -f data/model.pkl data/metrics.json data/price_data.csv mlconfig.json || true")
            os.system("git config user.email \"auto@allora.local\" && git config user.name \"Allora Auto Optimizer\"")
            os.system("git commit -m 'auto: evaluate and optimize model (and widen search if needed)' || true")
            os.system("git push origin HEAD || true")
    except Exception:
        pass
