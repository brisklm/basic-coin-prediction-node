import json
import os
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Any
from config import data_base_path, model_file_path
from model_selection import zptae_loss

METRICS_PATH = os.path.join(data_base_path, "metrics.json")


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
    y = df_lr["log_return"].values
    X = df_lr[["open", "high", "low", "close"]].values

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
    new_score = zptae_loss(y, new_pred)

    improved = new_score <= current_score

    # Write metrics
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    metrics = _safe_read_metrics()
    metrics.update({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "current_score": current_score,
        "new_score": new_score,
        "improved": improved
    })
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # If not improved, revert model
    if not improved and prev_model_bytes is not None:
        with open(model_file_path, "wb") as f:
            f.write(prev_model_bytes)

    # Auto-commit and push code/data changes (model, metrics)
    try:
        repo = os.environ.get("REPO_NAME")
        token = os.environ.get("GITHUB_TOKEN")
        if repo and token:
            os.system("git add data/*.pkl data/metrics.json || true")
            os.system("git add data/ price_data.csv || true")
            os.system("git add . || true")
            os.system("git config user.email \"auto@allora.local\" && git config user.name \"Allora Auto Optimizer\"")
            os.system("git commit -m 'auto: evaluate and optimize model' || true")
            os.system("git push origin HEAD || true")
    except Exception:
        pass
