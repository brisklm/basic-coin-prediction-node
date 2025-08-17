import json
import os
import pickle
from zipfile import ZipFile
import pandas as pd
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.svm import SVR
from updater import (
    download_binance_daily_data,
    download_binance_current_day_data,
    download_coingecko_data,
    download_coingecko_current_day_data,
)
from config import data_base_path, model_file_path, TOKEN, MODEL, CG_API_KEY
from features import build_features


binance_data_path = os.path.join(data_base_path, "binance")
coingecko_data_path = os.path.join(data_base_path, "coingecko")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")


def download_data_binance(token, training_days, region):
    files = download_binance_daily_data(f"{token}USDT", training_days, region, binance_data_path)
    print(f"Downloaded {len(files)} new files")
    return files

def download_data_coingecko(token, training_days):
    files = download_coingecko_data(token, training_days, coingecko_data_path, CG_API_KEY)
    print(f"Downloaded {len(files)} new files")
    return files


def download_data(token, training_days, region, data_provider):
    if data_provider == "coingecko":
        files = download_data_coingecko(token, int(training_days))
        if not files:
            # Retry with smaller windows acceptable by CG if 'max' or large windows fail
            for td in [365, 180, 90, 30, 14, 7]:
                print(f"Retrying CoinGecko with days={td}")
                files = download_data_coingecko(token, td)
                if files:
                    break
        if not files:
            print("CoinGecko unavailable; falling back to Binance daily zips (120d)")
            return download_data_binance(token, min(int(training_days), 120), region)
        return files
    elif data_provider == "binance":
        return download_data_binance(token, training_days, region)
    else:
        raise ValueError("Unsupported data provider")
    
def format_data(files, data_provider):
    if not files:
        print("Already up to date")
        return
    
    if data_provider == "binance":
        files = sorted([x for x in os.listdir(binance_data_path) if x.startswith(f"{TOKEN}USDT")])
    elif data_provider == "coingecko":
        files = sorted([x for x in os.listdir(coingecko_data_path) if x.endswith(".json")])

    # No files to process
    if len(files) == 0:
        return

    price_df = pd.DataFrame()
    if data_provider == "binance":
        for file in files:
            zip_file_path = os.path.join(binance_data_path, file)

            if not zip_file_path.endswith(".zip"):
                continue

            myzip = ZipFile(zip_file_path)
            with myzip.open(myzip.filelist[0]) as f:
                line = f.readline()
                header = 0 if line.decode("utf-8").startswith("open_time") else None
            df = pd.read_csv(myzip.open(myzip.filelist[0]), header=header).iloc[:, :11]
            df.columns = [
                "start_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "end_time",
                "volume_usd",
                "n_trades",
                "taker_volume",
                "taker_volume_usd",
            ]
            df.index = [pd.Timestamp(x + 1, unit="ms").to_datetime64() for x in df["end_time"]]
            df.index.name = "date"
            price_df = pd.concat([price_df, df])

            price_df.sort_index().to_csv(training_price_data_path)
    elif data_provider == "coingecko":
        for file in files:
            with open(os.path.join(coingecko_data_path, file), "r") as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                df.columns = [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close"
                ]
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.drop(columns=["timestamp"], inplace=True)
                df.set_index("date", inplace=True)
                price_df = pd.concat([price_df, df])

            price_df.sort_index().to_csv(training_price_data_path)


def load_frame(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    print("Loading data...")
    df = frame.loc[:, ["open", "high", "low", "close"]].dropna()
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].apply(
        pd.to_numeric
    )
    df["date"] = frame["date"].apply(pd.to_datetime)
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    # Aggregate to desired timeframe (e.g., 1D for daily)
    agg = df.resample(f"{timeframe}", label="right", closed="right", origin="end").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
    ).dropna()
    return agg


def compute_daily_log_return(daily_df: pd.DataFrame) -> pd.DataFrame:
    # Compute 24h log-return ln(close_t / close_{t-1})
    daily_df = daily_df.copy()
    daily_df["log_return"] = np.log(daily_df["close"]).diff()
    return daily_df.dropna()

def train_model(timeframe: str):
    # Load the aggregated price data
    if not os.path.exists(training_price_data_path):
        raise FileNotFoundError(
            f"Training data not found at {training_price_data_path}. "
            "Ensure data download succeeded (fallback to Binance is automatic if CoinGecko fails)."
        )
    price_data = pd.read_csv(training_price_data_path)
    df_daily = load_frame(price_data, timeframe)
    df_lr = compute_daily_log_return(df_daily)

    print(df_lr.tail())

    # Predict next day's log-return using last day's OHLC as features
    # Build features with a medium variant by default
    X_df, y_series = build_features(df_lr, variant=os.environ.get("FEATURES_VARIANT", "lags_medium"))
    y_train = y_series.values
    X_train = X_df.values

    print(f"Training data shape: {X_train.shape}, {y_train.shape}")

    # Model selection: if MODEL == "AUTO", choose based on simple CV else fixed
    selected_model_name = MODEL
    if selected_model_name is None or selected_model_name.upper() == "AUTO":
        from model_selection import select_best_model

        model = select_best_model(X_train, y_train)
    else:
        if selected_model_name == "LinearRegression":
            model = LinearRegression()
        elif selected_model_name == "SVR":
            model = SVR()
        elif selected_model_name == "KernelRidge":
            model = KernelRidge()
        elif selected_model_name == "BayesianRidge":
            model = BayesianRidge()
        else:
            raise ValueError("Unsupported model")

    # Train the model
    model.fit(X_train, y_train)

    # Persist model
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
    with open(model_file_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Trained model (log-return) saved to {model_file_path}")


def get_inference(token: str, timeframe: str, region: str, data_provider: str) -> float:
    """Load model and predict next 24h log-return."""
    with open(model_file_path, "rb") as f:
        loaded_model = pickle.load(f)

    # Get current aggregated features
    df_now = None
    if data_provider == "coingecko":
        try:
            df_now = load_frame(download_coingecko_current_day_data(token, CG_API_KEY), timeframe)
        except Exception as e:
            print(f"CoinGecko current-day fetch failed: {e}; falling back to Binance")
            df_now = load_frame(download_binance_current_day_data(f"{TOKEN}USDT", region), timeframe)
    else:
        df_now = load_frame(download_binance_current_day_data(f"{TOKEN}USDT", region), timeframe)

    print(df_now.tail())
    # For inference, construct features using latest window
    df_now_lr = df_now.copy()
    df_now_lr["log_return"] = np.log(df_now_lr["close"]).diff()
    X_df_now, _ = build_features(df_now_lr, variant=os.environ.get("FEATURES_VARIANT", "lags_medium"))
    X_new = X_df_now.tail(1).values

    log_return_pred = loaded_model.predict(X_new)
    return float(log_return_pred[0])