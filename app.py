import json
from flask import Flask, Response
from model import download_data, format_data, train_model, get_inference
from optimizer import evaluate_and_maybe_optimize
from config import model_file_path, TOKEN, TIMEFRAME, TRAINING_DAYS, REGION, DATA_PROVIDER


app = Flask(__name__)


def update_data():
    """Download price data, format data and train model."""
    files = download_data(TOKEN, TRAINING_DAYS, REGION, DATA_PROVIDER)
    format_data(files, DATA_PROVIDER)
    # Ensure data CSV exists before training
    import os
    from model import training_price_data_path
    if not os.path.exists(training_price_data_path):
        raise RuntimeError("Data download failed; cannot find training CSV. Check provider/API and network.")
    train_model(TIMEFRAME)
    # Evaluate and possibly optimize; also commits changes if improved
    try:
        evaluate_and_maybe_optimize()
    except Exception as e:
        print(f"Optimizer warning: {e}")


@app.route("/inference/<string:token>")
def generate_inference(token):
    """Generate inference for given token."""
    if not token or token.upper() != TOKEN:
        error_msg = "Token is required" if not token else "Token not supported"
        return Response(json.dumps({"error": error_msg}), status=400, mimetype='application/json')

    try:
        inference = get_inference(token.upper(), TIMEFRAME, REGION, DATA_PROVIDER)
        return Response(str(inference), status=200)
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')


@app.route("/update")
def update():
    """Update data and return status."""
    try:
        update_data()
        return "0"
    except Exception:
        return "1"


@app.route("/metrics")
def metrics():
    """Expose last evaluation metrics if available."""
    import json, os
    metrics_path = os.path.join(os.path.dirname(model_file_path), "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return Response(f.read(), status=200, mimetype="application/json")
    return Response(json.dumps({"status": "no-metrics"}), status=200, mimetype="application/json")


if __name__ == "__main__":
    update_data()
    app.run(host="0.0.0.0", port=8000)
