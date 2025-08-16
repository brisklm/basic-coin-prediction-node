from mcp.server.fastapi import FastAPIServer
from fastapi import FastAPI
from typing import Any
import os
import json

app = FastAPI()
server = FastAPIServer(app)

@app.get("/widen")
def widen() -> dict[str, Any]:
    mlcfg_path = os.path.join(os.getcwd(), "mlconfig.json")
    if not os.path.exists(mlcfg_path):
        return {"ok": False, "error": "mlconfig.json not found"}
    with open(mlcfg_path, "r") as f:
        mlcfg = json.load(f)
    ms = mlcfg.setdefault("model_selection", {})
    candidates = ms.setdefault("candidate_models", [])
    # ensure a couple of models
    names = {c.get("name") for c in candidates}
    for m in ["SVR", "KernelRidge", "LinearRegression", "BayesianRidge"]:
        if m not in names:
            candidates.append({"name": m})
    with open(mlcfg_path, "w") as f:
        json.dump(mlcfg, f, indent=2)
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
