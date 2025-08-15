from mcp.server.fastapi import FastAPIServer
from mcp.server.stdio import stdio_server
from fastapi import FastAPI
from typing import Any
import os

app = FastAPI()
server = FastAPIServer(app)

@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


if __name__ == "__main__":
    # stdio server for MCP tool registration
    stdio_server(app)
