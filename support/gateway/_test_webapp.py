"""Standalone smoke playground server.

Usage:  python -m localmelo.support.gateway._test_webapp
Opens on http://127.0.0.1:8401
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from .guard import LoopbackGuard
from .webapp import mount

PORT = 8401

app = FastAPI()
mount(app)
# Its sessions run tasks with tools: like the gateway, refuse requests that
# are not addressed to it.
app.add_middleware(LoopbackGuard, port=PORT)


if __name__ == "__main__":
    print(f"\n  Smoke playground: http://127.0.0.1:{PORT}/")
    print("  (connect your own chat + embedding URLs in the UI)\n")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
