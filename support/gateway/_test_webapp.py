"""Standalone smoke playground server.

Usage:  python -m localmelo.support.gateway._test_webapp
Opens on http://127.0.0.1:8401
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from .webapp import mount

app = FastAPI()
mount(app)


if __name__ == "__main__":
    print("\n  Smoke playground: http://127.0.0.1:8401/")
    print("  (connect your own chat + embedding URLs in the UI)\n")
    uvicorn.run(app, host="127.0.0.1", port=8401, log_level="info")
