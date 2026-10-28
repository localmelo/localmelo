"""Authenticated standalone playground, bound only to loopback."""

from __future__ import annotations

import uvicorn

from localmelo.support import config
from localmelo.support.gateway import GatewayServer

cfg = config.load()
cfg._enable_playground = True
server = GatewayServer(cfg)
app = server.app
PORT = server.port

if __name__ == "__main__":
    uvicorn.run(app, host=server.host, port=PORT, log_level="info")
