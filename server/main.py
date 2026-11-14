"""
server/main.py — FastAPI Application Entry Point for Tier 2 Server.

Coordinates:
  - Database initialization (SQLite WAL)
  - MQTT Bridge lifecycle (subscribing to Edge & broadcasting config updates)
  - REST API Routers (/api/v1)
  - Real-Time WebSocket Streaming (/ws/telemetry)
"""

import asyncio
import contextlib
import logging
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from .config import settings
from .database import init_db
from .mqtt_bridge import mqtt_bridge
from .mock_streamer import mock_streamer
from .ws_manager import ws_manager
from .routes import health, telemetry, alerts, feedback, config, push

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("server.main")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    # 1. Initialize database schema
    logger.info("Initializing database...")
    init_db()

    # 2. Register active asyncio loop with WebSocket connection manager
    ws_manager.set_loop(asyncio.get_running_loop())

    # 3. Start background MQTT bridge worker
    logger.info("Starting MQTT Bridge worker...")
    mqtt_bridge.start()

    # 4. Start background Mock Edge Streamer (continuous telemetry in mock mode)
    logger.info("Starting Mock Edge Streamer worker...")
    mock_streamer.start()

    yield

    # 5. Graceful shutdown
    logger.info("Stopping Mock Edge Streamer worker...")
    mock_streamer.stop()

    logger.info("Stopping MQTT Bridge worker...")
    mqtt_bridge.stop()


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title="Motor Fault Detection — Tier 2 Server API",
        version="1.0.0",
        description="Tier 2 Cloud / Gateway API in the 3-Tier Decoupled Predictive Maintenance Architecture.",
        lifespan=lifespan,
    )

    # CORS configuration for Web/Mobile frontends
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins if settings.cors_origins != ["*"] else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount REST routers
    app.include_router(health.router)
    app.include_router(telemetry.router)
    app.include_router(alerts.router)
    app.include_router(feedback.router)
    app.include_router(config.router)
    app.include_router(push.router)

    # Mount WebSocket Real-Time Stream
    @app.websocket("/ws/telemetry")
    async def websocket_telemetry_endpoint(websocket: WebSocket):
        await ws_manager.connect(websocket)
        try:
            while True:
                # Keep connection alive, listen for client heartbeat/messages
                _ = await websocket.receive_text()
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)
        except Exception:
            ws_manager.disconnect(websocket)

    # Mount Tier 3 Client Application (Web / PWA)
    client_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "client")
    if os.path.exists(client_dir):
        app.mount("/", StaticFiles(directory=client_dir, html=True), name="client")

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "server.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=False,
    )
