"""
server/ws_manager.py — WebSocket connection and real-time broadcast manager.

Bridges incoming MQTT edge events directly to connected browser/mobile clients
in <10 ms, eliminating client HTTP polling.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from fastapi import WebSocket

logger = logging.getLogger("server.ws")


class ConnectionManager:
    """Manages active WebSocket connections and real-time broadcasting."""

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register the running asyncio event loop for cross-thread calls."""
        self._loop = loop

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new client connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket client connected. Total active: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a disconnected client."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("WebSocket client disconnected. Total active: %d", len(self.active_connections))

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast a JSON message asynchronously to all connected clients."""
        if not self.active_connections:
            return

        dead_connections = []
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)

        for dead in dead_connections:
            self.disconnect(dead)

    def broadcast_sync(self, message: Dict[str, Any]) -> None:
        """
        Thread-safe broadcast helper called from the background MQTT worker thread.
        Dispatches coroutine into FastAPI's running asyncio event loop.
        """
        if not self.active_connections:
            return

        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self._loop)
        else:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)
            except Exception as exc:
                logger.debug("Could not dispatch broadcast_sync: %s", exc)


ws_manager = ConnectionManager()
