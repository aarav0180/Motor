/**
 * client/js/ws.js — WebSocket Live Stream Client with Resilient Auto-Reconnect.
 */

class WebSocketClient {
  constructor() {
    this.socket = null;
    this.reconnectAttempts = 0;
    this.reconnectTimer = null;
    this.heartbeatInterval = null;
    this.isConnected = false;

    // Callbacks
    this.onTelemetry = null;
    this.onAlert = null;
    this.onConfigUpdate = null;
    this.onStatusChange = null;
  }

  connect() {
    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    try {
      this._updateStatus("connecting");
      this.socket = new WebSocket(Config.wsUrl);

      this.socket.onopen = () => {
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this._updateStatus("connected");
        this._startHeartbeat();
      };

      this.socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          this._handleMessage(message);
        } catch (err) {
          console.warn("[WS] Error parsing payload:", err);
        }
      };

      this.socket.onclose = () => {
        this.isConnected = false;
        this._stopHeartbeat();
        this._updateStatus("disconnected");
        this._scheduleReconnect();
      };

      this.socket.onerror = (err) => {
        console.warn("[WS] Error encountered:", err);
      };
    } catch (exc) {
      console.error("[WS] Connection failed:", exc);
      this._scheduleReconnect();
    }
  }

  _handleMessage(message) {
    switch (message.type) {
      case "TELEMETRY_UPDATE":
        if (this.onTelemetry) this.onTelemetry(message.data);
        break;
      case "ALERT":
        if (this.onAlert) this.onAlert(message.data);
        break;
      case "CONFIG_UPDATE":
        if (this.onConfigUpdate) this.onConfigUpdate(message.data);
        break;
      default:
        console.debug("[WS] Unhandled message type:", message.type);
    }
  }

  _scheduleReconnect() {
    if (this.reconnectTimer) return;

    this.reconnectAttempts++;
    const delay = Math.min(
      Config.reconnectDelayMs * Math.pow(1.5, this.reconnectAttempts - 1),
      Config.maxReconnectDelayMs
    );

    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
  }

  _startHeartbeat() {
    this._stopHeartbeat();
    this.heartbeatInterval = setInterval(() => {
      if (this.socket && this.socket.readyState === WebSocket.OPEN) {
        this.socket.send("ping");
      }
    }, 15000);
  }

  _stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  _updateStatus(status) {
    if (this.onStatusChange) {
      this.onStatusChange(status);
    }
  }
}

window.wsClient = new WebSocketClient();
