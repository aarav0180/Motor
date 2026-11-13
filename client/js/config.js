/**
 * client/js/config.js — Dynamic Environment & Endpoint Config.
 */

const Config = {
  // Compute base API and WebSocket URLs from current window location
  get apiBaseUrl() {
    return window.location.origin;
  },

  get wsUrl() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}/ws/telemetry`;
  },

  motorId: "motor_1",
  reconnectDelayMs: 2000,
  maxReconnectDelayMs: 30000,
  heartbeatThresholdSeconds: 20,
};

window.Config = Config;
