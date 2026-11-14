/**
 * client/js/api.js — REST Client for Tier 2 API.
 */

const Api = {
  async request(path, options = {}) {
    const url = `${Config.apiBaseUrl}${path}`;
    const defaultHeaders = {
      "Content-Type": "application/json",
      Accept: "application/json",
    };

    const res = await fetch(url, {
      ...options,
      headers: { ...defaultHeaders, ...options.headers },
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}: ${res.statusText}`);
    }

    return res.json();
  },

  // Health & Server Heartbeat
  getHealth() {
    return this.request("/health");
  },

  // Telemetry
  getLatestTelemetry() {
    return this.request("/api/v1/telemetry/latest");
  },

  getTelemetryHistory(limit = 60) {
    return this.request(`/api/v1/telemetry/history?limit=${limit}`);
  },

  // Alerts
  getAlerts(unacknowledgedOnly = false, limit = 20) {
    return this.request(
      `/api/v1/alerts?unacknowledged_only=${unacknowledgedOnly}&limit=${limit}`
    );
  },

  acknowledgeAlert(alertId) {
    return this.request(`/api/v1/alerts/${alertId}/acknowledge`, {
      method: "POST",
    });
  },

  // Human-in-the-Loop Feedback
  submitFeedback(event, score, notes = "") {
    return this.request("/api/v1/feedback", {
      method: "POST",
      body: JSON.stringify({ event, score, notes }),
    });
  },

  // Configuration
  getConfig() {
    return this.request("/api/v1/config");
  },

  updateThreshold(threshold, reason = "manual_dashboard_override") {
    return this.request("/api/v1/config/threshold", {
      method: "POST",
      body: JSON.stringify({ threshold: parseFloat(threshold), reason }),
    });
  },

  setSensorMode(mode) {
    return this.request("/api/v1/config/sensor-mode", {
      method: "POST",
      body: JSON.stringify({ mode }),
    });
  },

  setModelMode(model) {
    return this.request("/api/v1/config/model", {
      method: "POST",
      body: JSON.stringify({ model }),
    });
  },

  // Web Push Subscriptions
  getVapidPublicKey() {
    return this.request("/api/v1/push/key");
  },

  subscribePush(subscriptionData) {
    return this.request("/api/v1/push/subscribe", {
      method: "POST",
      body: JSON.stringify(subscriptionData),
    });
  },

  unsubscribePush(endpoint) {
    return this.request("/api/v1/push/unsubscribe", {
      method: "POST",
      body: JSON.stringify({ endpoint }),
    });
  },

  triggerTestPush() {
    return this.request("/api/v1/push/test", {
      method: "POST",
    });
  },
};

window.Api = Api;
