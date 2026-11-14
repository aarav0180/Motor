/**
 * Dashboard orchestration for catch-up sync, live updates, and operator actions.
 * API contracts and UI styling stay in their respective modules.
 */
document.addEventListener("DOMContentLoaded", function () {
  window.app = new VibraGuardApp();
  window.app.init();
});

class VibraGuardApp {
  constructor() {
    this.chart = new TelemetryChart("telemetryChart");
    this.activeThreshold = 0.08;
    this.lastPacketTimestamp = null;
    this.selectedAlert = null;
    this.alertsByKey = new Map();
    this.activeFilter = "unreviewed";
    this.sensorMode = "mock";
    this.activeModel = "unsupervised";
    this.el = function (id) { return document.getElementById(id); };
  }

  async init() {
    this.chart.init(this.activeThreshold);
    this._bindEvents();
    this._icons();
    this._animateEntrance();
    this._initPush();
    await this.catchUpSync();
    this._connectWebSocket();
    this._startHeartbeatTimer();
  }

  _icons() {
    if (window.lucide && window.lucide.createIcons) window.lucide.createIcons();
  }

  _replaceIcon(button, name) {
    var current = button.querySelector("svg, i");
    if (!current) return;
    var icon = document.createElement("i");
    icon.setAttribute("data-lucide", name);
    current.replaceWith(icon);
    this._icons();
  }

  _animateEntrance() {
    if (!window.gsap || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    window.gsap.from(".welcome-row, .hero-card, .metric-card, .content-grid, .controls-panel", {
      opacity: 0, y: 12, duration: 0.55, stagger: 0.065, ease: "power2.out", clearProps: "all"
    });
  }

  async catchUpSync() {
    try {
      this._setPill("serverStatusPill", "online", "Server online");
      var config = await Api.getConfig();
      if (config) {
        if (Number.isFinite(Number(config.alert_threshold))) {
          this.activeThreshold = Number(config.alert_threshold);
          this.chart.updateThreshold(this.activeThreshold);
          this._updateThresholdDisplay(this.activeThreshold);
        }
        if (config.sensor_mode) {
          this.sensorMode = config.sensor_mode;
          this._updateSensorModeUI(this.sensorMode);
        }
        if (config.active_model) {
          this.activeModel = config.active_model;
          this._updateModelUI(this.activeModel);
        }
      }
      var history = await Api.getTelemetryHistory(60);
      if (history && history.length) {
        this.chart.loadHistory(history, this.activeThreshold);
        this._renderLatestTelemetry(history[history.length - 1]);
      } else {
        try { this._renderLatestTelemetry(await Api.getLatestTelemetry()); } catch (_) {}
      }
      await this.refreshAlerts();
    } catch (error) {
      console.warn("[Dashboard] Catch-up sync failed:", error);
      this._setPill("serverStatusPill", "offline", "Server unreachable");
    }
  }

  _connectWebSocket() {
    var self = this;
    wsClient.onStatusChange = function (status) {
      var state = status === "connected" ? "online" : status === "connecting" ? "warning" : "offline";
      var label = status === "connected" ? "Stream live" : status === "connecting" ? "Reconnecting" : "Stream disconnected";
      self._setPill("streamStatusPill", state, label);
      var note = self.el("chartAssetStatus");
      if (note) note.lastChild.textContent = status === "connected" ? " Live stream connected" : " Waiting for telemetry";
    };
    wsClient.onTelemetry = function (data) {
      self._renderLatestTelemetry(data);
      self.chart.pushPoint(data.timestamp || data.ts, Number(data.score) || 0, data.label);
    };
    wsClient.onAlert = function (data) {
      audioManager.playAlertChime();
      self.refreshAlerts();
      self.openFeedbackModal(data);
      if ("Notification" in window && Notification.permission === "granted") {
        try { new Notification("Motor anomaly alert", { body: data.message || ("Bearing anomaly score: " + Number(data.score).toFixed(4)), icon: "/manifest.json" }); } catch (_) {}
      }
    };
    wsClient.onConfigUpdate = function (data) { self._applyConfigUpdate(data || {}); };
    wsClient.connect();
  }

  _applyConfigUpdate(data) {
    if (data.alert_threshold !== undefined && Number.isFinite(Number(data.alert_threshold))) {
      this.activeThreshold = Number(data.alert_threshold);
      this.chart.updateThreshold(this.activeThreshold);
      this._updateThresholdDisplay(this.activeThreshold);
      this._showToast("Threshold adapted to " + this.activeThreshold.toFixed(5));
    }
    if (data.sensor_mode) {
      this.sensorMode = data.sensor_mode;
      this._updateSensorModeUI(data.sensor_mode);
      this._showToast("Sensor switched to " + data.sensor_mode.toUpperCase() + " mode");
    }
    if (data.active_model) {
      this.activeModel = data.active_model;
      this._updateModelUI(data.active_model);
      this._showToast("Model switched to " + data.active_model.toUpperCase());
    }
  }

  _renderLatestTelemetry(data) {
    if (!data) return;
    this.lastPacketTimestamp = Number(data.timestamp || data.ts || Date.now() / 1000);
    var score = Number(data.score) || 0;
    var diagnosis = data.diagnosis || (data.label === "normal" ? "healthy" : "anomaly");
    var confidence = (Number.isFinite(Number(data.confidence)) ? Number(data.confidence) : 1) * 100;
    var latency = ((Number(data.proc_time_s) || 0) * 1000).toFixed(2);
    var pct = this.activeThreshold > 0 ? Math.min(100, Math.round(score / this.activeThreshold * 100)) : 0;
    var state = score > this.activeThreshold ? "critical" : score > this.activeThreshold * 0.85 ? "warning" : "normal";
    this._text("statScore", score.toFixed(5));
    this._text("statDiagnosis", this._formatDiagnosis(diagnosis));
    this._text("statConfidence", confidence.toFixed(1) + "%");
    this._text("statLatencyValue", latency);
    this._text("statThreshold", this.activeThreshold.toFixed(5));
    this._text("statThresholdProgressLabel", pct + "% of threshold");

    var progress = this.el("statThresholdProgress");
    if (progress) {
      progress.style.width = pct + "%";
      progress.classList.toggle("is-warning", state === "warning");
      progress.classList.toggle("is-critical", state === "critical");
    }
    var card = this.el("heroStatusCard");
    var badge = this.el("heroHealthBadge");
    if (!card || !badge) return;
    card.classList.toggle("is-warning", state === "warning");
    card.classList.toggle("is-critical", state === "critical");

    var content = {
      normal: ["MOTOR STATUS: NORMAL OPERATION", "Your motor is operating within its healthy vibration baseline. Edge inference is active and monitoring every window.", "HEALTHY", "BASELINE STABLE"],
      warning: ["WARNING: ELEVATED VIBRATION", "Vibration is approaching its alert threshold (" + pct + "%). Continue monitoring the trend.", Math.max(0, 100 - Math.round(pct / 2)) + "%", "ELEVATED"],
      critical: ["CRITICAL FAULT DETECTED", "Vibration exceeded the active threshold of " + this.activeThreshold.toFixed(4) + ". Review the alert and inspect the machine.", "FAULT", this._formatDiagnosis(diagnosis).toUpperCase()]
    }[state];
    this._text("heroStatusText", content[0]);
    this._text("heroStatusDesc", content[1]);
    var diagnosisCaption = this.el("diagnosisCaption");
    if (diagnosisCaption) {
      diagnosisCaption.lastChild.textContent = state === "critical" ? " Edge alert raised" : state === "warning" ? " Approaching alert threshold" : " No fault detected";
    }
    badge.querySelector("strong").textContent = content[2];
    badge.querySelector("small").textContent = content[3];
  }

  _text(id, value) { var node = this.el(id); if (node) node.textContent = value; }

  _setPill(id, state, label) {
    var node = this.el(id);
    if (!node) return;
    node.classList.remove("is-online", "is-warning", "is-offline", "is-waiting");
    node.classList.add("is-" + state);
    var text = node.querySelector("span:last-child");
    if (text) text.textContent = label;
  }

  _updateThresholdDisplay(value) {
    this._text("statThreshold", value.toFixed(5));
    var input = this.el("manualThresholdInput");
    if (input) input.value = value.toFixed(5);
  }

  _updateSensorModeUI(mode) {
    this.el("sensorModeMockBtn").classList.toggle("is-selected", mode !== "physical");
    this.el("sensorModePhysicalBtn").classList.toggle("is-selected", mode === "physical");
  }

  _updateModelUI(mode) {
    this.el("modelModeUnsupBtn").classList.toggle("is-selected", mode !== "supervised");
    this.el("modelModeSuperBtn").classList.toggle("is-selected", mode === "supervised");
  }

  async setSensorMode(mode) {
    try {
      var result = await Api.setSensorMode(mode);
      this.sensorMode = mode;
      this._updateSensorModeUI(mode);
      this._showToast("Sensor set to " + mode.toUpperCase() + " · MQTT " + (result.mqtt_broadcast ? "sent" : "pending"));
    } catch (error) { this._showToast("Could not change sensor: " + error.message); }
  }

  async setModelMode(mode) {
    try {
      var result = await Api.setModelMode(mode);
      this.activeModel = mode;
      this._updateModelUI(mode);
      this._showToast("Model set to " + mode.toUpperCase() + " · MQTT " + (result.mqtt_broadcast ? "sent" : "pending"));
    } catch (error) { this._showToast("Could not change model: " + error.message); }
  }

  _formatDiagnosis(raw) {
    var labels = { healthy: "Healthy", ball_fault: "Ball bearing fault", inner_race_fault: "Inner race fault", outer_race_fault: "Outer race fault", anomaly: "Vibration anomaly" };
    return labels[raw] || raw || "Healthy";
  }

  _startHeartbeatTimer() {
    var self = this;
    function update() {
      if (!self.lastPacketTimestamp) {
        self._setPill("edgeHeartbeatPill", "waiting", "Edge waiting");
        return;
      }
      var age = Math.max(0, Math.floor(Date.now() / 1000 - self.lastPacketTimestamp));
      self._setPill("edgeHeartbeatPill", age < 15 ? "online" : age < 45 ? "warning" : "offline",
        age >= 45 ? "Edge offline · " + age + "s" : "Edge · " + age + "s ago");
    }
    update();
    setInterval(update, 1000);
  }

  async refreshAlerts() {
    try {
      var alerts = await Api.getAlerts(this.activeFilter === "unreviewed", 20);
      this._renderAlerts(alerts);
    } catch (error) { console.warn("[Alerts] Could not refresh:", error); }
  }

  _renderAlerts(alerts) {
    var list = this.el("alertsListContainer");
    this.alertsByKey.clear();
    var pending = alerts.filter(function (alert) { return !alert.acknowledged; }).length;
    ["unackAlertCountBadge", "activeFilterCount", "navAlertCount"].forEach((id) => this._text(id, pending));
    this.el("unackAlertCountBadge").classList.toggle("has-alerts", pending > 0);
    if (!alerts.length) {
      var all = this.activeFilter === "all";
      list.innerHTML = '<div class="empty-state"><span class="empty-icon"><i data-lucide="shield-check"></i></span><strong>' + (all ? "No alerts yet" : "All clear") + '</strong><p>' + (all ? "New events will appear here." : "No alerts need your attention.") + "</p></div>";
      this._icons();
      return;
    }

    var fragment = document.createDocumentFragment();
    alerts.forEach((alert, index) => {
      var key = String(index);
      this.alertsByKey.set(key, alert);
      var item = document.createElement("article");
      item.className = "alert-item" + (alert.acknowledged ? " is-acknowledged" : "");
      var marker = document.createElement("span");
      marker.className = "alert-marker";
      var info = document.createElement("div");
      info.className = "alert-info";
      var title = document.createElement("div");
      title.className = "alert-title";
      var diagnosis = document.createElement("span");
      diagnosis.textContent = this._formatDiagnosis(alert.diagnosis);
      var score = document.createElement("span");
      score.className = "alert-score";
      score.textContent = "Score " + Number(alert.score).toFixed(4);
      title.append(diagnosis, score);
      var meta = document.createElement("div");
      meta.className = "alert-meta";
      meta.textContent = new Date(Number(alert.timestamp) * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
      info.append(title, meta);
      var actions = document.createElement("div");
      actions.className = "alert-actions";
      actions.append(this._alertButton("Review", "review", key));
      actions.append(this._alertButton(alert.acknowledged ? "Reviewed" : "Acknowledge", alert.acknowledged ? "done" : "ack", key, true));
      item.append(marker, info, actions);
      fragment.append(item);
    });
    list.replaceChildren(fragment);
  }

  _alertButton(label, action, key, quiet) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "alert-action" + (quiet ? " is-quiet" : "") + (action === "done" ? " is-done" : "");
    button.dataset.action = action;
    button.dataset.alertKey = key;
    button.textContent = label;
    button.disabled = action === "done";
    return button;
  }

  openFeedbackModal(alert) {
    this.selectedAlert = alert;
    var score = Number(alert.score) || 0;
    this._text("modalAlertScore", score.toFixed(5));
    this._text("modalAlertThreshold", this.activeThreshold.toFixed(5));
    this._text("modalAlertDiagnosis", this._formatDiagnosis(alert.diagnosis));
    this._text("modalSuggestedThreshold", (score * 1.05).toFixed(5));
    this.el("modalNotesInput").value = "";
    this.el("hitlModal").classList.remove("hidden");
    this.el("modalCloseBtn").focus();
  }

  closeFeedbackModal() {
    this.el("hitlModal").classList.add("hidden");
    this.selectedAlert = null;
  }

  async handleFalseAlarm() {
    if (!this.selectedAlert) return;
    var score = Number(this.selectedAlert.score);
    var notes = this.el("modalNotesInput").value.trim() || "False alarm: transient machine surge";
    try {
      var result = await Api.submitFeedback("FALSE_ALARM", score, notes);
      if (result.adapted_threshold != null) {
        this.activeThreshold = Number(result.adapted_threshold);
        this.chart.updateThreshold(this.activeThreshold);
        this._updateThresholdDisplay(this.activeThreshold);
      }
      await this._finishReview();
      this._showToast("False alarm recorded · threshold " + Number(result.adapted_threshold).toFixed(5));
    } catch (error) { this._showToast("Feedback failed: " + error.message); }
  }

  async handleConfirmFault() {
    if (!this.selectedAlert) return;
    var score = Number(this.selectedAlert.score);
    var notes = this.el("modalNotesInput").value.trim() || "Confirmed mechanical fault";
    try {
      await Api.submitFeedback("CONFIRMED_FAULT", score, notes);
      await this._finishReview();
      this._showToast("Fault confirmed and saved to the maintenance log.");
    } catch (error) { this._showToast("Confirmation failed: " + error.message); }
  }

  async _finishReview() {
    var alert = this.selectedAlert;
    if (alert && alert.id) await Api.acknowledgeAlert(alert.id);
    this.closeFeedbackModal();
    await this.refreshAlerts();
  }

  async acknowledgeAlertById(id) {
    try {
      await Api.acknowledgeAlert(id);
      await this.refreshAlerts();
      this._showToast("Alert acknowledged.");
    } catch (error) { this._showToast("Could not acknowledge alert: " + error.message); }
  }

  async _initPush() {
    var subscribed = await pushManager.init();
    this._updatePushButton(subscribed);
  }

  _updatePushButton(subscribed) {
    var button = this.el("pushToggleBtn");
    button.classList.toggle("is-subscribed", subscribed);
    button.querySelector("span").textContent = subscribed ? "Alerts active" : "Enable alerts";
    this._replaceIcon(button, subscribed ? "bell-ring" : "bell");
  }

  async togglePush() {
    try {
      if (pushManager.hasSubscription()) {
        await pushManager.unsubscribe();
        this._updatePushButton(false);
        this._showToast("Background alerts disabled.");
      } else {
        await pushManager.subscribe();
        this._updatePushButton(true);
        this._showToast("This device is subscribed to background alerts.");
        if ("Notification" in window && Notification.permission === "granted") {
          try { new Notification("VibraGuard alerts active", { body: "This device is ready for motor alerts.", icon: "/manifest.json" }); } catch (_) {}
        }
      }
    } catch (error) { this._showToast("Push notification: " + error.message); }
  }

  async testPushChime() {
    audioManager.playAlertChime();
    if ("Notification" in window && Notification.permission === "granted") {
      try { new Notification("Test motor alert", { body: "Vibration alert sound check.", icon: "/manifest.json" }); } catch (_) {}
    }
    try {
      var result = await Api.triggerTestPush();
      this._showToast(result.subscribers_queued === 0
        ? "No subscribed devices yet. Enable alerts to register this browser."
        : "Test push sent to " + result.subscribers_queued + " device(s).");
    } catch (error) { this._showToast("Test notification failed: " + error.message); }
  }

  async _saveThreshold() {
    var value = Number(this.el("manualThresholdInput").value);
    if (!Number.isFinite(value) || value <= 0) {
      this._showToast("Enter a threshold greater than zero.");
      this.el("manualThresholdInput").focus();
      return;
    }
    try {
      await Api.updateThreshold(value);
      this.activeThreshold = value;
      this.chart.updateThreshold(value);
      this._updateThresholdDisplay(value);
      this._showToast("Threshold updated to " + value.toFixed(5) + ".");
    } catch (error) { this._showToast("Threshold update failed: " + error.message); }
  }

  _bindEvents() {
    document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") this.catchUpSync(); });
    this.el("sensorModeMockBtn").addEventListener("click", () => this.setSensorMode("mock"));
    this.el("sensorModePhysicalBtn").addEventListener("click", () => this.setSensorMode("physical"));
    this.el("modelModeUnsupBtn").addEventListener("click", () => this.setModelMode("unsupervised"));
    this.el("modelModeSuperBtn").addEventListener("click", () => this.setModelMode("supervised"));
    this.el("pushToggleBtn").addEventListener("click", () => this.togglePush());
    this.el("audioToggleBtn").addEventListener("click", (event) => {
      var muted = audioManager.toggleMute();
      this._replaceIcon(event.currentTarget, muted ? "volume-x" : "volume-2");
      event.currentTarget.setAttribute("aria-label", muted ? "Turn alert audio on" : "Mute alert audio");
      this._showToast(muted ? "Alert audio muted." : "Alert audio enabled.");
    });
    this.el("saveThresholdBtn").addEventListener("click", () => this._saveThreshold());
    this.el("manualThresholdInput").addEventListener("keydown", (event) => { if (event.key === "Enter") this._saveThreshold(); });
    this.el("testPushBtn").addEventListener("click", () => this.testPushChime());

    this.el("filterAllAlertsBtn").addEventListener("click", () => { this.activeFilter = "all"; this._setActiveFilter(); this.refreshAlerts(); });
    this.el("filterUnackAlertsBtn").addEventListener("click", () => { this.activeFilter = "unreviewed"; this._setActiveFilter(); this.refreshAlerts(); });
    this.el("alertsListContainer").addEventListener("click", (event) => {
      var button = event.target.closest("[data-action]");
      if (!button) return;
      var alert = this.alertsByKey.get(button.dataset.alertKey);
      if (!alert) return;
      if (button.dataset.action === "review") this.openFeedbackModal(alert);
      if (button.dataset.action === "ack") this.acknowledgeAlertById(alert.id);
    });

    this.el("modalFalseAlarmBtn").addEventListener("click", () => this.handleFalseAlarm());
    this.el("modalConfirmFaultBtn").addEventListener("click", () => this.handleConfirmFault());
    this.el("modalCloseBtn").addEventListener("click", () => this.closeFeedbackModal());
    this.el("modalBackdrop").addEventListener("click", () => this.closeFeedbackModal());
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !this.el("hitlModal").classList.contains("hidden")) this.closeFeedbackModal();
    });
    var audio = this.el("audioToggleBtn");
    audio.querySelector("i").setAttribute("data-lucide", audioManager.isMuted ? "volume-x" : "volume-2");
    audio.setAttribute("aria-label", audioManager.isMuted ? "Turn alert audio on" : "Mute alert audio");
    this._setActiveFilter();
  }

  _setActiveFilter() {
    this.el("filterUnackAlertsBtn").classList.toggle("is-active", this.activeFilter === "unreviewed");
    this.el("filterAllAlertsBtn").classList.toggle("is-active", this.activeFilter === "all");
  }

  _showToast(message) {
    var toast = document.createElement("div");
    toast.className = "toast";
    var icon = document.createElement("i");
    icon.setAttribute("data-lucide", "info");
    var text = document.createElement("span");
    text.textContent = message;
    toast.append(icon, text);
    this.el("toastRegion").append(toast);
    this._icons();
    window.setTimeout(() => {
      toast.classList.add("is-leaving");
      window.setTimeout(() => toast.remove(), 280);
    }, 3500);
  }
}
