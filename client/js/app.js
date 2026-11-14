/**
 * client/js/app.js — VibraGuard UI Orchestrator & State Management.
 */

document.addEventListener("DOMContentLoaded", () => {
  const app = new VibraGuardApp();
  app.init();
});

class VibraGuardApp {
  constructor() {
    this.chart = new TelemetryChart("telemetryChart");
    this.activeThreshold = 0.0506;
    this.lastPacketTimestamp = null;
    this.selectedAlert = null;
    this.alertsList = [];
    this.activeFilter = "unreviewed"; // 'all' or 'unreviewed'
    this.sensorMode = "mock";
    this.activeModel = "unsupervised";
  }

  async init() {
    console.log("[VibraGuard] Initializing dashboard...");

    // 1. Initialize Chart
    this.chart.init(this.activeThreshold);

    // 2. Setup Push Notification Manager
    this._initPush();

    // 3. Attach Event Listeners
    this._bindEvents();

    // 4. Perform Catch-Up Sync
    await this.catchUpSync();

    // 5. Connect WebSocket Live Stream
    this._connectWebSocket();

    // 6. Start Heartbeat Timer
    this._startHeartbeatTimer();
  }

  // -------------------------------------------------------------------------
  // Catch-Up Sync (Hybrid Pub/Sub Resiliency)
  // -------------------------------------------------------------------------
  async catchUpSync() {
    try {
      this._updateServerStatus(true);

      // A. Fetch current configuration
      const config = await Api.getConfig();
      if (config) {
        if (config.alert_threshold) {
          this.activeThreshold = config.alert_threshold;
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

      // B. Fetch telemetry history for chart
      const history = await Api.getTelemetryHistory(60);
      if (history && history.length > 0) {
        this.chart.loadHistory(history, this.activeThreshold);
        const latest = history[history.length - 1];
        this._renderLatestTelemetry(latest);
      } else {
        // Fallback: try latest telemetry
        try {
          const latest = await Api.getLatestTelemetry();
          if (latest) this._renderLatestTelemetry(latest);
        } catch (_) {}
      }

      // C. Fetch alerts for triage queue
      await this.refreshAlerts();
    } catch (err) {
      console.warn("[App] Catch-up sync warning:", err);
      this._updateServerStatus(false);
    }
  }

  // -------------------------------------------------------------------------
  // WebSocket Live Stream Handling
  // -------------------------------------------------------------------------
  _connectWebSocket() {
    wsClient.onStatusChange = (status) => {
      const statusPill = document.getElementById("streamStatusPill");
      if (!statusPill) return;

      if (status === "connected") {
        statusPill.className = "flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-950/60 text-emerald-400 border border-emerald-800/60";
        statusPill.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400 pulse-emerald"></span> Stream: Live`;
      } else if (status === "connecting") {
        statusPill.className = "flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-amber-950/60 text-amber-400 border border-amber-800/60";
        statusPill.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span> Stream: Reconnecting`;
      } else {
        statusPill.className = "flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-rose-950/60 text-rose-400 border border-rose-800/60";
        statusPill.innerHTML = `<span class="w-2 h-2 rounded-full bg-rose-400"></span> Stream: Disconnected`;
      }
    };

    wsClient.onTelemetry = (data) => {
      this._renderLatestTelemetry(data);
      this.chart.pushPoint(data.timestamp || data.ts, data.score, data.label);
    };

    wsClient.onAlert = (data) => {
      audioManager.playAlertChime();
      this.refreshAlerts();
      this.openFeedbackModal(data);
      if ("Notification" in window && Notification.permission === "granted") {
        try {
          new Notification("🚨 MOTOR ANOMALY ALERT", {
            body: data.message || `Bearing anomaly score: ${Number(data.score).toFixed(4)}`,
            icon: "/manifest.json",
          });
        } catch (_) {}
      }
    };

    wsClient.onConfigUpdate = (data) => {
      if (data && data.alert_threshold) {
        this.activeThreshold = data.alert_threshold;
        this.chart.updateThreshold(this.activeThreshold);
        this._updateThresholdDisplay(this.activeThreshold);
        this._showToast(`Threshold adapted: ${this.activeThreshold.toFixed(5)} (${data.reason || "update"})`);
      }
      if (data && data.sensor_mode) {
        this.sensorMode = data.sensor_mode;
        this._updateSensorModeUI(this.sensorMode);
        this._showToast(`Sensor switched to ${this.sensorMode.toUpperCase()} mode`);
      }
      if (data && data.active_model) {
        this.activeModel = data.active_model;
        this._updateModelUI(this.activeModel);
        this._showToast(`Model switched to ${this.activeModel.toUpperCase()}`);
      }
    };

    wsClient.connect();
  }

  // -------------------------------------------------------------------------
  // UI Renderers & Visual State
  // -------------------------------------------------------------------------
  _renderLatestTelemetry(data) {
    if (!data) return;

    this.lastPacketTimestamp = data.timestamp || data.ts || Date.now() / 1000;

    const score = parseFloat(data.score) || 0.0;
    const diagnosis = data.diagnosis || (data.label === "normal" ? "healthy" : "anomaly");
    const confidence = (parseFloat(data.confidence) || 1.0) * 100;
    const procTimeMs = ((parseFloat(data.proc_time_s) || 0.0) * 1000).toFixed(2);

    // Update KPI Numbers
    document.getElementById("statScore").textContent = score.toFixed(5);
    document.getElementById("statDiagnosis").textContent = this._formatDiagnosis(diagnosis);
    document.getElementById("statConfidence").textContent = `${confidence.toFixed(1)}%`;
    document.getElementById("statLatency").textContent = `${procTimeMs} ms`;

    // Progress Bar (Percent of Threshold)
    const pctOfThreshold = Math.min(100, Math.round((score / this.activeThreshold) * 100));
    const progressBar = document.getElementById("statThresholdProgress");
    const progressLabel = document.getElementById("statThresholdProgressLabel");
    if (progressBar && progressLabel) {
      progressBar.style.width = `${pctOfThreshold}%`;
      progressLabel.textContent = `${pctOfThreshold}% of threshold`;

      if (score > this.activeThreshold) {
        progressBar.className = "h-full bg-rose-500 rounded-full transition-all duration-300";
      } else if (score > this.activeThreshold * 0.85) {
        progressBar.className = "h-full bg-amber-500 rounded-full transition-all duration-300";
      } else {
        progressBar.className = "h-full bg-cyan-500 rounded-full transition-all duration-300";
      }
    }

    // Main Status Banner Badge
    const heroCard = document.getElementById("heroStatusCard");
    const statusText = document.getElementById("heroStatusText");
    const statusDesc = document.getElementById("heroStatusDesc");
    const healthBadge = document.getElementById("heroHealthBadge");

    if (score > this.activeThreshold) {
      heroCard.className = "glass-card p-6 rounded-2xl border pulse-critical bg-rose-950/20";
      statusText.textContent = "CRITICAL FAULT DETECTED";
      statusText.className = "text-xl font-bold tracking-wide text-rose-400";
      statusDesc.textContent = `Vibration anomaly exceeded safety threshold of ${this.activeThreshold.toFixed(4)}. Immediate inspection required.`;
      healthBadge.className = "text-center px-4 py-3 rounded-xl bg-rose-900/40 border border-rose-700/60";
      healthBadge.innerHTML = `<span class="block text-2xl font-black text-rose-400 font-mono">FAULT</span><span class="text-xs text-rose-300/80 font-medium uppercase tracking-wider">${diagnosis}</span>`;
    } else if (score > this.activeThreshold * 0.85) {
      heroCard.className = "glass-card p-6 rounded-2xl border border-amber-700/50 bg-amber-950/15";
      statusText.textContent = "WARNING: ELEVATED VIBRATION";
      statusText.className = "text-xl font-bold tracking-wide text-amber-400";
      statusDesc.textContent = `Vibration approaching threshold (${pctOfThreshold}%). Monitoring trending harmonics.`;
      healthBadge.className = "text-center px-4 py-3 rounded-xl bg-amber-900/40 border border-amber-700/60";
      healthBadge.innerHTML = `<span class="block text-2xl font-black text-amber-400 font-mono">${(100 - pctOfThreshold / 2).toFixed(0)}%</span><span class="text-xs text-amber-300/80 font-medium uppercase tracking-wider">ELEVATED</span>`;
    } else {
      heroCard.className = "glass-card p-6 rounded-2xl border border-slate-800 bg-slate-900/40";
      statusText.textContent = "MOTOR STATUS: NORMAL OPERATION";
      statusText.className = "text-xl font-bold tracking-wide text-emerald-400";
      statusDesc.textContent = "Real-time FFT reconstruction error within calibrated baseline statistics.";
      healthBadge.className = "text-center px-4 py-3 rounded-xl bg-emerald-950/60 border border-emerald-800/60";
      healthBadge.innerHTML = `<span class="block text-2xl font-black text-emerald-400 font-mono">HEALTHY</span><span class="text-xs text-emerald-300/80 font-medium uppercase tracking-wider">100% BASELINE</span>`;
    }
  }

  _updateThresholdDisplay(val) {
    const el = document.getElementById("statThreshold");
    if (el) el.textContent = val.toFixed(5);
    const input = document.getElementById("manualThresholdInput");
    if (input) input.value = val.toFixed(5);
  }

  _updateSensorModeUI(mode) {
    const mockBtn = document.getElementById("sensorModeMockBtn");
    const physBtn = document.getElementById("sensorModePhysicalBtn");
    if (!mockBtn || !physBtn) return;

    if (mode === "physical") {
      physBtn.className = "px-2.5 py-1 rounded-lg font-bold text-xs bg-emerald-500 text-black transition-all shadow-sm";
      mockBtn.className = "px-2.5 py-1 rounded-lg font-bold text-xs text-slate-400 hover:text-slate-200 transition-all";
    } else {
      mockBtn.className = "px-2.5 py-1 rounded-lg font-bold text-xs bg-cyan-500 text-black transition-all shadow-sm";
      physBtn.className = "px-2.5 py-1 rounded-lg font-bold text-xs text-slate-400 hover:text-slate-200 transition-all";
    }
  }

  async setSensorMode(mode) {
    try {
      const res = await Api.setSensorMode(mode);
      this.sensorMode = mode;
      this._updateSensorModeUI(mode);
      this._showToast(`Sensor switched to ${mode.toUpperCase()} mode (MQTT: ${res.mqtt_broadcast ? "OK" : "Pending"})`);
    } catch (e) {
      alert(`Could not change sensor mode: ${e.message}`);
    }
  }

  _updateModelUI(model) {
    const unsupBtn = document.getElementById("modelModeUnsupBtn");
    const superBtn = document.getElementById("modelModeSuperBtn");
    if (!unsupBtn || !superBtn) return;

    if (model === "supervised") {
      superBtn.className = "px-2.5 py-1 rounded-lg font-bold text-xs bg-amber-400 text-black transition-all shadow-sm";
      unsupBtn.className = "px-2.5 py-1 rounded-lg font-bold text-xs text-slate-400 hover:text-slate-200 transition-all";
    } else {
      unsupBtn.className = "px-2.5 py-1 rounded-lg font-bold text-xs bg-emerald-500 text-black transition-all shadow-sm";
      superBtn.className = "px-2.5 py-1 rounded-lg font-bold text-xs text-slate-400 hover:text-slate-200 transition-all";
    }
  }

  async setModelMode(model) {
    try {
      const res = await Api.setModelMode(model);
      this.activeModel = model;
      this._updateModelUI(model);
      this._showToast(`Edge model switched to ${model.toUpperCase()} (MQTT: ${res.mqtt_broadcast ? "OK" : "Pending"})`);
    } catch (e) {
      alert(`Could not change model: ${e.message}`);
    }
  }

  _updateServerStatus(isOnline) {
    const pill = document.getElementById("serverStatusPill");
    if (!pill) return;
    if (isOnline) {
      pill.className = "flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-950/60 text-emerald-400 border border-emerald-800/60";
      pill.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400"></span> Server: Online`;
    } else {
      pill.className = "flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-rose-950/60 text-rose-400 border border-rose-800/60";
      pill.innerHTML = `<span class="w-2 h-2 rounded-full bg-rose-400"></span> Server: Unreachable`;
    }
  }

  _startHeartbeatTimer() {
    setInterval(() => {
      const pill = document.getElementById("edgeHeartbeatPill");
      if (!pill || !this.lastPacketTimestamp) return;

      const elapsedSec = Math.floor(Date.now() / 1000 - this.lastPacketTimestamp);

      if (elapsedSec < 15) {
        pill.className = "flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-950/60 text-emerald-400 border border-emerald-800/60";
        pill.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400 pulse-emerald"></span> Edge Pi: ${elapsedSec}s ago`;
      } else if (elapsedSec < 45) {
        pill.className = "flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-amber-950/60 text-amber-400 border border-amber-800/60";
        pill.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400"></span> Edge Pi: ${elapsedSec}s ago`;
      } else {
        pill.className = "flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-rose-950/60 text-rose-400 border border-rose-800/60";
        pill.innerHTML = `<span class="w-2 h-2 rounded-full bg-rose-400"></span> Edge Pi: Offline (${elapsedSec}s)`;
      }
    }, 1000);
  }

  _formatDiagnosis(raw) {
    if (!raw) return "Healthy";
    const map = {
      healthy: "Healthy",
      ball_fault: "Ball Bearing Fault",
      inner_race_fault: "Inner Race Fault",
      outer_race_fault: "Outer Race Fault",
      anomaly: "Vibration Anomaly",
    };
    return map[raw] || raw;
  }

  // -------------------------------------------------------------------------
  // Alerts Queue & Triage
  // -------------------------------------------------------------------------
  async refreshAlerts() {
    try {
      const unackOnly = this.activeFilter === "unreviewed";
      const alerts = await Api.getAlerts(unackOnly, 20);
      this.alertsList = alerts;

      const countBadge = document.getElementById("unackAlertCountBadge");
      const unackCount = alerts.filter((a) => !a.acknowledged).length;
      if (countBadge) {
        countBadge.textContent = unackCount;
        countBadge.className = unackCount > 0
          ? "px-2 py-0.5 rounded-full text-xs font-bold bg-rose-500 text-white animate-pulse"
          : "px-2 py-0.5 rounded-full text-xs font-bold bg-slate-800 text-slate-400";
      }

      this._renderAlertsTable(alerts);
    } catch (e) {
      console.warn("[Alerts] Could not load alerts list:", e);
    }
  }

  _renderAlertsTable(alerts) {
    const container = document.getElementById("alertsListContainer");
    if (!container) return;

    if (!alerts || alerts.length === 0) {
      container.innerHTML = `
        <div class="py-8 text-center text-slate-500 text-sm">
          <p>No ${this.activeFilter === "unreviewed" ? "unreviewed" : ""} alerts recorded.</p>
        </div>
      `;
      return;
    }

    container.innerHTML = alerts
      .map((alert) => {
        const timeStr = new Date(alert.timestamp * 1000).toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        });
        const isAck = alert.acknowledged;

        return `
          <div class="flex items-center justify-between p-3 rounded-xl bg-slate-900/60 border ${
            isAck ? "border-slate-800/40 text-slate-400" : "border-rose-900/60 bg-rose-950/10"
          } transition-all">
            <div class="flex items-center gap-3">
              <span class="w-2.5 h-2.5 rounded-full ${isAck ? "bg-slate-600" : "bg-rose-500 animate-ping"}"></span>
              <div>
                <div class="flex items-center gap-2">
                  <span class="font-semibold text-sm ${isAck ? "text-slate-400" : "text-slate-200"}">
                    ${this._formatDiagnosis(alert.diagnosis)}
                  </span>
                  <span class="font-mono text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                    Score: ${alert.score.toFixed(4)}
                  </span>
                </div>
                <div class="text-xs text-slate-500 mt-0.5">Triggered at ${timeStr}</div>
              </div>
            </div>

            <div class="flex items-center gap-2">
              <button onclick="window.app.openFeedbackModal(${JSON.stringify(alert).replace(/"/g, '&quot;')})" 
                class="px-2.5 py-1 text-xs font-medium rounded-lg bg-cyan-950/80 text-cyan-300 border border-cyan-800/80 hover:bg-cyan-900 transition-all">
                Triage
              </button>
              ${
                !isAck
                  ? `<button onclick="window.app.acknowledgeAlertById(${alert.id})" 
                      class="px-2 py-1 text-xs rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 transition-all">
                      Ack
                    </button>`
                  : `<span class="text-xs text-emerald-500/80 font-medium">✓ Done</span>`
              }
            </div>
          </div>
        `;
      })
      .join("");
  }

  // -------------------------------------------------------------------------
  // 👨‍🔧 Human-in-the-Loop Feedback Logic
  // -------------------------------------------------------------------------
  openFeedbackModal(alertData) {
    this.selectedAlert = alertData;

    const modal = document.getElementById("hitlModal");
    if (!modal) return;

    const score = parseFloat(alertData.score) || 0.0;
    const diagnosis = this._formatDiagnosis(alertData.diagnosis);
    const suggestedThreshold = (score * 1.05).toFixed(5);

    document.getElementById("modalAlertScore").textContent = score.toFixed(5);
    document.getElementById("modalAlertThreshold").textContent = this.activeThreshold.toFixed(5);
    document.getElementById("modalAlertDiagnosis").textContent = diagnosis;
    document.getElementById("modalSuggestedThreshold").textContent = suggestedThreshold;
    document.getElementById("modalNotesInput").value = "";

    modal.classList.remove("hidden");
  }

  closeFeedbackModal() {
    const modal = document.getElementById("hitlModal");
    if (modal) modal.classList.add("hidden");
    this.selectedAlert = null;
  }

  async handleFalseAlarm() {
    if (!this.selectedAlert) return;

    const score = parseFloat(this.selectedAlert.score);
    const notes = document.getElementById("modalNotesInput").value.trim() || "False alarm: transient machine surge";

    try {
      const res = await Api.submitFeedback("FALSE_ALARM", score, notes);
      this._showToast(`Success! Edge threshold adapted to ${res.adapted_threshold.toFixed(5)}`);

      if (this.selectedAlert.id) {
        await Api.acknowledgeAlert(this.selectedAlert.id);
      }

      this.closeFeedbackModal();
      await this.refreshAlerts();
    } catch (err) {
      alert(`Feedback submission failed: ${err.message}`);
    }
  }

  async handleConfirmFault() {
    if (!this.selectedAlert) return;

    const score = parseFloat(this.selectedAlert.score);
    const notes = document.getElementById("modalNotesInput").value.trim() || "Confirmed bearing failure";

    try {
      await Api.submitFeedback("CONFIRMED_FAULT", score, notes);
      this._showToast("Fault confirmed & logged for maintenance overhaul.");

      if (this.selectedAlert.id) {
        await Api.acknowledgeAlert(this.selectedAlert.id);
      }

      this.closeFeedbackModal();
      await this.refreshAlerts();
    } catch (err) {
      alert(`Confirmation failed: ${err.message}`);
    }
  }

  async acknowledgeAlertById(alertId) {
    try {
      await Api.acknowledgeAlert(alertId);
      await this.refreshAlerts();
      this._showToast("Alert marked as acknowledged.");
    } catch (e) {
      console.error(e);
    }
  }

  // -------------------------------------------------------------------------
  // Push Notifications Setup
  // -------------------------------------------------------------------------
  async _initPush() {
    const isSubscribed = await pushManager.init();
    this._updatePushBtn(isSubscribed);
  }

  _updatePushBtn(isSubscribed) {
    const btn = document.getElementById("pushToggleBtn");
    if (!btn) return;

    if (isSubscribed) {
      btn.className = "flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-emerald-950/70 text-emerald-400 border border-emerald-800/80 hover:bg-emerald-900 transition-all";
      btn.innerHTML = `✓ Push Active`;
    } else {
      btn.className = "flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-cyan-950/70 text-cyan-400 border border-cyan-800/80 hover:bg-cyan-900 transition-all";
      btn.innerHTML = `🔔 Enable Push`;
    }
  }

  async togglePush() {
    try {
      if (pushManager.hasSubscription()) {
        await pushManager.unsubscribe();
        this._updatePushBtn(false);
        this._showToast("Unsubscribed from background push alerts.");
      } else {
        await pushManager.subscribe();
        this._updatePushBtn(true);
        this._showToast("✓ Subscribed! You will now receive background alerts.");
        // Prompt immediate local notification confirmation
        if ("Notification" in window && Notification.permission === "granted") {
          try {
            new Notification("🔔 VibraGuard Push Alerts Active", {
              body: "Device registered for real-time bearing anomaly warnings.",
              icon: "/manifest.json",
            });
          } catch (_) {}
        }
      }
    } catch (err) {
      console.error("[App] Push subscription error:", err);
      alert(`Push Notification: ${err.message}`);
    }
  }

  async testPushChime() {
    try {
      // 1. Play synthesized audio chime immediately
      audioManager.playAlertChime();

      // 2. Display local browser notification if permitted
      if ("Notification" in window && Notification.permission === "granted") {
        try {
          new Notification("🚨 TEST MOTOR ALERT", {
            body: "Vibration threshold breach simulated! Hardware alarm active.",
            icon: "/manifest.json",
          });
        } catch (_) {}
      }

      // 3. Dispatch web push to all registered endpoints on server
      const res = await Api.triggerTestPush();
      if (res && res.subscribers_queued === 0) {
        this._showToast("⚠️ Device not subscribed yet. Click '🔔 Enable Push' above to register your browser!", 5000);
      } else {
        this._showToast(`🚀 Cloud push dispatched to ${res.subscribers_queued || 1} device(s)!`);
      }
    } catch (e) {
      alert(`Test push error: ${e.message}`);
    }
  }

  // -------------------------------------------------------------------------
  // UI Event Bindings
  // -------------------------------------------------------------------------
  _bindEvents() {
    // Window Focus / Visibility Change (Catch-Up Sync trigger)
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible") {
        console.log("[App] Tab resumed; running catch-up sync...");
        this.catchUpSync();
      }
    });

    // Sensor Mode Toggle Buttons
    const mockBtn = document.getElementById("sensorModeMockBtn");
    const physBtn = document.getElementById("sensorModePhysicalBtn");
    if (mockBtn) mockBtn.addEventListener("click", () => this.setSensorMode("mock"));
    if (physBtn) physBtn.addEventListener("click", () => this.setSensorMode("physical"));

    // Model Mode Toggle Buttons
    const unsupBtn = document.getElementById("modelModeUnsupBtn");
    const superBtn = document.getElementById("modelModeSuperBtn");
    if (unsupBtn) unsupBtn.addEventListener("click", () => this.setModelMode("unsupervised"));
    if (superBtn) superBtn.addEventListener("click", () => this.setModelMode("supervised"));

    // Push Toggle Button
    const pushBtn = document.getElementById("pushToggleBtn");
    if (pushBtn) pushBtn.addEventListener("click", () => this.togglePush());

    // Audio Mute Toggle
    const audioBtn = document.getElementById("audioToggleBtn");
    if (audioBtn) {
      audioBtn.addEventListener("click", () => {
        const isMuted = audioManager.toggleMute();
        audioBtn.textContent = isMuted ? "🔇 Audio Muted" : "🔊 Audio On";
      });
      audioBtn.textContent = audioManager.isMuted ? "🔇 Audio Muted" : "🔊 Audio On";
    }

    // Manual Threshold Save
    const saveThreshBtn = document.getElementById("saveThresholdBtn");
    if (saveThreshBtn) {
      saveThreshBtn.addEventListener("click", async () => {
        const val = parseFloat(document.getElementById("manualThresholdInput").value);
        if (isNaN(val) || val <= 0) {
          alert("Please enter a valid threshold > 0");
          return;
        }
        try {
          await Api.updateThreshold(val);
          this.activeThreshold = val;
          this.chart.updateThreshold(val);
          this._updateThresholdDisplay(val);
          this._showToast(`Threshold updated to ${val.toFixed(5)}`);
        } catch (e) {
          alert(e.message);
        }
      });
    }

    // Alert filter buttons
    const filterAllBtn = document.getElementById("filterAllAlertsBtn");
    const filterUnackBtn = document.getElementById("filterUnackAlertsBtn");
    if (filterAllBtn && filterUnackBtn) {
      filterAllBtn.addEventListener("click", () => {
        this.activeFilter = "all";
        filterAllBtn.className = "px-3 py-1 rounded-lg text-xs font-semibold bg-slate-800 text-slate-200";
        filterUnackBtn.className = "px-3 py-1 rounded-lg text-xs font-semibold text-slate-400 hover:text-slate-200";
        this.refreshAlerts();
      });
      filterUnackBtn.addEventListener("click", () => {
        this.activeFilter = "unreviewed";
        filterUnackBtn.className = "px-3 py-1 rounded-lg text-xs font-semibold bg-slate-800 text-slate-200";
        filterAllBtn.className = "px-3 py-1 rounded-lg text-xs font-semibold text-slate-400 hover:text-slate-200";
        this.refreshAlerts();
      });
    }

    // HITL Modal buttons
    const falseAlarmBtn = document.getElementById("modalFalseAlarmBtn");
    if (falseAlarmBtn) falseAlarmBtn.addEventListener("click", () => this.handleFalseAlarm());

    const confirmFaultBtn = document.getElementById("modalConfirmFaultBtn");
    if (confirmFaultBtn) confirmFaultBtn.addEventListener("click", () => this.handleConfirmFault());

    const closeModalBtn = document.getElementById("modalCloseBtn");
    if (closeModalBtn) closeModalBtn.addEventListener("click", () => this.closeFeedbackModal());

    window.app = this;
  }

  _showToast(msg) {
    const toast = document.createElement("div");
    toast.className = "fixed bottom-5 right-5 z-50 px-4 py-3 rounded-xl bg-slate-900 border border-cyan-800/80 text-cyan-300 text-xs font-medium shadow-2xl backdrop-blur-md transition-all";
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }
}
