/**
 * client/js/charts.js — Real-Time Streaming Chart.js Manager.
 */

class TelemetryChart {
  constructor(canvasId) {
    this.canvasId = canvasId;
    this.chart = null;
    this.maxPoints = 60;
    this.currentThreshold = 0.05;
  }

  init(initialThreshold = 0.05) {
    this.currentThreshold = initialThreshold;
    const ctx = document.getElementById(this.canvasId).getContext("2d");

    // Create subtle gradient for under the curve
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, "rgba(6, 182, 212, 0.25)");
    gradient.addColorStop(1, "rgba(6, 182, 212, 0.0)");

    this.chart = new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Anomaly Score",
            data: [],
            borderColor: "#06b6d4",
            backgroundColor: gradient,
            borderWidth: 2,
            tension: 0.3,
            fill: true,
            pointRadius: (ctx) => {
              const val = ctx.raw;
              return val > this.currentThreshold ? 5 : 2;
            },
            pointBackgroundColor: (ctx) => {
              const val = ctx.raw;
              return val > this.currentThreshold ? "#ef4444" : "#06b6d4";
            },
          },
          {
            label: "Threshold (τ)",
            data: [],
            borderColor: "#f43f5e",
            borderWidth: 1.5,
            borderDash: [6, 4],
            fill: false,
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            display: true,
            labels: { color: "#94a3b8", boxWidth: 14, font: { size: 11 } },
          },
          tooltip: {
            backgroundColor: "rgba(15, 23, 42, 0.9)",
            borderColor: "#334155",
            borderWidth: 1,
            titleColor: "#f8fafc",
            bodyColor: "#94a3b8",
          },
        },
        scales: {
          x: {
            grid: { color: "rgba(30, 41, 59, 0.5)" },
            ticks: { color: "#64748b", maxTicksLimit: 8, font: { size: 10 } },
          },
          y: {
            grid: { color: "rgba(30, 41, 59, 0.5)" },
            ticks: { color: "#64748b", font: { size: 10 } },
            suggestedMin: 0.0,
            suggestedMax: 0.1,
          },
        },
      },
    });
  }

  pushPoint(timestamp, score, label) {
    if (!this.chart) return;

    const timeStr = new Date(timestamp * 1000).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });

    const labels = this.chart.data.labels;
    const scoreData = this.chart.data.datasets[0].data;
    const thresholdData = this.chart.data.datasets[1].data;

    labels.push(timeStr);
    scoreData.push(score);
    thresholdData.push(this.currentThreshold);

    if (labels.length > this.maxPoints) {
      labels.shift();
      scoreData.shift();
      thresholdData.shift();
    }

    // Auto-adjust scale max if anomaly is high
    const maxVal = Math.max(...scoreData, this.currentThreshold);
    this.chart.options.scales.y.suggestedMax = Math.max(0.1, maxVal * 1.25);

    this.chart.update("none"); // update without sluggish animation
  }

  updateThreshold(newThreshold) {
    this.currentThreshold = newThreshold;
    if (this.chart) {
      this.chart.data.datasets[1].data = this.chart.data.datasets[1].data.map(() => newThreshold);
      const maxVal = Math.max(...this.chart.data.datasets[0].data, this.currentThreshold);
      this.chart.options.scales.y.suggestedMax = Math.max(0.1, maxVal * 1.25);
      this.chart.update();
    }
  }

  loadHistory(historyArray, activeThreshold) {
    if (!this.chart || !Array.isArray(historyArray)) return;

    this.currentThreshold = activeThreshold || this.currentThreshold;
    const labels = [];
    const scoreData = [];
    const thresholdData = [];

    historyArray.forEach((item) => {
      const timeStr = new Date(item.timestamp * 1000).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
      labels.push(timeStr);
      scoreData.push(item.score);
      thresholdData.push(this.currentThreshold);
    });

    this.chart.data.labels = labels;
    this.chart.data.datasets[0].data = scoreData;
    this.chart.data.datasets[1].data = thresholdData;

    const maxVal = Math.max(...scoreData, this.currentThreshold);
    this.chart.options.scales.y.suggestedMax = Math.max(0.1, maxVal * 1.25);
    this.chart.update();
  }
}

window.TelemetryChart = TelemetryChart;
