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
    gradient.addColorStop(0, "rgba(49, 95, 232, 0.16)");
    gradient.addColorStop(1, "rgba(49, 95, 232, 0.0)");

    this.chart = new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Anomaly Score",
            data: [],
            borderColor: "#315fe8",
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
              return val > this.currentThreshold ? "#c33c4e" : "#315fe8";
            },
          },
          {
            label: "Threshold (τ)",
            data: [],
            borderColor: "#d9969b",
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
            labels: { color: "#7782a0", boxWidth: 14, font: { family: "DM Sans", size: 10 } },
          },
          tooltip: {
            backgroundColor: "rgba(20, 31, 64, 0.95)",
            borderColor: "#314f9f",
            borderWidth: 1,
            titleColor: "#ffffff",
            bodyColor: "#d5def5",
          },
        },
        scales: {
          x: {
            grid: { color: "rgba(30, 48, 95, 0.07)" },
            border: { display: false },
            ticks: { color: "#929bb0", maxTicksLimit: 8, font: { family: "DM Sans", size: 9 } },
          },
          y: {
            grid: { color: "rgba(30, 48, 95, 0.07)" },
            border: { display: false },
            ticks: { color: "#929bb0", font: { family: "DM Sans", size: 9 } },
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
