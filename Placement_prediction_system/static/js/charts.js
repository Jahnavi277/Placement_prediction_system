(function () {
  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function baseColors() {
    return {
      success: cssVar("--success"),
      danger: cssVar("--danger"),
      info: cssVar("--info"),
      accent: cssVar("--accent"),
      muted: cssVar("--text-muted"),
      border: cssVar("--border"),
      text: cssVar("--text"),
    };
  }

  Chart.defaults.font.family = "'Public Sans', sans-serif";
  Chart.defaults.font.size = 12;

  window.SP_CHARTS = {};

  window.initDonutChart = function (canvasId, placedPct, notPlacedPct) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    const c = baseColors();
    window.SP_CHARTS[canvasId] = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: ["Placed", "Not Placed"],
        datasets: [
          {
            data: [placedPct, notPlacedPct],
            backgroundColor: [c.success, c.danger],
            borderWidth: 0,
            hoverOffset: 6,
          },
        ],
      },
      options: {
        cutout: "68%",
        plugins: { legend: { display: false } },
        animation: { animateRotate: true, duration: 900 },
      },
    });
  };

  window.initBarChart = function (canvasId, labels, values, colorKey) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    const c = baseColors();
    window.SP_CHARTS[canvasId] = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            data: values,
            backgroundColor: c[colorKey] || c.info,
            borderRadius: 5,
            maxBarThickness: 34,
          },
        ],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false }, ticks: { color: c.muted } },
          y: { grid: { color: c.border }, ticks: { color: c.muted }, beginAtZero: true },
        },
        animation: { duration: 800 },
      },
    });
  };

  window.initPieChart = function (canvasId, labels, values, colors) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    window.SP_CHARTS[canvasId] = new Chart(ctx, {
      type: "pie",
      data: {
        labels: labels,
        datasets: [{ data: values, backgroundColor: colors, borderWidth: 0 }],
      },
      options: {
        plugins: { legend: { position: "bottom", labels: { boxWidth: 10, color: baseColors().muted } } },
        animation: { duration: 800 },
      },
    });
  };

  window.initRadarChart = function (canvasId, labels, values) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    const c = baseColors();
    window.SP_CHARTS[canvasId] = new Chart(ctx, {
      type: "radar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Profile",
            data: values,
            backgroundColor: "rgba(184, 144, 31, 0.18)",
            borderColor: c.accent,
            borderWidth: 2,
            pointBackgroundColor: c.accent,
            pointRadius: 3,
          },
        ],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          r: {
            min: 0,
            max: 100,
            angleLines: { color: c.border },
            grid: { color: c.border },
            pointLabels: { color: c.muted, font: { size: 10.5 } },
            ticks: { display: false, backdropColor: "transparent" },
          },
        },
        animation: { duration: 800 },
      },
    });
  };

  window.initLineChart = function (canvasId, labels, values) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    const c = baseColors();
    window.SP_CHARTS[canvasId] = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            data: values,
            borderColor: c.info,
            backgroundColor: "rgba(53, 87, 143, 0.12)",
            borderWidth: 2.5,
            fill: true,
            tension: 0.35,
            pointBackgroundColor: c.info,
            pointRadius: 4,
          },
        ],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false }, ticks: { color: c.muted } },
          y: { grid: { color: c.border }, ticks: { color: c.muted }, suggestedMin: 0 },
        },
        animation: { duration: 800 },
      },
    });
  };

  window.initProbBars = function (placedPct) {
    const placedEl = document.getElementById("probPlacedFill");
    const notEl = document.getElementById("probNotFill");
    if (placedEl) requestAnimationFrame(() => (placedEl.style.width = placedPct + "%"));
    if (notEl) requestAnimationFrame(() => (notEl.style.width = 100 - placedPct + "%"));
  };
})();
