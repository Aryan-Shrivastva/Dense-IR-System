// Dashboard charts — all data from real API
let scoreChart, blendChart, rankChart;

async function loadAnalytics() {
  try {
    const data = await apiAnalytics();
    document.getElementById("d-searches").textContent = data.total_searches;
    document.getElementById("d-clicks").textContent   = data.total_clicks;
    document.getElementById("d-alpha").textContent    = data.avg_alpha.toFixed(2);
    document.getElementById("d-latency").textContent  = data.avg_latency_ms + "ms";

    const qList = document.getElementById("top-queries");
    qList.innerHTML = (data.top_queries || []).map(q =>
      `<div class="query-row"><span>${q.query}</span><span>${q.cnt}×</span></div>`
    ).join("");
  } catch (e) {
    console.error("Analytics error:", e);
  }
}

function renderScoreChart(colbertScores) {
  if (scoreChart) scoreChart.destroy();
  const ctx = document.getElementById("scoreChart").getContext("2d");
  scoreChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: colbertScores.map((_, i) => i + 1),
      datasets: [{
        label: "ColBERT MaxSim",
        data: colbertScores.map(s => parseFloat(s.toFixed(4))),
        borderColor: "#534AB7",
        backgroundColor: "#EEEDFE",
        borderWidth: 2, pointRadius: 2, fill: true, tension: 0.3,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: "rank", font: { size: 10 } } },
        y: { min: 0, ticks: { font: { size: 10 } } }
      }
    }
  });
}

function renderBlendChart(alpha) {
  if (blendChart) blendChart.destroy();
  const ctx = document.getElementById("blendChart").getContext("2d");
  blendChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Semantic (ColBERT)", "User interest"],
      datasets: [{
        data: [
          parseFloat((alpha * 100).toFixed(1)),
          parseFloat(((1 - alpha) * 100).toFixed(1))
        ],
        backgroundColor: ["#534AB7", "#1D9E75"],
        borderWidth: 0,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: "60%",
      plugins: { legend: { display: true, position: "bottom",
        labels: { font: { size: 11 }, boxWidth: 12 } } }
    }
  });
}

function renderRankChart(results) {
  if (rankChart) rankChart.destroy();
  const ctx = document.getElementById("rankChart").getContext("2d");
  rankChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: results.map(r => `#${r.rank} ${(r.title || "").split(" ").slice(0, 3).join(" ")}`),
      datasets: [
        {
          label: "Semantic",
          data: results.map(r => parseFloat(r.semantic_score.toFixed(4))),
          backgroundColor: "#534AB799",
          borderColor: "#534AB7", borderWidth: 1,
        },
        {
          label: "Hybrid final",
          data: results.map(r => parseFloat(r.final_score.toFixed(4))),
          backgroundColor: "#1D9E7599",
          borderColor: "#1D9E75", borderWidth: 1,
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: true, labels: { font: { size: 11 }, boxWidth: 12 } } },
      scales: {
        x: { ticks: { font: { size: 9 }, maxRotation: 35, autoSkip: false } },
        y: { min: 0, max: 1, ticks: { font: { size: 10 }, stepSize: 0.2 } }
      }
    }
  });
}

function renderTokenGrid(tokenWeights) {
  const grid = document.getElementById("token-grid");
  if (!tokenWeights || tokenWeights.length === 0) {
    grid.innerHTML = "<span style='color:#888;font-size:13px'>Run a search first</span>";
    return;
  }
  const max = tokenWeights[0].weight;
  grid.innerHTML = tokenWeights.map(t => {
    const norm = t.weight / (max || 1);
    const bg   = norm > 0.7 ? "#EEEDFE" : norm > 0.4 ? "#E1F5EE" : "#F1EFE8";
    const col  = norm > 0.7 ? "#3C3489" : norm > 0.4 ? "#085041" : "#5F5E5A";
    const size = Math.round(11 + norm * 4);
    return `<span class="token" style="background:${bg};color:${col};font-size:${size}px">
      ${t.token} <span style="opacity:.6;font-size:10px">${t.weight.toFixed(2)}</span>
    </span>`;
  }).join("");
}

// Listen for searches from the main page (if both pages are open)
window.addEventListener("searchComplete", e => {
  const data = e.detail;
  renderScoreChart(data.results.map(r => r.colbert_score));
  renderBlendChart(data.alpha);
  renderRankChart(data.results);
  renderTokenGrid(data.token_weights || []);
  loadAnalytics();
});

// Auto-refresh analytics every 10 seconds
loadAnalytics();
setInterval(loadAnalytics, 10000);
