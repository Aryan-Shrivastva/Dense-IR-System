// Charts for the analytics dashboard — all from real API data
let scoreChart, blendChart, rankChart;

async function loadAnalytics() {
  try {
    const data = await apiAnalytics();
    document.getElementById("d-searches").textContent = data.total_searches;
    document.getElementById("d-clicks").textContent = data.total_clicks;
    document.getElementById("d-alpha").textContent = data.avg_alpha.toFixed(2);
    document.getElementById("d-latency").textContent = data.avg_latency_ms + "ms";
    const qList = document.getElementById("top-queries");
    qList.innerHTML = (data.top_queries || []).map(q =>
      `<div class="query-row"><span>${q.query}</span><span>${q.cnt}</span></div>`
    ).join("");
  } catch (e) {
    console.error("Analytics error:", e);
  }
}

function renderScoreChart(colbertScores, stage1Count) {
  if (scoreChart) scoreChart.destroy();
  const ctx = document.getElementById("scoreChart");
  if (!ctx) return;

  // Build labels: show all stage-1 candidates (rank positions)
  const labels = [];
  const dataPoints = [];
  for (let i = 0; i < stage1Count; i++) {
    labels.push(i + 1);
    // If we have a ColBERT score for this rank, use it; otherwise 0
    dataPoints.push(i < colbertScores.length ? parseFloat(colbertScores[i].toFixed(4)) : 0);
  }

  const primary = getComputedStyle(document.documentElement).getPropertyValue("--primary").trim() || "#534AB7";
  const primaryL = getComputedStyle(document.documentElement).getPropertyValue("--primary-l").trim() || "#EEEDFE";

  scoreChart = new Chart(ctx.getContext("2d"), {
    type: "line",
    data: {
      labels: labels,
      datasets: [{
        label: "ColBERT MaxSim",
        data: dataPoints,
        borderColor: primary,
        backgroundColor: primaryL,
        borderWidth: 2, pointRadius: 2, fill: true, tension: 0.3,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: "candidate rank", font: { size: 10 } } },
        y: { min: 0, ticks: { font: { size: 10 } } }
      }
    }
  });
}

function renderBlendChart(alpha) {
  if (blendChart) blendChart.destroy();
  const ctx = document.getElementById("blendChart");
  if (!ctx) return;

  // Defensive: if alpha is undefined/null, assume balanced (0.5)
  alpha = (alpha !== undefined && alpha !== null) ? parseFloat(alpha) : 0.5;
  if (isNaN(alpha)) alpha = 0.5;

  const primary = getComputedStyle(document.documentElement).getPropertyValue("--primary").trim() || "#534AB7";
  const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#1D9E75";

  blendChart = new Chart(ctx.getContext("2d"), {
    type: "doughnut",
    data: {
      labels: ["Semantic (ColBERT)", "User interest"],
      datasets: [{
        data: [
          parseFloat((alpha * 100).toFixed(1)),
          parseFloat(((1 - alpha) * 100).toFixed(1))
        ],
        backgroundColor: [primary, accent],
        borderWidth: 0,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: "60%",
      plugins: { legend: { display: true, position: "bottom", labels: { font: { size: 11 }, boxWidth: 12 } } }
    }
  });
}

function renderRankChart(results) {
  if (rankChart) rankChart.destroy();
  const ctx = document.getElementById("rankChart");
  if (!ctx) return;

  if (!results || results.length === 0) return;

  const primary = getComputedStyle(document.documentElement).getPropertyValue("--primary").trim() || "#534AB7";
  const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#1D9E75";

  rankChart = new Chart(ctx.getContext("2d"), {
    type: "bar",
    data: {
      labels: results.map(r => "#" + r.rank),
      datasets: [
        {
          label: "Semantic",
          data: results.map(r => {
            const s = parseFloat(r.semantic_score);
            return isNaN(s) ? 0 : parseFloat(s.toFixed(4));
          }),
          backgroundColor: primary + "99",
          borderColor: primary, borderWidth: 1,
        },
        {
          label: "Hybrid",
          data: results.map(r => {
            const f = parseFloat(r.final_score);
            return isNaN(f) ? 0 : parseFloat(f.toFixed(4));
          }),
          backgroundColor: accent + "99",
          borderColor: accent, borderWidth: 1,
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: true, labels: { font: { size: 11 }, boxWidth: 12 } } },
      scales: {
        x: { ticks: { font: { size: 10 } } },
        y: { min: 0, max: 1, ticks: { font: { size: 10 }, stepSize: 0.2 } }
      }
    }
  });
}

function renderTokenGrid(tokenWeights) {
  const grid = document.getElementById("token-grid");
  if (!grid) return;
  if (!tokenWeights || tokenWeights.length === 0) {
    grid.innerHTML = "<span style='color:var(--muted);font-size:12px'>Run a search first</span>";
    return;
  }
  const max = tokenWeights[0].weight;
  const primary = getComputedStyle(document.documentElement).getPropertyValue("--primary").trim() || "#534AB7";
  const primaryL = getComputedStyle(document.documentElement).getPropertyValue("--primary-l").trim() || "#EEEDFE";
  grid.innerHTML = tokenWeights.map(t => {
    const norm = t.weight / (max || 1);
    const bg = norm > 0.6 ? primaryL : "var(--bg2)";
    const col = norm > 0.6 ? primary : "var(--muted)";
    const size = Math.round(11 + norm * 3);
    return `<span class="token" style="background:${bg};color:${col};font-size:${size}px">${t.token}</span>`;
  }).join("");
}

// Listen for searchComplete events from search.js
window.addEventListener("searchComplete", e => {
  const d = e.detail;
  if (!d) return;

  // Alpha blend chart — use alpha from response (not from results)
  renderBlendChart(d.alpha);

  // ColBERT score distribution — use stage1_all_scores for complete picture
  const allScores = d.stage1_all_scores || (d.results ? d.results.map(r => r.colbert_score) : []);
  const stage1Count = d.stage1_count || allScores.length || 100;
  renderScoreChart(allScores, stage1Count);

  // Rank comparison chart
  renderRankChart(d.results);

  // Token weights
  renderTokenGrid(d.token_weights || []);

  // Refresh analytics
  loadAnalytics();
});

// Also support a custom event for manually passing full score distribution
window.addEventListener("colbertScoresComplete", e => {
  const { scores, stage1_count, alpha } = e.detail;
  if (alpha !== undefined) renderBlendChart(alpha);
  const allScores = scores || [];
  const count = stage1_count || allScores.length || 100;
  renderScoreChart(allScores, count);
});

loadAnalytics();
setInterval(loadAnalytics, 10000);