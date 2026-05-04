let currentMethod = "hybrid";
let lastQuery = "";
let lastResponse = null;
const clickedDocs = new Set();

// Method toggle
document.querySelectorAll(".method-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".method-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentMethod = btn.dataset.method;
  });
});

// User change — refresh profile
document.getElementById("user-select").addEventListener("change", async function () {
  clickedDocs.clear();
  await updateProfileCard(this.value);
});

async function doSearch() {
  const query  = document.getElementById("query-input").value.trim();
  const userId = document.getElementById("user-select").value;
  const topK   = parseInt(document.getElementById("topk-select").value);
  if (!query) return;

  lastQuery = query;
  document.getElementById("status").textContent = "Searching...";
  document.getElementById("results").innerHTML = "";

  try {
    const data = await apiSearch(query, userId, currentMethod, topK);
    lastResponse = data;
    renderResults(data, userId);
    updateSidebar(data);
    // dispatch event for dashboard if open (after lastResponse is set)
  } catch (e) {
    document.getElementById("status").textContent =
      "Error: make sure the API is running. " + e.message;
  }
}

function hostnameFromUrl(url) {
  try {
    return new URL(url).hostname;
  } catch {
    return "";
  }
}

function renderResults(data, userId) {
  const { results, latency_ms, method, alpha } = data;
  document.getElementById("status").textContent =
    `${results.length} results · ${method} · ${latency_ms}ms · α=${alpha.toFixed(2)}`;

  const container = document.getElementById("results");
  container.innerHTML = "";

  results.forEach(r => {
    const wasClicked = clickedDocs.has(r.doc_id);
    const card = document.createElement("div");
    card.className = "result-card" + (wasClicked ? " clicked" : "");

    const colBadge = `<span class="badge badge-colbert">${method === "bm25" ? "BM25" : "ColBERT"}</span>`;
    const persBadge = r.personalized
      ? `<span class="badge badge-personalized">personalized</span>` : "";

    const host = r.url ? hostnameFromUrl(r.url) : "";
    const linkHtml = host
      ? `<a href="${r.url}" target="_blank" rel="noopener" style="color:#534AB7">${host}</a>`
      : "";

    card.innerHTML = `
      <div class="result-top">
        <span class="result-rank">#${r.rank}</span>
        <div style="flex:1">
          <div class="result-title">${r.title || "(no title)"}</div>
        </div>
        <div class="badges">${colBadge}${persBadge}</div>
      </div>
      <div class="result-snippet">${r.snippet}</div>
      <div class="score-bars">
        <span class="score-bar-label">Semantic</span>
        <div class="score-bar-bg"><div class="score-bar-fill" style="width:${(r.semantic_score*100).toFixed(0)}%;background:#7F77DD"></div></div>
        <span class="score-bar-label" style="min-width:30px;text-align:right">${r.semantic_score.toFixed(2)}</span>
      </div>
      <div class="score-bars">
        <span class="score-bar-label">Hybrid</span>
        <div class="score-bar-bg"><div class="score-bar-fill" style="width:${(r.final_score*100).toFixed(0)}%;background:#1D9E75"></div></div>
        <span class="score-bar-label" style="min-width:30px;text-align:right">${r.final_score.toFixed(2)}</span>
      </div>
      <div class="result-meta">
        <span>user score: ${r.user_score.toFixed(3)}</span>
        ${linkHtml}
      </div>`;

    card.addEventListener("click", () => handleClick(r, userId, card));
    container.appendChild(card);
  });
}

async function handleClick(result, userId, card) {
  if (clickedDocs.has(result.doc_id)) return;
  clickedDocs.add(result.doc_id);
  card.classList.add("clicked");

  const docIdx = typeof result.doc_idx === "number" ? result.doc_idx : -1;

  try {
    const resp = await apiClick(userId, result.doc_id, docIdx, lastQuery);
    await updateProfileCard(userId);
    document.getElementById("profile-info").textContent =
      `${resp.click_count} click${resp.click_count !== 1 ? "s" : ""} logged`;
  } catch (e) {
    console.error("Click error:", e);
  }
}

function updateSidebar(data) {
  document.getElementById("pipeline-card").style.display = "block";
  document.getElementById("s1-count").textContent = data.stage1_count + " docs";
  document.getElementById("s2-count").textContent = data.stage2_count + " docs";
  document.getElementById("latency-val").textContent = data.latency_ms + "ms";
  // Dispatch event so charts.js can render alpha blend + ColBERT distribution
  window.dispatchEvent(new CustomEvent("searchComplete", { detail: data }));
}

async function updateProfileCard(userId) {
  try {
    const p = await apiUserProfile(userId);
    document.getElementById("profile-info").textContent =
      p.has_profile
        ? `${p.click_count} click${p.click_count !== 1 ? "s" : ""} · profile active`
        : "No clicks yet";
  } catch (e) {}
}

// Enter key
document.getElementById("query-input").addEventListener("keydown", e => {
  if (e.key === "Enter") doSearch();
});

// Init
updateProfileCard(document.getElementById("user-select").value);
