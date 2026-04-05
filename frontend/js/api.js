const API_BASE = "";  // same origin — FastAPI serves the frontend

async function apiSearch(query, userId, method, topK) {
  const res = await fetch(`${API_BASE}/api/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query, user_id: userId, method, top_k: topK
    }),
  });
  if (!res.ok) throw new Error(`Search failed: ${res.status}`);
  return res.json();
}

async function apiClick(userId, docId, docIdx, query) {
  const res = await fetch(`${API_BASE}/api/click`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: userId, doc_id: docId, doc_idx: docIdx, query
    }),
  });
  if (!res.ok) throw new Error(`Click log failed: ${res.status}`);
  return res.json();
}

async function apiUserProfile(userId) {
  const res = await fetch(`${API_BASE}/api/user/${encodeURIComponent(userId)}/profile`);
  return res.json();
}

async function apiUserHistory(userId) {
  const res = await fetch(`${API_BASE}/api/user/${encodeURIComponent(userId)}/history`);
  return res.json();
}

async function apiAnalytics() {
  const res = await fetch(`${API_BASE}/api/analytics`);
  return res.json();
}
