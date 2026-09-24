/**
 * app.js — Cross-Camera Re-ID Dashboard (Phase 6)
 *
 * Three coordinated regions (§8):
 *   1. Camera wall  — GET /cameras once → <img src="/stream/{id}"> per tile
 *   2. Shopper ledger — GET /shoppers on load + /ws live updates
 *   3. Analytics footer — GET /analytics/store polled every 4s
 *
 * Phase 6 additions:
 *   - Per-shopper detail modal (click any card)
 *   - Journey path visualization (CSS zone bubbles)
 *   - CSV export via /shoppers/export.csv
 *   - Match score display on cross-camera events
 *   - Ledger filter (All / Active / Inactive)
 *   - Cross-camera match counter in footer
 *
 * Phase 7: reads API key from sessionStorage (set by login.html).
 */

"use strict";

/* ── Config ──────────────────────────────────────────────────────────── */
const API_BASE      = window.location.origin;
const WS_URL        = `${API_BASE.replace(/^http/, "ws")}/ws`;
const POLL_INTERVAL = 4000;
const RECONNECT_WAIT= 3000;

/* ── Auth (Phase 7) ─────────────────────────────────────────────────── */
function getApiKey() {
  return sessionStorage.getItem("reid_api_key") || "";
}

function authHeaders() {
  const key = getApiKey();
  return key ? { "Authorization": `Bearer ${key}` } : {};
}

async function apiFetch(path, opts = {}) {
  opts.headers = Object.assign({}, opts.headers || {}, authHeaders());
  const res = await fetch(`${API_BASE}${path}`, opts);
  if (res.status === 401) {
    // Redirect to login if key rejected
    if (window.location.pathname !== "/login") {
      sessionStorage.removeItem("reid_api_key");
      window.location.href = "/login";
    }
  }
  return res;
}

/* ── State ───────────────────────────────────────────────────────────── */
/** @type {Map<string, object>} global_id → summary */
const shopperMap = new Map();
let ws = null;
let analyticsTimer = null;
let activeFilter = "all";
let crossCameraMatchCount = 0;
let currentDetailId = null;

/* ── DOM refs ─────────────────────────────────────────────────────────── */
const cameraWall        = document.getElementById("camera-wall");
const shopperLedger     = document.getElementById("shopper-ledger");
const wsLabel           = document.getElementById("ws-label");
const wsDot             = document.getElementById("ws-dot");
const wsStatus          = document.getElementById("ws-status");
const camCountBadge     = document.getElementById("cam-count-badge");
const ledgerCountBadge  = document.getElementById("ledger-count-badge");
const clockEl           = document.getElementById("clock-display");
const exportBtn         = document.getElementById("export-btn");
const analyticsUpdatedEl= document.getElementById("analytics-updated");

const statUnique  = document.getElementById("stat-unique-val");
const statActive  = document.getElementById("stat-active-val");
const statDwell   = document.getElementById("stat-dwell-val");
const statMatch   = document.getElementById("stat-match-val");
const zoneBars    = document.getElementById("zone-bars");

// Modal
const modal          = document.getElementById("detail-modal");
const modalTitle     = document.getElementById("modal-title");
const modalStatus    = document.getElementById("modal-status");
const modalJourney   = document.getElementById("modal-journey");
const modalFirstSeen = document.getElementById("modal-first-seen");
const modalLastSeen  = document.getElementById("modal-last-seen");
const modalTotalDwell= document.getElementById("modal-total-dwell");
const modalVisitCount= document.getElementById("modal-visit-count");
const modalVisitTbody= document.getElementById("modal-visit-tbody");
const modalCloseBtn  = document.getElementById("modal-close");

/* ═══════════════════════════════════════════════════════════════════════
   CLOCK
═══════════════════════════════════════════════════════════════════════ */
function updateClock() {
  clockEl.textContent = new Date().toLocaleTimeString("en-GB", { hour12: false });
}
updateClock();
setInterval(updateClock, 1000);

/* ═══════════════════════════════════════════════════════════════════════
   WS STATUS
═══════════════════════════════════════════════════════════════════════ */
function setWsStatus(state) {
  wsStatus.className = `ws-indicator ${state}`;
  wsLabel.textContent = state === "live" ? "live" : "connecting…";
}

/* ═══════════════════════════════════════════════════════════════════════
   CAMERA WALL
═══════════════════════════════════════════════════════════════════════ */
async function loadCameraWall() {
  let cameras = [];
  try {
    const res = await apiFetch("/cameras");
    cameras = await res.json();
  } catch (e) {
    console.warn("Could not load cameras:", e);
  }

  cameraWall.innerHTML = "";
  camCountBadge.textContent = `${cameras.length} feed${cameras.length !== 1 ? "s" : ""}`;

  if (cameras.length === 0) {
    cameraWall.innerHTML = '<div class="cam-placeholder">No camera feeds configured.</div>';
    return;
  }

  const cols = cameras.length <= 4 ? 2 : 3;
  cameraWall.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;

  cameras.forEach(cam => {
    const tile = document.createElement("div");
    tile.className = "cam-tile";
    tile.setAttribute("role", "listitem");
    tile.id = `cam-tile-${cam.camera_id}`;

    // Build stream URL with auth key if needed
    const key = getApiKey();
    const streamUrl = key
      ? `${API_BASE}/stream/${cam.camera_id}?key=${encodeURIComponent(key)}`
      : `${API_BASE}/stream/${cam.camera_id}`;

    const img = document.createElement("img");
    img.src = streamUrl;
    img.alt = `Live feed: ${cam.name}`;
    img.onerror = () => setTimeout(() => { img.src = streamUrl + "&t=" + Date.now(); }, 2000);

    const label = document.createElement("div");
    label.className = "cam-tile-label";
    label.innerHTML = `
      <span class="cam-tile-name mono">${cam.zone}</span>
      <span class="cam-tile-live">LIVE</span>
    `;
    tile.appendChild(img);
    tile.appendChild(label);
    cameraWall.appendChild(tile);
  });
}

/* ═══════════════════════════════════════════════════════════════════════
   HELPERS
═══════════════════════════════════════════════════════════════════════ */
function fmtDwell(s) {
  if (s == null || isNaN(s)) return "—";
  if (s < 60) return `${Math.round(s)}s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

function fmtTime(ts) {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleTimeString("en-GB", { hour12: false });
}

function fmtDateTime(ts) {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString("en-GB", { hour12: false });
}

/* ═══════════════════════════════════════════════════════════════════════
   SHOPPER LEDGER
═══════════════════════════════════════════════════════════════════════ */
function buildShopperCard(summary, isNew = false, flashMatch = false) {
  const card = document.createElement("div");
  card.className = "shopper-card" + (summary.active ? " active" : "");
  card.setAttribute("role", "listitem");
  card.id = `shopper-card-${summary.global_id}`;
  if (isNew) card.classList.add("new-identity");
  if (flashMatch) card.classList.add("match-flash");

  const zone  = summary.current_zone || "—";
  const dwell = fmtDwell(summary.total_dwell_seconds);
  const camId = summary.last_camera_id || "—";

  // Build compact path (last 4 zones)
  const path = summary.path || [];
  const recentPath = path.length > 4
    ? "…" + path.slice(-4).join(" → ")
    : path.join(" → ") || "—";

  const scoreHtml = summary.last_match_score > 0
    ? `<span class="score-badge">${(summary.last_match_score * 100).toFixed(0)}%</span>`
    : "";

  card.innerHTML = `
    <div class="shopper-active-dot" aria-hidden="true"></div>
    <div class="shopper-info">
      <div class="shopper-id">${summary.global_id} ${scoreHtml}</div>
      <div class="shopper-meta">
        <span class="zone-tag">${zone}</span>
        <span class="cam-tag mono">${camId}</span>
      </div>
      <div class="shopper-path" title="${path.join(' → ')}">${recentPath}</div>
    </div>
    <div class="shopper-dwell">
      <div class="dwell-num">${dwell}</div>
    </div>
  `;

  card.addEventListener("click", () => openDetailModal(summary.global_id));
  return card;
}

function getFilteredShoppers() {
  const all = [...shopperMap.values()];
  if (activeFilter === "active") return all.filter(s => s.active);
  if (activeFilter === "inactive") return all.filter(s => !s.active);
  return all;
}

function renderLedger() {
  const placeholder = shopperLedger.querySelector(".ledger-placeholder");
  if (placeholder) placeholder.remove();

  const sorted = getFilteredShoppers().sort((a, b) => {
    if (a.active !== b.active) return a.active ? -1 : 1;
    return (b.last_seen || 0) - (a.last_seen || 0);
  });

  ledgerCountBadge.textContent = `${shopperMap.size} ID${shopperMap.size !== 1 ? "s" : ""}`;

  // Update existing cards in-place
  sorted.forEach(s => {
    const existing = document.getElementById(`shopper-card-${s.global_id}`);
    if (existing) {
      const zoneEl  = existing.querySelector(".zone-tag");
      const camEl   = existing.querySelector(".cam-tag");
      const pathEl  = existing.querySelector(".shopper-path");
      const dwellEl = existing.querySelector(".dwell-num");
      const idEl    = existing.querySelector(".shopper-id");
      const path = s.path || [];
      const recentPath = path.length > 4
        ? "…" + path.slice(-4).join(" → ")
        : path.join(" → ") || "—";
      if (zoneEl)  zoneEl.textContent  = s.current_zone || "—";
      if (camEl)   camEl.textContent   = s.last_camera_id || "—";
      if (pathEl)  { pathEl.textContent = recentPath; pathEl.title = path.join(" → "); }
      if (dwellEl) dwellEl.textContent = fmtDwell(s.total_dwell_seconds);
      if (idEl) {
        const scoreHtml = s.last_match_score > 0
          ? ` <span class="score-badge">${(s.last_match_score * 100).toFixed(0)}%</span>` : "";
        idEl.innerHTML = s.global_id + scoreHtml;
      }
      existing.classList.toggle("active", !!s.active);
    }
  });

  // Insert new cards (prepend newest first)
  [...sorted].reverse().forEach(s => {
    if (!document.getElementById(`shopper-card-${s.global_id}`)) {
      const card = buildShopperCard(s, true, false);
      shopperLedger.prepend(card);
    }
  });

  // Remove cards that no longer match the filter
  shopperLedger.querySelectorAll(".shopper-card").forEach(el => {
    const id = el.id.replace("shopper-card-", "");
    const shopper = shopperMap.get(id);
    if (!shopper) { el.remove(); return; }
    const visible = sorted.some(s => s.global_id === id);
    if (!visible) el.remove();
  });
}

function upsertShopper(summary, flashMatch = false, matchScore = 0) {
  summary.last_match_score = matchScore;
  shopperMap.set(summary.global_id, summary);
  if (flashMatch) {
    crossCameraMatchCount++;
    if (statMatch) statMatch.textContent = crossCameraMatchCount;
    const card = document.getElementById(`shopper-card-${summary.global_id}`);
    if (card) {
      card.classList.remove("match-flash");
      void card.offsetWidth;
      card.classList.add("match-flash");
    }
  }
  renderLedger();
  // Refresh modal if it's showing this shopper
  if (currentDetailId === summary.global_id) {
    loadDetailModal(summary.global_id);
  }
}

/* ═══════════════════════════════════════════════════════════════════════
   DETAIL MODAL (Phase 6)
═══════════════════════════════════════════════════════════════════════ */
async function openDetailModal(globalId) {
  currentDetailId = globalId;
  modal.removeAttribute("hidden");
  await loadDetailModal(globalId);
}

async function loadDetailModal(globalId) {
  // Fetch full detail including zone_visits
  let detail;
  try {
    const res = await apiFetch(`/shoppers/${encodeURIComponent(globalId)}`);
    if (!res.ok) return;
    detail = await res.json();
  } catch (e) {
    console.warn("Could not fetch shopper detail:", e);
    return;
  }

  modalTitle.textContent = detail.global_id;
  modalStatus.textContent = detail.active ? "● Active" : "○ Inactive";
  modalStatus.style.color = detail.active ? "var(--green)" : "var(--text-muted)";

  modalFirstSeen.textContent = fmtDateTime(detail.first_seen);
  modalLastSeen.textContent  = fmtDateTime(detail.last_seen);
  modalTotalDwell.textContent= fmtDwell(detail.total_dwell_seconds);
  modalVisitCount.textContent= detail.zone_visit_count ?? "—";

  // Journey path visualization
  const path = detail.path || [];
  const currentZone = detail.current_zone;
  const deduped = path.filter((z, i) => i === 0 || z !== path[i - 1]);

  modalJourney.innerHTML = "";
  if (deduped.length === 0) {
    modalJourney.innerHTML = '<span style="color:var(--text-muted)">No zones recorded yet.</span>';
  } else {
    deduped.forEach((zone, i) => {
      const step = document.createElement("div");
      step.className = "journey-step";
      const isCurrent = (i === deduped.length - 1) && detail.active;
      step.innerHTML = `<span class="journey-zone${isCurrent ? " current" : ""}">${zone}</span>`;
      modalJourney.appendChild(step);
      if (i < deduped.length - 1) {
        const arrow = document.createElement("div");
        arrow.className = "journey-step";
        arrow.innerHTML = '<span class="journey-arrow">→</span>';
        modalJourney.appendChild(arrow);
      }
    });
  }

  // Visit table
  const visits = detail.zone_visits || [];
  modalVisitTbody.innerHTML = "";
  visits.forEach((v, i) => {
    const tr = document.createElement("tr");
    const isOpen = !v.exit_time;
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td class="td-zone">${v.zone}</td>
      <td>${v.camera_id}</td>
      <td>${fmtTime(v.enter_time)}</td>
      <td class="${isOpen ? "td-open" : ""}">${isOpen ? "open" : fmtTime(v.exit_time)}</td>
      <td>${fmtDwell(v.dwell_seconds)}</td>
    `;
    modalVisitTbody.appendChild(tr);
  });
}

function closeModal() {
  modal.setAttribute("hidden", "");
  currentDetailId = null;
}

modalCloseBtn.addEventListener("click", closeModal);
modal.addEventListener("click", e => { if (e.target === modal) closeModal(); });
document.addEventListener("keydown", e => { if (e.key === "Escape") closeModal(); });

/* ═══════════════════════════════════════════════════════════════════════
   ANALYTICS FOOTER
═══════════════════════════════════════════════════════════════════════ */
async function fetchAnalytics() {
  try {
    const res  = await apiFetch("/analytics/store");
    const data = await res.json();
    renderAnalytics(data);
    if (analyticsUpdatedEl) {
      analyticsUpdatedEl.textContent = "Updated " + new Date().toLocaleTimeString("en-GB", { hour12: false });
    }
  } catch (_) {}
}

function renderAnalytics(data) {
  if (statUnique) statUnique.textContent = data.unique_shoppers ?? "—";
  if (statActive) statActive.textContent = data.currently_active ?? "—";
  const d = data.avg_dwell_seconds;
  if (statDwell)  statDwell.textContent = typeof d === "number" ? fmtDwell(d) : "—";
  if (statMatch)  statMatch.textContent = crossCameraMatchCount;

  const zp = data.zone_popularity || {};
  const entries = Object.entries(zp).sort((a, b) => b[1].visit_count - a[1].visit_count);
  const maxV = entries.length ? entries[0][1].visit_count : 1;

  zoneBars.innerHTML = "";
  entries.forEach(([zone, stats]) => {
    const pct = Math.max(4, Math.round((stats.visit_count / maxV) * 100));
    const row = document.createElement("div");
    row.className = "zone-bar-row";
    row.setAttribute("role", "listitem");
    row.innerHTML = `
      <span class="zone-bar-name" title="${zone}">${zone}</span>
      <div class="zone-bar-track" aria-hidden="true">
        <div class="zone-bar-fill" style="width:${pct}%"></div>
      </div>
      <span class="zone-bar-count">${stats.visit_count}</span>
      <span class="zone-bar-dwell">${fmtDwell(stats.avg_dwell_seconds)}</span>
    `;
    zoneBars.appendChild(row);
  });
}

/* ═══════════════════════════════════════════════════════════════════════
   WEBSOCKET
═══════════════════════════════════════════════════════════════════════ */
function connectWebSocket() {
  if (ws) { try { ws.close(); } catch (_) {} }
  const key = getApiKey();
  const url = key ? `${WS_URL}?key=${encodeURIComponent(key)}` : WS_URL;
  ws = new WebSocket(url);
  setWsStatus("reconnecting");

  ws.onopen = () => {
    setWsStatus("live");
    console.info("[ws] Connected.");
  };

  ws.onmessage = event => {
    let msg;
    try { msg = JSON.parse(event.data); } catch (_) { return; }
    switch (msg.type) {
      case "initial_state":
        (msg.shoppers || []).forEach(s => shopperMap.set(s.global_id, s));
        renderLedger();
        if (msg.analytics) renderAnalytics(msg.analytics);
        break;
      case "identity_update":
        if (msg.summary && msg.summary.global_id) {
          upsertShopper(msg.summary, !!msg.cross_camera_match, msg.match_score || 0);
        }
        break;
      case "ping":
        break;
    }
  };

  ws.onerror = err => console.warn("[ws] Error:", err);
  ws.onclose = () => {
    setWsStatus("reconnecting");
    setTimeout(connectWebSocket, RECONNECT_WAIT);
  };
}

/* ═══════════════════════════════════════════════════════════════════════
   RE-ID REGISTER  (Phase 9)
═══════════════════════════════════════════════════════════════════════ */
const registerTbody      = document.getElementById("register-tbody");
const registerBadge      = document.getElementById("register-exemplar-badge");
const registerResetBtn   = document.getElementById("register-reset-btn");
let registerTimer        = null;

/** Build a small camera breakdown badge group */
function buildCamChips(perCamera) {
  if (!perCamera || Object.keys(perCamera).length === 0) return '<span class="cam-chip empty">—</span>';
  return Object.entries(perCamera)
    .sort((a, b) => b[1] - a[1])
    .map(([cam, cnt]) =>
      `<span class="cam-chip" title="${cnt} exemplar${cnt !== 1 ? "s" : ""} from ${cam}">${cam}&nbsp;<b>${cnt}</b></span>`
    )
    .join("");
}

/** Build a compact gallery-size bar */
function buildGalleryBar(total, cap) {
  const maxCap = cap || 32;
  const pct    = Math.min(100, Math.round((total / maxCap) * 100));
  return `
    <div class="gallery-bar-wrap" title="${total} exemplar${total !== 1 ? "s" : ""}">
      <div class="gallery-bar-track">
        <div class="gallery-bar-fill" style="width:${pct}%"></div>
      </div>
      <span class="gallery-bar-label">${total}</span>
    </div>`;
}

async function loadRegister() {
  let data;
  try {
    const res = await apiFetch("/registry/exemplars");
    if (!res.ok) return;
    data = await res.json();
  } catch (_) { return; }

  if (!Array.isArray(data)) return;

  // Update total exemplars badge
  const totalEx = data.reduce((s, d) => s + (d.total_exemplars || 0), 0);
  if (registerBadge) {
    registerBadge.textContent = `${totalEx} exemplar${totalEx !== 1 ? "s" : ""}`;
  }

  if (!registerTbody) return;

  // Remove placeholder row once we have data
  const placeholder = document.getElementById("register-placeholder-row");
  if (data.length > 0 && placeholder) placeholder.remove();
  else if (data.length === 0) {
    registerTbody.innerHTML = `
      <tr id="register-placeholder-row">
        <td colspan="6" class="register-placeholder-cell">
          <span>No identities registered yet — start a camera feed to begin.</span>
        </td>
      </tr>`;
    return;
  }

  // Upsert rows (update existing, append new, remove stale)
  const seen = new Set();
  data.forEach(entry => {
    const id    = entry.global_id;
    seen.add(id);
    const rowId = `reg-row-${id}`;
    let row     = document.getElementById(rowId);

    const statusHtml = entry.active
      ? '<span class="reg-status active">● Active</span>'
      : '<span class="reg-status inactive">○ Inactive</span>';
    const camHtml    = buildCamChips(entry.per_camera);
    const barHtml    = buildGalleryBar(entry.total_exemplars);

    if (row) {
      // Update in-place
      row.querySelector(".reg-gallery").innerHTML   = barHtml;
      row.querySelector(".reg-cam").innerHTML       = camHtml;
      row.querySelector(".reg-status-td").innerHTML = statusHtml;
      row.querySelector(".reg-last").textContent    = fmtDateTime(entry.last_seen);
    } else {
      row = document.createElement("tr");
      row.id        = rowId;
      row.className = "register-row" + (entry.active ? " reg-active" : "");
      row.innerHTML = `
        <td class="reg-id mono">${id}</td>
        <td class="reg-first mono">${fmtDateTime(entry.first_seen)}</td>
        <td class="reg-last mono">${fmtDateTime(entry.last_seen)}</td>
        <td class="reg-gallery">${barHtml}</td>
        <td class="reg-cam">${camHtml}</td>
        <td class="reg-status-td">${statusHtml}</td>
      `;
      registerTbody.appendChild(row);
    }
    // Sync active class
    row.classList.toggle("reg-active", !!entry.active);
  });

  // Remove rows for identities no longer in the registry
  registerTbody.querySelectorAll(".register-row").forEach(row => {
    const id = row.id.replace("reg-row-", "");
    if (!seen.has(id)) row.remove();
  });
}

async function resetRegistry() {
  const confirmed = window.confirm(
    "⚠️  Reset Registry?\n\n" +
    "This will permanently delete ALL registered identities and gallery exemplars.\n" +
    "The identity counter will restart from GSI-0001.\n\n" +
    "This action cannot be undone."
  );
  if (!confirmed) return;

  if (registerResetBtn) {
    registerResetBtn.disabled    = true;
    registerResetBtn.textContent = "Resetting…";
  }
  try {
    const res = await apiFetch("/registry/reset", { method: "POST" });
    if (!res.ok) {
      alert("Reset failed: " + res.status);
      return;
    }
    // Clear local ledger state
    shopperMap.clear();
    crossCameraMatchCount = 0;
    if (statMatch) statMatch.textContent = "0";
    renderLedger();
    // Clear register table
    if (registerTbody) {
      registerTbody.innerHTML = `
        <tr id="register-placeholder-row">
          <td colspan="6" class="register-placeholder-cell">
            <span>Registry cleared — waiting for new detections.</span>
          </td>
        </tr>`;
    }
    if (registerBadge) registerBadge.textContent = "0 exemplars";
    console.info("[register] Registry reset.");
  } catch (e) {
    alert("Reset error: " + e.message);
  } finally {
    if (registerResetBtn) {
      registerResetBtn.disabled = false;
      registerResetBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.5"/></svg>
        Reset Registry`;
    }
  }
}

if (registerResetBtn) {
  registerResetBtn.addEventListener("click", resetRegistry);
}

/* ═══════════════════════════════════════════════════════════════════════
   FILTER BUTTONS
═══════════════════════════════════════════════════════════════════════ */
document.querySelectorAll(".filter-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    activeFilter = btn.dataset.filter;
    renderLedger();
  });
});

/* ═══════════════════════════════════════════════════════════════════════
   CSV EXPORT (Phase 6)
═══════════════════════════════════════════════════════════════════════ */
if (exportBtn) {
  exportBtn.addEventListener("click", async () => {
    exportBtn.textContent = "Exporting…";
    exportBtn.disabled = true;
    try {
      const res = await apiFetch("/shoppers/export.csv");
      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement("a");
      a.href = url;
      a.download = `shoppers_${new Date().toISOString().slice(0,19).replace(/:/g,"-")}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error("Export failed:", e);
    } finally {
      exportBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> Export CSV`;
      exportBtn.disabled = false;
    }
  });
}

/* ═══════════════════════════════════════════════════════════════════════
   BOOTSTRAP
═══════════════════════════════════════════════════════════════════════ */
async function loadInitialShoppers() {
  try {
    const res = await apiFetch("/shoppers");
    const list = await res.json();
    list.forEach(s => shopperMap.set(s.global_id, s));
    renderLedger();
  } catch (e) {
    console.warn("Could not preload shoppers:", e);
  }
}

async function init() {
  await loadCameraWall();
  await loadInitialShoppers();
  connectWebSocket();
  fetchAnalytics();
  analyticsTimer = setInterval(fetchAnalytics, POLL_INTERVAL);
  // Re-ID Register: initial load + poll every 5 s
  await loadRegister();
  registerTimer = setInterval(loadRegister, 5000);
}

init();

