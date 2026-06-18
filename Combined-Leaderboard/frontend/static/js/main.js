/* Visual Cognition Leaderboard — three-task / two-section frontend */
(function () {
  "use strict";

  var state = {
    sections: null,
    taskInfo: {},        // task_id -> info
    vc: [],              // visual-cognition leaderboard rows
    spatial: [],         // spatial leaderboard rows
    stats: {},
    selectedCaps: [],    // model names selected for the radar
  };

  /* ----------------------------------------------------------------- utils */
  function $(id) { return document.getElementById(id); }
  function el(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function fmtPct(v) { return v == null ? "—" : (v * 100).toFixed(1) + "%"; }
  function fmtVci(v) { return v == null ? "—" : (v * 100).toFixed(1); }
  function fmtDelta(v) {
    if (v == null) return "—";
    return (v >= 0 ? "+" : "") + (v * 100).toFixed(1);
  }
  function prettyLabel(s) {
    return String(s || "").replace(/_/g, " ").replace(/\b\w/g, function (c) { return c.toUpperCase(); });
  }
  function getCss(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
  }
  function hexA(hex, a) {
    hex = (hex || "").replace("#", "");
    if (hex.length === 3) hex = hex.split("").map(function (c) { return c + c; }).join("");
    var n = parseInt(hex || "888888", 16);
    return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + a + ")";
  }
  function modelType(meta) {
    if (!meta) return "—";
    return esc(meta.type || meta.org || meta.family || "—");
  }
  async function getJSON(url) {
    var r = await fetch(url);
    if (!r.ok) throw new Error("HTTP " + r.status);
    return r.json();
  }

  /* ------------------------------------------------------------------ theme */
  function initTheme() {
    var saved = localStorage.getItem("vci-theme");
    if (saved) document.documentElement.setAttribute("data-theme", saved);
    var btn = $("theme_toggle");
    if (btn) btn.addEventListener("click", function () {
      var cur = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", cur);
      localStorage.setItem("vci-theme", cur);
      renderRadar();
      renderCotChart();
    });
  }

  /* ------------------------------------------------------------------- tabs */
  function initTabs() {
    function show(tab) {
      document.querySelectorAll(".tab-btn").forEach(function (b) {
        b.classList.toggle("is-active", b.getAttribute("data-tab") === tab);
      });
      document.querySelectorAll(".tab-panel").forEach(function (p) {
        p.classList.toggle("is-active", p.id === "panel-" + tab);
      });
    }
    document.querySelectorAll("[data-tab]").forEach(function (b) {
      b.addEventListener("click", function (e) {
        e.preventDefault();
        show(b.getAttribute("data-tab"));
      });
    });
  }

  /* ------------------------------------------------------------------ health */
  function checkHealth() {
    fetch("/api/health").then(function (r) {
      var ok = r.ok;
      var dot = $("status_dot"), txt = $("status_text");
      if (dot) dot.classList.toggle("ok", ok);
      if (txt) txt.textContent = ok ? "Online" : "Offline";
    }).catch(function () {
      var dot = $("status_dot"), txt = $("status_text");
      if (dot) dot.classList.remove("ok");
      if (txt) txt.textContent = "Offline";
    });
  }

  /* --------------------------------------------------------------- loaders */
  async function loadSections() {
    state.sections = await getJSON("/api/sections");
    renderTaxonomy();
    renderSubmitCards();
  }

  async function loadTaskInfo() {
    var ids = ["do_you_see_me", "minds_eye", "spatial"];
    await Promise.all(ids.map(async function (id) {
      try { state.taskInfo[id] = await getJSON("/api/tasks/" + id + "/info"); }
      catch (e) { state.taskInfo[id] = {}; }
    }));
    renderVcStats();
    renderSpatialStats();
    renderManifest();
  }

  async function loadVisualCognition() {
    var data = await getJSON("/api/leaderboard/visual-cognition");
    state.vc = data.leaderboard || [];
    renderVcTable();
    renderCapPicker();
    renderRadar();
  }

  async function loadSpatial() {
    var data = await getJSON("/api/leaderboard/spatial");
    state.spatial = data.leaderboard || [];
    renderSpatialTable();
    renderCotChart();
  }

  async function loadStats() {
    try { state.stats = await getJSON("/api/statistics/overview"); }
    catch (e) { state.stats = {}; }
    renderVcStats();
    renderSpatialStats();
  }

  /* ------------------------------------------------------------------ stats */
  function renderVcStats() {
    var s = state.stats || {};
    if ($("stat_vc_models")) $("stat_vc_models").textContent = s.visual_cognition_models != null ? s.visual_cognition_models : "0";
    if ($("stat_best_vci")) $("stat_best_vci").textContent = s.best_vci != null ? fmtVci(s.best_vci) : "—";
    if ($("stat_dysm_samples")) $("stat_dysm_samples").textContent = (state.taskInfo.do_you_see_me || {}).total_samples || "—";
    if ($("stat_me_samples")) $("stat_me_samples").textContent = (state.taskInfo.minds_eye || {}).total_samples || "—";
  }
  function renderSpatialStats() {
    var s = state.stats || {};
    var sp = state.taskInfo.spatial || {};
    if ($("stat_sp_models")) $("stat_sp_models").textContent = s.spatial_models != null ? s.spatial_models : "0";
    if ($("stat_best_spatial")) $("stat_best_spatial").textContent = s.best_spatial_accuracy != null ? fmtPct(s.best_spatial_accuracy) : "—";
    if ($("stat_sp_datasets")) $("stat_sp_datasets").textContent = (sp.datasets || []).length || "13";
    if ($("stat_sp_diag")) $("stat_sp_diag").textContent = s.with_diagnostics != null ? s.with_diagnostics : "0";
  }

  /* -------------------------------------------------------- VC leaderboard */
  function statusChip(r) {
    if (r.complete) return '<span class="chip layer-perception">Complete</span>';
    if (r.has_perception) return '<span class="chip">Perception only</span>';
    if (r.has_imagery) return '<span class="chip layer-imagery">Imagery only</span>';
    return "—";
  }
  function rankBadge(rank) {
    var cls = rank <= 3 ? "rank-badge rank-" + rank : "rank-badge";
    return '<span class="' + cls + '">' + rank + "</span>";
  }
  function renderVcTable() {
    var body = $("vc_body");
    if (!body) return;
    if (!state.vc.length) {
      body.innerHTML = '<tr><td colspan="7" class="empty-row">No submissions yet. Be the first — see the Submit tab.</td></tr>';
      return;
    }
    body.innerHTML = "";
    state.vc.forEach(function (r) {
      var tr = el("tr", "clickable");
      tr.innerHTML =
        "<td>" + rankBadge(r.rank) + "</td>" +
        "<td><strong>" + esc(r.model_name) + "</strong></td>" +
        "<td>" + modelType(r.model_meta) + "</td>" +
        '<td class="num vci-val">' + fmtVci(r.vci) + "</td>" +
        '<td class="num">' + fmtPct(r.perception_accuracy) + "</td>" +
        '<td class="num">' + fmtPct(r.imagery_accuracy) + "</td>" +
        "<td>" + statusChip(r) + "</td>";
      tr.addEventListener("click", function () { openReport(r.model_name); });
      body.appendChild(tr);
    });
  }

  /* ---------------------------------------------------- Spatial leaderboard */
  function renderSpatialTable() {
    var body = $("spatial_body");
    if (!body) return;
    if (!state.spatial.length) {
      body.innerHTML = '<tr><td colspan="8" class="empty-row">No Task-3 submissions yet.</td></tr>';
      return;
    }
    body.innerHTML = "";
    state.spatial.forEach(function (r) {
      var d = r.diagnostics || {};
      var deltaCls = d.cot_delta == null ? "" : (d.cot_delta < 0 ? "neg" : "pos");
      var tr = el("tr", "clickable");
      tr.innerHTML =
        "<td>" + rankBadge(r.rank) + "</td>" +
        "<td><strong>" + esc(r.model_name) + "</strong></td>" +
        "<td>" + modelType(r.model_meta) + "</td>" +
        '<td class="num vci-val">' + fmtPct(r.accuracy) + "</td>" +
        '<td class="num">' + (r.total_samples || 0) + "</td>" +
        '<td class="num ' + deltaCls + '">' + fmtDelta(d.cot_delta) + "</td>" +
        '<td class="num">' + fmtPct(d.shortcut_score) + "</td>" +
        '<td class="num">' + fmtPct(d.hallucination_resistance) + "</td>";
      tr.addEventListener("click", function () { openReport(r.model_name); });
      body.appendChild(tr);
    });
  }

  /* ------------------------------------------------------------- manifest */
  function renderManifest() {
    var body = $("manifest_body");
    if (!body) return;
    var datasets = (state.taskInfo.spatial || {}).datasets || [];
    if (!datasets.length) {
      body.innerHTML = '<tr><td colspan="5" class="empty-row">Manifest not available.</td></tr>';
      return;
    }
    body.innerHTML = "";
    datasets.forEach(function (d) {
      var tr = el("tr");
      tr.innerHTML =
        "<td><strong>" + esc(d.name) + "</strong></td>" +
        "<td>" + esc(d.type) + "</td>" +
        '<td class="num">' + (d.approx_n ? "~" + d.approx_n : "—") + "</td>" +
        "<td>" + (d.tags || []).map(function (t) { return '<span class="chip">' + esc(t) + "</span>"; }).join(" ") + "</td>" +
        "<td>" + esc(d.license || "—") + "</td>";
      body.appendChild(tr);
    });
  }

  /* ------------------------------------------------------------- taxonomy */
  function renderTaxonomy() {
    var box = $("taxonomy_box");
    if (!box || !state.sections) return;
    var html = "";
    (state.sections.sections || []).forEach(function (sec) {
      html += "<h4>" + esc(sec.label) + "</h4><ul>";
      (sec.tasks || []).forEach(function (t) {
        html += "<li><strong>" + esc(t.label) + "</strong> — " + esc(t.description) +
          (t.supports_diagnostics ? ' <span class="chip">diagnostics</span>' : "") + "</li>";
      });
      html += "</ul>";
    });
    box.innerHTML = html;
  }

  /* ------------------------------------------------- capability radar (VC) */
  function capabilityData(row) {
    var caps = {};
    ["perception_groups", "imagery_groups"].forEach(function (k) {
      var g = row[k] || {};
      Object.keys(g).forEach(function (name) { caps[name] = g[name].accuracy; });
    });
    return caps;
  }
  function allCapabilityAxes() {
    var set = {};
    state.vc.forEach(function (r) {
      var c = capabilityData(r);
      Object.keys(c).forEach(function (k) { set[k] = true; });
    });
    return Object.keys(set).sort();
  }
  var PALETTE = ["#2563eb", "#db2777", "#16a34a", "#d97706", "#7c3aed"];

  function renderCapPicker() {
    var box = $("cap_model_picker");
    if (!box) return;
    box.innerHTML = "";
    if (!state.vc.length) return;
    if (!state.selectedCaps.length) {
      state.selectedCaps = state.vc.slice(0, 3).map(function (r) { return r.model_name; });
    }
    state.vc.forEach(function (r) {
      var on = state.selectedCaps.indexOf(r.model_name) >= 0;
      var chip = el("button", "pick-chip" + (on ? " on" : ""), esc(r.model_name));
      chip.addEventListener("click", function () {
        var i = state.selectedCaps.indexOf(r.model_name);
        if (i >= 0) state.selectedCaps.splice(i, 1);
        else { if (state.selectedCaps.length >= 3) state.selectedCaps.shift(); state.selectedCaps.push(r.model_name); }
        renderCapPicker(); renderRadar();
      });
      box.appendChild(chip);
    });
  }

  function renderRadar() {
    var cv = $("radar_canvas");
    if (!cv) return;
    var ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
    var axes = allCapabilityAxes();
    var legend = $("radar_legend");
    if (legend) legend.innerHTML = "";
    if (axes.length < 3) {
      ctx.fillStyle = getCss("--text-muted");
      ctx.font = "14px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Submit Visual Cognition tasks to see capability profiles.", cv.width / 2, cv.height / 2);
      return;
    }
    var cx = cv.width / 2, cy = cv.height / 2 + 6, R = Math.min(cx, cy) - 70;
    var border = getCss("--border"), muted = getCss("--text-muted");

    ctx.strokeStyle = border;
    ctx.lineWidth = 1;
    for (var ring = 1; ring <= 4; ring++) {
      ctx.beginPath();
      for (var a = 0; a <= axes.length; a++) {
        var ang = (Math.PI * 2 * a) / axes.length - Math.PI / 2;
        var rr = (R * ring) / 4;
        var x = cx + rr * Math.cos(ang), y = cy + rr * Math.sin(ang);
        if (a === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.fillStyle = muted;
    ctx.font = "11px Inter, sans-serif";
    axes.forEach(function (name, i) {
      var ang = (Math.PI * 2 * i) / axes.length - Math.PI / 2;
      var x = cx + R * Math.cos(ang), y = cy + R * Math.sin(ang);
      ctx.strokeStyle = border;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke();
      var lx = cx + (R + 22) * Math.cos(ang), ly = cy + (R + 14) * Math.sin(ang);
      ctx.textAlign = Math.abs(Math.cos(ang)) < 0.3 ? "center" : (Math.cos(ang) > 0 ? "left" : "right");
      ctx.fillText(prettyLabel(name).slice(0, 16), lx, ly);
    });

    var rows = state.vc.filter(function (r) { return state.selectedCaps.indexOf(r.model_name) >= 0; });
    rows.forEach(function (r, idx) {
      var color = PALETTE[idx % PALETTE.length];
      var caps = capabilityData(r);
      ctx.beginPath();
      axes.forEach(function (name, i) {
        var v = caps[name] != null ? caps[name] : 0;
        var ang = (Math.PI * 2 * i) / axes.length - Math.PI / 2;
        var rr = R * v;
        var x = cx + rr * Math.cos(ang), y = cy + rr * Math.sin(ang);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.fillStyle = hexA(color, 0.15);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.fill(); ctx.stroke();
      if (legend) {
        legend.appendChild(el("span", "legend-item",
          '<span class="legend-dot" style="background:' + color + '"></span>' + esc(r.model_name)));
      }
    });
  }

  /* ------------------------------------------------------- CoT delta chart */
  function renderCotChart() {
    var cv = $("cot_canvas");
    if (!cv) return;
    var ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
    var rows = state.spatial.filter(function (r) { return r.diagnostics && r.diagnostics.cot_delta != null; });
    var muted = getCss("--text-muted");
    if (!rows.length) {
      ctx.fillStyle = muted; ctx.font = "14px Inter, sans-serif"; ctx.textAlign = "center";
      ctx.fillText("Submit Task-3 with CoT predictions to populate.", cv.width / 2, cv.height / 2);
      return;
    }
    var pad = 90, top = 20, bottom = cv.height - 40;
    var maxAbs = 0.05;
    rows.forEach(function (r) { maxAbs = Math.max(maxAbs, Math.abs(r.diagnostics.cot_delta)); });
    var midX = pad + (cv.width - pad - 20) / 2;
    var half = (cv.width - pad - 20) / 2;
    var slot = (bottom - top) / rows.length;
    var bh = Math.max(8, slot - 8);
    var border = getCss("--border");
    ctx.strokeStyle = border; ctx.beginPath(); ctx.moveTo(midX, top); ctx.lineTo(midX, bottom); ctx.stroke();
    ctx.font = "12px Inter, sans-serif";
    rows.forEach(function (r, i) {
      var v = r.diagnostics.cot_delta;
      var y = top + i * slot + 4;
      var w = (Math.abs(v) / maxAbs) * half;
      var color = v < 0 ? (getCss("--danger") || "#dc2626") : (getCss("--success") || "#16a34a");
      ctx.fillStyle = color;
      if (v < 0) ctx.fillRect(midX - w, y, w, bh);
      else ctx.fillRect(midX, y, w, bh);
      ctx.fillStyle = muted; ctx.textAlign = "right";
      ctx.fillText(r.model_name.slice(0, 14), pad - 8, y + bh / 2 + 4);
      ctx.fillStyle = getCss("--text"); ctx.textAlign = v < 0 ? "right" : "left";
      ctx.fillText(fmtDelta(v) + "%", v < 0 ? midX - w - 6 : midX + w + 6, y + bh / 2 + 4);
    });
  }

  /* ------------------------------------------------------------- submit UI */
  var SUBMIT_TASKS = [
    { id: "do_you_see_me", label: "Do-You-See-Me", section: "Perception (Visual Cognition)", harness: false },
    { id: "minds_eye", label: "Mind's-Eye", section: "Imagery (Visual Cognition)", harness: false },
    { id: "spatial", label: "Spatial Reasoning", section: "Task 3 (13 benchmarks)", harness: true },
  ];

  function renderSubmitCards() {
    var grid = $("submit_cards");
    if (!grid) return;
    grid.innerHTML = "";
    SUBMIT_TASKS.forEach(function (t) {
      var card = el("div", "card submit-card");
      var harnessRow = t.harness
        ? '<div class="dl-row"><a class="btn ghost" href="/api/spatial/manifest">Manifest (JSON)</a></div>' +
          '<p class="muted small">Run the harness in <code>spatial_harness/</code> to produce the response file (standard + cot + no_image + no_image_plus).</p>'
        : "";
      card.innerHTML =
        "<h2>" + esc(t.label) + "</h2>" +
        '<p class="muted small">' + esc(t.section) + "</p>" +
        '<div class="dl-row">' +
          '<a class="btn" href="/api/tasks/' + t.id + '/questions">Questions (JSON)</a>' +
          '<a class="btn ghost" href="/api/tasks/' + t.id + '/template.json">Template (JSON)</a>' +
          '<a class="btn ghost" href="/api/tasks/' + t.id + '/template.csv">Template (CSV)</a>' +
        "</div>" +
        harnessRow +
        '<form class="task-submit-form" data-task="' + t.id + '">' +
          '<label class="field"><span>Model name <em>*</em></span>' +
            '<input type="text" name="model_name" maxlength="255" placeholder="e.g. GPT-4o" required></label>' +
          '<label class="field"><span>Response file (JSON/CSV) <em>*</em></span>' +
            '<input type="file" name="file" accept=".json,.csv" required></label>' +
          '<label class="field"><span>API token <small>(optional)</small></span>' +
            '<input type="text" name="api_token" placeholder="Bearer token if required"></label>' +
          '<button type="submit" class="btn primary">Submit ' + esc(t.label) + "</button>" +
          '<div class="form-msg" role="status"></div>' +
        "</form>";
      grid.appendChild(card);
    });
    grid.querySelectorAll(".task-submit-form").forEach(function (form) {
      form.addEventListener("submit", onSubmitTask);
    });
  }

  async function onSubmitTask(e) {
    e.preventDefault();
    var form = e.currentTarget;
    var taskId = form.getAttribute("data-task");
    var msg = form.querySelector(".form-msg");
    var btn = form.querySelector("button[type=submit]");
    var fd = new FormData();
    fd.append("model_name", form.model_name.value.trim());
    fd.append("file", form.file.files[0]);
    var token = (form.api_token.value || "").trim();
    msg.textContent = "Scoring…"; msg.className = "form-msg"; btn.disabled = true;
    try {
      var headers = {};
      if (token) headers["Authorization"] = "Bearer " + token;
      var r = await fetch("/api/tasks/" + taskId + "/submit", { method: "POST", body: fd, headers: headers });
      var data = await r.json();
      if (!r.ok) throw new Error(data.error || ("HTTP " + r.status));
      msg.className = "form-msg ok";
      msg.innerHTML = "Scored: <strong>" + fmtPct(data.accuracy) + "</strong> over " +
        (data.total_samples || 0) + " samples.";
      await refreshLeaderboards();
    } catch (err) {
      msg.className = "form-msg err";
      msg.textContent = "Error: " + err.message;
    } finally {
      btn.disabled = false;
    }
  }

  async function refreshLeaderboards() {
    await Promise.all([loadVisualCognition(), loadSpatial(), loadStats()]);
  }

  /* ----------------------------------------------------------- report modal */
  async function openReport(modelName) {
    var modal = $("report_modal"), content = $("report_content");
    if (!modal || !content) return;
    modal.hidden = false;
    content.innerHTML = '<p class="muted">Loading…</p>';
    try {
      var r = await getJSON("/api/model/" + encodeURIComponent(modelName) + "/report");
      content.innerHTML = reportHtml(r);
    } catch (e) {
      content.innerHTML = '<p class="err">Failed to load report: ' + esc(e.message) + "</p>";
    }
  }
  function closeReport() { var m = $("report_modal"); if (m) m.hidden = true; }

  function groupsTable(groups) {
    var keys = Object.keys(groups || {});
    if (!keys.length) return '<p class="muted small">—</p>';
    var rows = keys.map(function (k) {
      var g = groups[k];
      return "<tr><td>" + esc(prettyLabel(k)) + "</td><td class='num'>" + fmtPct(g.accuracy) +
        "</td><td class='num'>" + g.correct_samples + "/" + g.total_samples + "</td></tr>";
    }).join("");
    return "<table class='lb-table small'><thead><tr><th>Group</th><th class='num'>Acc.</th><th class='num'>n</th></tr></thead><tbody>" + rows + "</tbody></table>";
  }

  function reportHtml(r) {
    var vc = r.visual_cognition || {};
    var tasks = r.tasks || {};
    var html = '<h2 id="report_title">' + esc(r.model_name) + "</h2>";
    html += '<div class="kpi-row">';
    html += '<div class="kpi"><div class="kpi-label">VCI</div><div class="kpi-val">' + fmtVci(vc.vci) + "</div></div>";
    html += '<div class="kpi"><div class="kpi-label">Perception</div><div class="kpi-val">' + fmtPct(vc.perception_accuracy) + "</div></div>";
    html += '<div class="kpi"><div class="kpi-label">Imagery</div><div class="kpi-val">' + fmtPct(vc.imagery_accuracy) + "</div></div>";
    var sp = tasks.spatial;
    html += '<div class="kpi"><div class="kpi-label">Spatial</div><div class="kpi-val">' + (sp ? fmtPct(sp.accuracy) : "—") + "</div></div>";
    html += "</div>";

    if (tasks.do_you_see_me) {
      html += "<h3>Do-You-See-Me — capabilities</h3>" + groupsTable(tasks.do_you_see_me.groups);
    }
    if (tasks.minds_eye) {
      html += "<h3>Mind's-Eye — capabilities</h3>" + groupsTable(tasks.minds_eye.groups);
    }
    if (sp) {
      html += "<h3>Spatial — per benchmark</h3>" + groupsTable(sp.groups);
      var d = sp.diagnostics;
      if (d) {
        html += "<h3>Diagnostics</h3><div class='kpi-row'>";
        html += "<div class='kpi'><div class='kpi-label'>CoT Δ</div><div class='kpi-val " +
          (d.cot_delta < 0 ? "neg" : "pos") + "'>" + fmtDelta(d.cot_delta) + "%</div></div>";
        html += "<div class='kpi'><div class='kpi-label'>Shortcut ↓</div><div class='kpi-val'>" + fmtPct(d.shortcut_score) + "</div></div>";
        html += "<div class='kpi'><div class='kpi-label'>Halluc. ↑</div><div class='kpi-val'>" + fmtPct(d.hallucination_resistance) + "</div></div>";
        html += "</div>";
      }
    }
    return html;
  }

  function initModal() {
    document.querySelectorAll("[data-close]").forEach(function (b) {
      b.addEventListener("click", closeReport);
    });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") closeReport(); });
  }

  /* ------------------------------------------------------------------- boot */
  function boot() {
    initTheme();
    initTabs();
    initModal();
    checkHealth();
    setInterval(checkHealth, 30000);
    loadSections();
    loadTaskInfo();
    refreshLeaderboards();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
