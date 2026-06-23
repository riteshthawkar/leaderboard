/* Shared site chrome: theme, health pill, mobile nav, active link, copy buttons.
   Loaded on every page. Page-specific interactivity (leaderboard tables,
   submit forms) lives in main.js, which listens for the "themechange" event
   to re-render its canvases. */
(function () {
  "use strict";

  function $(id) { return document.getElementById(id); }

  /* ---- theme ---- */
  function initTheme() {
    var saved = localStorage.getItem("vci-theme");
    if (saved) document.documentElement.setAttribute("data-theme", saved);
    var btn = $("theme_toggle");
    if (!btn) return;
    btn.addEventListener("click", function () {
      var next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem("vci-theme", next);
      window.dispatchEvent(new Event("themechange"));
    });
  }

  /* ---- backend health pill ---- */
  function checkHealth() {
    var dot = $("status_dot"), txt = $("status_text");
    fetch("/api/health").then(function (r) {
      if (dot) dot.classList.toggle("ok", r.ok);
      if (txt) txt.textContent = r.ok ? "Online" : "Offline";
    }).catch(function () {
      if (dot) dot.classList.remove("ok");
      if (txt) txt.textContent = "Offline";
    });
  }

  /* ---- mobile nav ---- */
  function initNav() {
    var toggle = $("nav_toggle"), menu = $("nav_menu");
    if (toggle && menu) {
      toggle.addEventListener("click", function () { menu.classList.toggle("open"); });
      menu.addEventListener("click", function (e) {
        if (e.target.classList.contains("nav-link")) menu.classList.remove("open");
      });
    }
    // active link based on body[data-page]
    var page = document.body.getAttribute("data-page");
    document.querySelectorAll(".nav-link[data-nav]").forEach(function (a) {
      if (a.getAttribute("data-nav") === page) a.classList.add("is-active");
    });
  }

  /* ---- copy-to-clipboard for citations ---- */
  function initCopy() {
    document.querySelectorAll(".copy-btn[data-copy]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var src = $(btn.getAttribute("data-copy"));
        if (!src) return;
        var text = src.textContent || "";
        var done = function () { var t = btn.textContent; btn.textContent = "Copied"; setTimeout(function () { btn.textContent = t; }, 1400); };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done).catch(done);
        } else {
          var ta = document.createElement("textarea"); ta.value = text; document.body.appendChild(ta);
          ta.select(); try { document.execCommand("copy"); } catch (e) {} document.body.removeChild(ta); done();
        }
      });
    });
  }

  function boot() {
    initTheme();
    initNav();
    initCopy();
    checkHealth();
    setInterval(checkHealth, 30000);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
