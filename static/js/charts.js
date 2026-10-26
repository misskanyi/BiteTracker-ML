/* BiteTracker chart rendering and theming.
 *
 * Chart specs come from web/core/charts.py via json_script. Each trace carries a
 * colour role in `meta` (e.g. "series-1", "predicted", or a list of roles for
 * per-bar colours). Roles resolve to CSS custom properties (--bt-chart-*), so
 * colours, grid, text and hover styling follow the active theme. Switching theme
 * only restyles the charts in the browser: no request is made and no data changes.
 */
(function () {
  "use strict";
  var charts = [];

  function token(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function roleColour(role) {
    return token("--bt-chart-" + role) || token("--bt-chart-other");
  }

  function themeLayout() {
    var text = token("--bt-text"), muted = token("--bt-text-muted"), grid = token("--bt-chart-grid");
    var axis = { gridcolor: grid, zerolinecolor: grid, linecolor: grid, tickfont: { color: muted }, title: { font: { color: muted } } };
    return {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: token("--bt-font-sans"), color: muted, size: 12 },
      "title.font.color": text,
      "title.font.size": 15,
      "legend.font.color": text,
      "xaxis.gridcolor": axis.gridcolor, "xaxis.zerolinecolor": axis.zerolinecolor, "xaxis.linecolor": axis.linecolor,
      "xaxis.tickfont.color": muted, "xaxis.title.font.color": muted,
      "yaxis.gridcolor": axis.gridcolor, "yaxis.zerolinecolor": axis.zerolinecolor, "yaxis.linecolor": axis.linecolor,
      "yaxis.tickfont.color": muted, "yaxis.title.font.color": muted,
      hoverlabel: { bgcolor: token("--bt-surface-elevated"), bordercolor: token("--bt-border-strong"), font: { color: text, family: token("--bt-font-sans") } },
    };
  }

  // Only attributes the trace already uses are touched, so no new visual elements appear.
  function traceUpdate(trace) {
    var role = trace.meta;
    if (!role) return null;
    var update = {};
    if (Array.isArray(role)) {
      update["marker.color"] = [role.map(roleColour)];
    } else {
      var colour = roleColour(role);
      if (trace.type === "bar") update["marker.color"] = colour;
      else { update["line.color"] = colour; update["marker.color"] = colour; }
    }
    if (trace.error_y && trace.error_y.visible) update["error_y.color"] = token("--bt-text-muted");
    return update;
  }

  function applyTheme(el) {
    var done = [Plotly.relayout(el, themeLayout())];
    el.data.forEach(function (trace, i) {
      var update = traceUpdate(trace);
      if (update) done.push(Plotly.restyle(el, update, [i]));
    });
    return Promise.all(done);
  }

  // --- Scroll-in reveal ------------------------------------------------------------
  // Charts draw straight away (so layout never jumps) but their marks stay hidden
  // (.is-pending) until the chart scrolls into view. Then lines wipe in left to right,
  // bars grow from the axis one after another, and value labels fade in last (CSS).
  // The OS setting or the user's own "Reduce animations" preference (Settings → Appearance).
  var reduceMotion = document.documentElement.classList.contains("bt-reduce-motion")
    || (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  var REVEAL_MS = 1600;

  // Plotly rebuilds bar nodes on every redraw, so the stagger delays are re-applied after each one.
  function staggerBars(el) {
    el.querySelectorAll(".barlayer .trace").forEach(function (trace, t) {
      trace.querySelectorAll(".point").forEach(function (point, i) {
        point.style.setProperty("--bt-delay", (i * 70 + t * 40) + "ms");
      });
    });
  }

  function reveal(el) {
    if (!el.classList.contains("is-pending")) return;
    staggerBars(el);
    el.classList.remove("is-pending");
    el.classList.add("is-animating");
    setTimeout(function () { el.classList.remove("is-animating"); }, REVEAL_MS);
  }

  var observer = !reduceMotion && "IntersectionObserver" in window
    ? new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          observer.unobserve(entry.target);
          reveal(entry.target);
        });
      }, { threshold: 0.25 })
    : null;

  function render(el) {
    var spec = JSON.parse(document.getElementById(el.dataset.chart).textContent);
    if (el.hasAttribute("data-hide-title")) {
      // The card heading names the chart; the title stays in the spec for the aria-label.
      spec.layout.title = { text: "" };
      spec.layout.margin = Object.assign({}, spec.layout.margin, { t: 32 }); // room for the hover toolbar
    }
    if (el.hasAttribute("data-fill")) {
      // Height comes from the card (CSS), so the chart fills space next to taller sidebars.
      delete spec.layout.height;
      spec.layout.autosize = true;
    }
    if (el.clientWidth < 520) {
      // Phones: dates on the x axis are self-explanatory, and the axis title collides with the legend.
      spec.layout.xaxis = Object.assign({}, spec.layout.xaxis, { title: { text: "" } });
      spec.layout.margin = Object.assign({}, spec.layout.margin, { l: 44, r: 8 });
    }
    var config = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["select2d", "lasso2d"] };
    if (observer) el.classList.add("is-pending");
    Plotly.newPlot(el, spec.data, spec.layout, config).then(function () {
      el.on("plotly_afterplot", function () { staggerBars(el); });
      return applyTheme(el);
    }).then(function () {
      el.classList.add("is-ready");
      if (observer) observer.observe(el);
    });
    charts.push(el);
  }

  // Plotly measures labels when it draws, so wait for the self-hosted font first (falls back immediately if unsupported).
  var fontsReady = document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve();
  fontsReady.then(function () { document.querySelectorAll("[data-chart]").forEach(render); });
  document.addEventListener("bt:themechange", function () { charts.forEach(applyTheme); });
})();
