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
    Plotly.relayout(el, themeLayout());
    el.data.forEach(function (trace, i) {
      var update = traceUpdate(trace);
      if (update) Plotly.restyle(el, update, [i]);
    });
  }

  function render(el) {
    var spec = JSON.parse(document.getElementById(el.dataset.chart).textContent);
    var config = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["select2d", "lasso2d"] };
    Plotly.newPlot(el, spec.data, spec.layout, config).then(function () {
      applyTheme(el);
      el.classList.add("is-ready");
    });
    charts.push(el);
  }

  document.querySelectorAll("[data-chart]").forEach(render);
  document.addEventListener("bt:themechange", function () { charts.forEach(applyTheme); });
})();
