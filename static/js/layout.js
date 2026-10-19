/* BiteTracker layout behaviour: sidebar collapse, collapsed-state tooltips, form loading states. */
(function () {
  "use strict";
  var root = document.documentElement;
  var STORAGE_KEY = "bt-sidebar";
  var desktop = window.matchMedia("(min-width: 992px)");
  var tooltips = [];

  function isCollapsed() {
    return root.classList.contains("bt-sidebar-collapsed");
  }

  // Tooltips only make sense when labels are hidden (collapsed, desktop).
  function syncTooltips() {
    var enable = isCollapsed() && desktop.matches;
    tooltips.forEach(function (tip) { if (enable) tip.enable(); else { tip.hide(); tip.disable(); } });
  }

  function syncToggle() {
    var button = document.getElementById("bt-collapse-toggle");
    if (!button) return;
    var collapsed = isCollapsed();
    button.setAttribute("aria-expanded", collapsed ? "false" : "true");
    button.setAttribute("aria-label", collapsed ? "Expand sidebar" : "Collapse sidebar");
  }

  // Charts and other width-dependent content re-measure after the sidebar animates.
  function announceResize() {
    window.dispatchEvent(new Event("resize"));
  }

  function setCollapsed(collapsed) {
    tooltips.forEach(function (tip) { tip.hide(); });
    root.classList.toggle("bt-sidebar-collapsed", collapsed);
    try { localStorage.setItem(STORAGE_KEY, collapsed ? "collapsed" : "expanded"); } catch (e) { /* session only */ }
    syncToggle();
    syncTooltips();
  }

  document.addEventListener("DOMContentLoaded", function () {
    var sidebar = document.getElementById("bt-sidebar");
    if (sidebar && window.bootstrap) {
      sidebar.querySelectorAll("[data-bt-tooltip]").forEach(function (el) {
        // The collapse button keeps focus after a click, so a focus trigger would leave its tooltip stuck open.
        var trigger = el.id === "bt-collapse-toggle" ? "hover" : "hover focus";
        tooltips.push(new bootstrap.Tooltip(el, {
          title: el.getAttribute("data-bt-tooltip"), placement: "right", trigger: trigger,
          customClass: "bt-tooltip",
        }));
      });
      sidebar.addEventListener("transitionend", function (event) {
        if (event.target === sidebar && event.propertyName === "width") announceResize();
      });
    }

    var toggle = document.getElementById("bt-collapse-toggle");
    if (toggle) toggle.addEventListener("click", function () { setCollapsed(!isCollapsed()); });
    syncToggle();
    syncTooltips();
    if (desktop.addEventListener) desktop.addEventListener("change", syncTooltips);

    // Disable submit buttons and show progress while a form is being processed.
    document.querySelectorAll("form[data-loading]").forEach(function (form) {
      form.addEventListener("submit", function () {
        // Announce the wait to screen readers; buttons are disabled to prevent duplicate submissions.
        var status = form.querySelector("[data-loading-status]");
        if (status) status.textContent = form.dataset.loading + " This can take a few seconds.";
        form.setAttribute("aria-busy", "true");
        form.querySelectorAll("button[type=submit]").forEach(function (button) {
          button.dataset.label = button.innerHTML;
          button.disabled = true;
          button.setAttribute("aria-busy", "true");
          button.innerHTML = '<span class="spinner-border spinner-border-sm me-2" aria-hidden="true"></span>' + form.dataset.loading;
        });
      });
    });
  });

  // Back/forward cache restores the page as it was left, so undo the busy state.
  window.addEventListener("pageshow", function (event) {
    if (!event.persisted) return;
    document.querySelectorAll("form[data-loading]").forEach(function (form) {
      form.removeAttribute("aria-busy");
      var status = form.querySelector("[data-loading-status]");
      if (status) status.textContent = "";
      form.querySelectorAll("button[type=submit][data-label]").forEach(function (button) {
        button.disabled = false;
        button.removeAttribute("aria-busy");
        button.innerHTML = button.dataset.label;
      });
    });
  });
})();
