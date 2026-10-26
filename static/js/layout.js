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

  // Branded loaders appear only if the wait outlasts LOADER_DELAY, so fast responses never flash one.
  var LOADER_DELAY = 400;
  var PAGE_LOADER_TIMEOUT = 15000;
  var loaderTimers = [];

  function showLoaderLater(show) {
    loaderTimers.push(window.setTimeout(show, LOADER_DELAY));
  }

  function hideLoaders() {
    loaderTimers.forEach(window.clearTimeout);
    loaderTimers = [];
    document.querySelectorAll("[data-loader-overlay], #bt-page-loader").forEach(function (el) { el.hidden = true; });
    var status = document.getElementById("bt-page-loader-status");
    if (status) status.textContent = "";
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

    // Show/hide password fields from an inline eye button.
    document.querySelectorAll("[data-password-toggle]").forEach(function (button) {
      var input = document.getElementById(button.dataset.passwordToggle);
      if (!input) return;
      button.addEventListener("click", function () {
        var show = input.type === "password";
        input.type = show ? "text" : "password";
        button.setAttribute("aria-pressed", show ? "true" : "false");
        button.setAttribute("aria-label", show ? "Hide password" : "Show password");
        var icon = button.querySelector(".bi");
        if (icon) icon.className = show ? "bi bi-eye-slash" : "bi bi-eye";
        input.focus();
      });
    });

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
        // Long operations also cover the form with the ring loader; the status line above announces it.
        var overlay = form.querySelector("[data-loader-overlay]");
        if (overlay) showLoaderLater(function () { overlay.hidden = false; });
      });
    });

    // Slow pages: show the pulsing logo while the server prepares the next page.
    var pageLoader = document.getElementById("bt-page-loader");
    if (pageLoader) {
      document.querySelectorAll("a[data-loading-nav]").forEach(function (link) {
        link.addEventListener("click", function (event) {
          // New-tab and modified clicks leave this page in place, so there is nothing to wait for here.
          if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
          var message = link.dataset.loadingNav;
          showLoaderLater(function () {
            pageLoader.querySelector("[data-loader-message]").textContent = message;
            pageLoader.hidden = false;
            var status = document.getElementById("bt-page-loader-status");
            if (status) status.textContent = message;
          });
          // A stopped or failed navigation fires no event; never leave the page behind an endless loader.
          loaderTimers.push(window.setTimeout(hideLoaders, PAGE_LOADER_TIMEOUT));
        });
      });
      // Escape is how users usually stop a load.
      document.addEventListener("keydown", function (event) {
        if (event.key === "Escape" && !pageLoader.hidden) hideLoaders();
      });
    }
  });

  // Back/forward cache restores the page as it was left, so undo the busy state.
  window.addEventListener("pageshow", function (event) {
    if (!event.persisted) return;
    hideLoaders();
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
