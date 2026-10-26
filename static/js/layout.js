/* BiteTracker layout behaviour: sidebar states, collapsed-state tooltips, form loading states. */
(function () {
  "use strict";
  var root = document.documentElement;
  var STORAGE_KEY = "bt-sidebar-pinned";
  var desktop = window.matchMedia("(min-width: 992px)");
  var canHover = window.matchMedia("(hover: hover)");
  var tooltips = [];

  /* Desktop sidebar state. Two inputs, kept apart:
     - pinned: the saved preference (html.bt-sidebar-pinned, also set before paint by _theme_init.html);
     - peek: a temporary expansion while the pointer rests on the sidebar or the keyboard is inside it
       (aside.is-peek). Peeking floats over the page; pinning makes the page shift over.
     Visible state = pinned ? "pinned" : peek ? "peek" : "collapsed". */
  var OPEN_DELAY = 120;   // hover intent: a cursor just crossing the rail does not open it
  var CLOSE_DELAY = 180;  // grace period: brief exits (into a tooltip, past the edge) do not close it
  var sidebar = null;
  var hovering = false;   // a mouse or pen is over the sidebar
  var keyboard = false;   // keyboard focus is inside the sidebar
  var peekTimer = null;

  function isPinned() {
    return root.classList.contains("bt-sidebar-pinned");
  }

  function isPeeking() {
    return !!sidebar && sidebar.classList.contains("is-peek");
  }

  // Labels are on screen when pinned or peeking. Tooltips stand in for them only where hovering
  // can't open the sidebar (touch screens); elsewhere they would flash just before the labels slide in.
  function syncTooltips() {
    var labelsHidden = desktop.matches && !isPinned() && !isPeeking();
    tooltips.forEach(function (entry) {
      if (entry.always || (labelsHidden && !canHover.matches)) entry.tip.enable();
      else { entry.tip.hide(); entry.tip.disable(); }
    });
  }

  function syncControls() {
    var pinned = isPinned();
    var pin = document.getElementById("bt-sidebar-pin");
    if (pin) {
      var label = pinned ? "Unpin sidebar" : "Pin sidebar open";
      pin.setAttribute("aria-pressed", pinned ? "true" : "false");
      pin.setAttribute("aria-label", label);
      pin.setAttribute("data-bt-tooltip", label);
      pin.querySelector(".bi").className = pinned ? "bi bi-pin-angle-fill" : "bi bi-pin-angle";
    }
    var toggle = document.getElementById("bt-sidebar-toggle");
    if (toggle) {
      toggle.setAttribute("aria-expanded", pinned ? "true" : "false");
      toggle.setAttribute("aria-label", pinned ? "Collapse sidebar" : "Expand sidebar");
    }
  }

  function setPeek(open) {
    window.clearTimeout(peekTimer);
    peekTimer = null;
    if (!sidebar) return;
    open = open && desktop.matches && !isPinned();
    if (open === isPeeking()) return;
    sidebar.classList.toggle("is-peek", open);
    syncTooltips();
  }

  // Cancels whatever is pending and schedules the opposite only if it would change something,
  // so rapid in/out movement settles on the latest intent instead of queueing transitions.
  function requestPeek(open, delay) {
    window.clearTimeout(peekTimer);
    peekTimer = null;
    if (open === isPeeking()) return;
    peekTimer = window.setTimeout(function () { setPeek(open); }, delay);
  }

  function setPinned(pinned) {
    tooltips.forEach(function (entry) { entry.tip.hide(); });
    root.classList.toggle("bt-sidebar-pinned", pinned);
    try {
      localStorage.setItem(STORAGE_KEY, pinned ? "1" : "0");
      localStorage.removeItem("bt-sidebar");  // the old expand/collapse key, read once by _theme_init.html
    } catch (e) { /* this page view only */ }
    // Unpinning while the pointer or focus is still inside keeps it open until they leave;
    // pinning takes over from a peek in the same frame, so the width never dips.
    if (sidebar) sidebar.classList.toggle("is-peek", !pinned && desktop.matches && (hovering || keyboard));
    window.clearTimeout(peekTimer);
    syncControls();
    syncTooltips();
  }

  // Charts and other width-dependent content re-measure once the page has made room for a
  // pinned sidebar (peeking overlays the page, so it never changes the content width).
  function announceResize() {
    window.dispatchEvent(new Event("resize"));
  }

  function initSidebar() {
    sidebar = document.getElementById("bt-sidebar");
    if (!sidebar) return;

    sidebar.addEventListener("pointerenter", function (event) {
      if (event.pointerType === "touch") return;  // touch taps navigate; the topbar toggle expands
      hovering = true;
      requestPeek(true, OPEN_DELAY);
    });
    sidebar.addEventListener("pointerleave", function (event) {
      if (event.pointerType === "touch") return;
      hovering = false;
      if (!keyboard) requestPeek(false, CLOSE_DELAY);
    });

    // Keyboard users get the labels as soon as focus enters; a mouse click inside does not count.
    sidebar.addEventListener("focusin", function (event) {
      if (!event.target.matches(":focus-visible")) return;
      keyboard = true;
      setPeek(true);
    });
    sidebar.addEventListener("focusout", function (event) {
      if (sidebar.contains(event.relatedTarget)) return;
      keyboard = false;
      if (!hovering) requestPeek(false, CLOSE_DELAY);
    });
    sidebar.addEventListener("keydown", function (event) {
      if (event.key !== "Escape" || !isPeeking()) return;
      keyboard = false;
      hovering = false;
      setPeek(false);
    });

    var pin = document.getElementById("bt-sidebar-pin");
    if (pin) pin.addEventListener("click", function () { setPinned(!isPinned()); });
    var toggle = document.getElementById("bt-sidebar-toggle");
    if (toggle) toggle.addEventListener("click", function () { setPinned(!isPinned()); });

    // Mobile drawer: close it as soon as a link is chosen, rather than while the next page loads.
    sidebar.addEventListener("click", function (event) {
      if (desktop.matches || !event.target.closest("a[href]") || !window.bootstrap) return;
      var drawer = bootstrap.Offcanvas.getInstance(sidebar);
      if (drawer) drawer.hide();
    });

    var main = document.querySelector(".bt-main");
    if (main) {
      main.addEventListener("transitionend", function (event) {
        if (event.target === main && event.propertyName === "margin-left") announceResize();
      });
    }

    function onLayoutChange() {
      if (!desktop.matches) { hovering = false; keyboard = false; setPeek(false); }
      syncTooltips();
    }
    if (desktop.addEventListener) {
      desktop.addEventListener("change", onLayoutChange);
      canHover.addEventListener("change", syncTooltips);
    }
    // Another tab changed the preference: follow it.
    window.addEventListener("storage", function (event) {
      if (event.key === STORAGE_KEY) setPinned(event.newValue === "1");
    });

    syncControls();
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
    var nav = document.getElementById("bt-sidebar");
    if (nav && window.bootstrap) {
      nav.querySelectorAll("[data-bt-tooltip]").forEach(function (el) {
        var isPin = el.id === "bt-sidebar-pin";
        tooltips.push({
          always: isPin,  // the pin's tooltip explains it in every state; nav tooltips only replace hidden labels
          tip: new bootstrap.Tooltip(el, {
            // Read on every show, so the pin's tooltip follows its pinned/unpinned label.
            title: function () { return el.getAttribute("data-bt-tooltip"); },
            placement: isPin ? "bottom" : "right",
            // The pin keeps focus after a click, so a focus trigger would leave its tooltip stuck open.
            trigger: isPin ? "hover" : "hover focus",
            delay: { show: isPin ? 300 : 0, hide: 0 },
            customClass: "bt-tooltip",
          }),
        });
      });
    }
    initSidebar();

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
