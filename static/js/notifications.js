/*
 * Live notifications: keeps the bell current and shows new notifications as toasts.
 *
 * - Polls notifications:feed every 20 s while the tab is visible (no WebSockets, Redis or Celery).
 * - A toast is a preview only: closing it, or letting it time out after ~5 s, leaves the
 *   notification unread, so the bell's count is always the number of unread notifications.
 * - Toasts are shown once per browser: the highest notification id already seen is kept in
 *   localStorage (shared by tabs). Without storage, nothing older than the page is toasted.
 * - Every value from the server is inserted as text, never as HTML.
 */
(function () {
  "use strict";

  var POLL_MS = 20000;
  var MAX_POLL_MS = 300000;
  var TOAST_MS = 5000;
  var MAX_VISIBLE = 3;
  var ICON_RE = /^bi-[a-z0-9-]+$/;

  var root = document.querySelector("[data-bt-notify]");
  if (!root || !window.fetch) return;

  var feedUrl = root.getAttribute("data-feed-url");
  var storeKey = "bt-notify-seen-" + root.getAttribute("data-user");
  var countEl = root.querySelector("[data-notify-count]");
  var bellButton = root.querySelector("[data-notify-button]");
  var panel = root.querySelector("[data-notify-panel]");
  var markAllForm = root.querySelector("[data-notify-mark-all]");
  var region = document.getElementById("bt-toasts");
  var header = document.querySelector(".bt-topbar");
  var csrfMeta = document.querySelector('meta[name="csrf-token"]');

  var lastSeen = Number(root.getAttribute("data-latest-id")) || 0;
  var stored = readSeen();
  if (stored !== null) lastSeen = stored; else writeSeen(lastSeen);

  var interval = POLL_MS;
  var timer = null;
  var stopped = false;
  var panelSignature = "";
  var visibleToasts = [];
  var queue = [];

  // --- storage -------------------------------------------------------------------------
  function readSeen() {
    try {
      var value = window.localStorage.getItem(storeKey);
      return value === null ? null : Number(value) || 0;
    } catch (e) { return null; }
  }
  function writeSeen(id) {
    try { window.localStorage.setItem(storeKey, String(id)); } catch (e) { /* private mode */ }
  }
  window.addEventListener("storage", function (event) {
    if (event.key === storeKey && event.newValue) lastSeen = Math.max(lastSeen, Number(event.newValue) || 0);
  });

  // --- helpers ---------------------------------------------------------------------------
  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }
  function icon(name) {
    var i = el("i", "bi " + (ICON_RE.test(name || "") ? name : "bi-bell"));
    i.setAttribute("aria-hidden", "true");
    return i;
  }
  function csrfToken() { return csrfMeta ? csrfMeta.getAttribute("content") : ""; }
  function isInternal(url) { return typeof url === "string" && url.charAt(0) === "/" && url.charAt(1) !== "/"; }
  function postForm(action, extra) {
    // A real POST form (CSRF token included) so opening a notification works like the bell's own forms.
    var form = el("form");
    form.method = "post";
    form.action = action;
    var token = el("input"); token.type = "hidden"; token.name = "csrfmiddlewaretoken"; token.value = csrfToken();
    form.appendChild(token);
    Object.keys(extra || {}).forEach(function (key) {
      var input = el("input"); input.type = "hidden"; input.name = key; input.value = extra[key];
      form.appendChild(input);
    });
    return form;
  }
  function postJson(url) {
    return fetch(url, {
      method: "POST", credentials: "same-origin",
      headers: { "Accept": "application/json", "X-CSRFToken": csrfToken() },
    }).then(function (response) {
      if (!response.ok) throw new Error("HTTP " + response.status);
      return response.json();
    });
  }

  // --- bell --------------------------------------------------------------------------------
  function setCount(count) {
    var previous = Number(countEl.getAttribute("data-value") || countEl.textContent) || 0;
    countEl.setAttribute("data-value", String(count));
    countEl.textContent = count > 99 ? "99+" : String(count);
    countEl.hidden = count === 0;
    if (markAllForm) markAllForm.hidden = count === 0;
    bellButton.setAttribute("aria-label", count ? "Notifications, " + count + " unread" : "Notifications");
    if (count > previous) {
      countEl.classList.remove("is-bumped");
      void countEl.offsetWidth; // restart the animation
      countEl.classList.add("is-bumped");
    }
  }

  function renderItem(n) {
    var li = el("li", "bt-notify-item" + (n.is_read ? "" : " is-unread"));
    li.setAttribute("data-notification-id", n.id);

    var open = postForm(n.open_url);
    var button = el("button", "bt-notify-item__button");
    button.type = "submit";
    var iconWrap = el("span", "bt-notify-item__icon");
    iconWrap.setAttribute("aria-hidden", "true");
    iconWrap.appendChild(icon(n.icon));
    var body = el("span", "bt-notify-item__body");
    var title = el("span", "bt-notify-item__title");
    if (!n.is_read) title.appendChild(el("span", "visually-hidden", "Unread: "));
    title.appendChild(document.createTextNode(n.title));
    body.appendChild(title);
    if (n.message) {
      var message = n.message.length > 110 ? n.message.slice(0, 109) + "…" : n.message;
      body.appendChild(el("span", "bt-notify-item__message", message));
    }
    var time = el("span", "bt-notify-item__time");
    var t = el("time", null, n.ago);
    t.setAttribute("datetime", n.created_at);
    t.title = n.created_display;
    time.appendChild(t);
    if (n.has_destination) {
      time.appendChild(document.createTextNode(" · "));
      time.appendChild(el("span", "bt-notify-item__dest", "Open"));
    }
    body.appendChild(time);
    button.appendChild(iconWrap);
    button.appendChild(body);
    open.appendChild(button);
    li.appendChild(open);

    var tools = el("div", "bt-notify-item__tools");
    var here = window.location.pathname + window.location.search;
    if (!n.is_read) {
      var read = postForm(n.read_url, { next: here });
      read.setAttribute("data-notify-read", "");
      var readBtn = el("button", "bt-dismiss");
      readBtn.type = "submit";
      readBtn.title = "Mark as read";
      readBtn.setAttribute("aria-label", "Mark as read: " + n.title);
      readBtn.appendChild(icon("bi-check2"));
      read.appendChild(readBtn);
      tools.appendChild(read);
    }
    var del = postForm(n.delete_url, { next: here });
    var delBtn = el("button", "bt-dismiss");
    delBtn.type = "submit";
    delBtn.title = "Remove";
    delBtn.setAttribute("aria-label", "Remove notification: " + n.title);
    delBtn.appendChild(icon("bi-x-lg"));
    del.appendChild(delBtn);
    tools.appendChild(del);
    li.appendChild(tools);
    return li;
  }

  function renderPanel(items) {
    var signature = items.map(function (n) { return n.id + (n.is_read ? "r" : "u"); }).join(",");
    if (signature === panelSignature) return;
    // Don't rebuild under the user's focus (it would drop keyboard focus); next poll catches up.
    if (panelSignature && panel.contains(document.activeElement)) return;
    panelSignature = signature;
    panel.textContent = "";
    if (!items.length) {
      var empty = el("div", "empty-state py-4");
      empty.appendChild(icon("bi-bell-slash"));
      empty.appendChild(el("p", "mb-0 small", "You're all caught up."));
      panel.appendChild(empty);
      return;
    }
    var list = el("ul", "list-unstyled mb-0 bt-notify-list");
    items.forEach(function (n) { list.appendChild(renderItem(n)); });
    panel.appendChild(list);
  }

  // Mark one / all as read without leaving the page (forms still work without JavaScript).
  root.addEventListener("submit", function (event) {
    var form = event.target;
    var isRead = form.hasAttribute("data-notify-read");
    var isAll = form.hasAttribute("data-notify-mark-all");
    if (!isRead && !isAll) return;
    event.preventDefault();
    postJson(form.action).then(function (data) {
      setCount(data.unread_count);
      if (isAll) {
        panel.querySelectorAll(".is-unread").forEach(function (li) { li.classList.remove("is-unread"); });
        panel.querySelectorAll("[data-notify-read]").forEach(function (f) { f.remove(); });
        panel.querySelectorAll(".bt-notify-item__title .visually-hidden").forEach(function (s) { s.remove(); });
      } else {
        var li = form.closest(".bt-notify-item");
        if (li) {
          li.classList.remove("is-unread");
          var hidden = li.querySelector(".bt-notify-item__title .visually-hidden");
          if (hidden) hidden.remove();
        }
        var next = form.closest(".bt-notify-item__tools");
        form.remove();
        var focusTarget = next && next.querySelector("button");
        if (focusTarget) focusTarget.focus();
      }
      panelSignature = "";
      poll();
    }).catch(function () { form.submit(); });
  });

  // --- toasts ------------------------------------------------------------------------------
  // Keep the stack just under the header (whose position moves with the data banner and scrolling),
  // so toasts never cover the bell or the account menu.
  function placeRegion() {
    if (!region || !header) return;
    region.style.top = Math.max(header.getBoundingClientRect().bottom, 0) + 12 + "px";
  }
  var placing = false;
  window.addEventListener("scroll", function () {
    if (placing || !visibleToasts.length) return;
    placing = true;
    window.requestAnimationFrame(function () { placing = false; placeRegion(); });
  }, { passive: true });
  window.addEventListener("resize", placeRegion);

  function panelOpen() { return bellButton.getAttribute("aria-expanded") === "true"; }

  function showToast(n) {
    if (!region || panelOpen()) return; // the open panel already shows it
    placeRegion();
    if (visibleToasts.length >= MAX_VISIBLE) { queue.push(n); return; }

    var toast = el("div", "bt-toast");
    toast.setAttribute("data-testid", "toast");
    toast.style.setProperty("--bt-toast-ms", TOAST_MS + "ms");
    var iconWrap = el("span", "bt-toast__icon");
    iconWrap.setAttribute("aria-hidden", "true");
    iconWrap.appendChild(icon(n.icon));
    var body = el("div", "bt-toast__body");
    body.appendChild(el("p", "bt-toast__title", n.title));
    if (n.message) body.appendChild(el("p", "bt-toast__message", n.message));
    if (n.open_url && (n.has_destination || n.summary)) {
      var actions = el("div", "bt-toast__actions");
      var view = el("button", null, n.summary ? "Show notifications" : "View");
      view.type = "button";
      view.addEventListener("click", function () {
        if (n.summary) { dismiss(); if (bellButton && window.bootstrap) window.bootstrap.Dropdown.getOrCreateInstance(bellButton).show(); return; }
        var form = postForm(n.open_url);
        document.body.appendChild(form);
        form.submit();
      });
      actions.appendChild(view);
      body.appendChild(actions);
    }
    var close = el("button", "bt-toast__close");
    close.type = "button";
    close.setAttribute("aria-label", "Dismiss: " + n.title);
    close.appendChild(icon("bi-x-lg"));
    var timerBar = el("span", "bt-toast__timer");
    timerBar.setAttribute("aria-hidden", "true");

    toast.appendChild(iconWrap);
    toast.appendChild(body);
    toast.appendChild(close);
    toast.appendChild(timerBar);
    region.appendChild(toast);
    visibleToasts.push(toast);

    var remaining = TOAST_MS;
    var started = Date.now();
    var hideTimer = window.setTimeout(dismiss, remaining);
    var gone = false;

    function pause() {
      if (gone) return;
      window.clearTimeout(hideTimer);
      remaining -= Date.now() - started;
      toast.classList.add("is-paused");
    }
    function resume() {
      if (gone || toast.matches(":hover") || toast.contains(document.activeElement)) return;
      started = Date.now();
      toast.classList.remove("is-paused");
      hideTimer = window.setTimeout(dismiss, Math.max(remaining, 1500));
    }
    function dismiss() {
      if (gone) return;
      gone = true;
      window.clearTimeout(hideTimer);
      // Leaving the toast never marks the notification as read.
      var removeNow = function () {
        toast.remove();
        visibleToasts = visibleToasts.filter(function (t) { return t !== toast; });
        if (queue.length) showToast(queue.shift());
      };
      var reduced = document.documentElement.classList.contains("bt-reduce-motion")
        || (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
      if (reduced) { removeNow(); return; }
      toast.classList.add("is-leaving");
      toast.addEventListener("animationend", removeNow, { once: true });
      window.setTimeout(removeNow, 400); // in case animationend never fires
    }

    toast.btDismiss = dismiss;
    close.addEventListener("click", dismiss);
    toast.addEventListener("mouseenter", pause);
    toast.addEventListener("mouseleave", resume);
    toast.addEventListener("focusin", pause);
    toast.addEventListener("focusout", function () { window.setTimeout(resume, 0); });
    toast.addEventListener("keydown", function (event) { if (event.key === "Escape") dismiss(); });
  }

  function toastNew(items) {
    if (!items.length) return;
    if (items.length > MAX_VISIBLE) {
      items.slice(-(MAX_VISIBLE - 1)).forEach(showToast);
      var more = items.length - (MAX_VISIBLE - 1);
      showToast({ title: more + " more new notifications", message: "Open the bell to see them all.",
                  icon: "bi-bell", summary: true, open_url: "#" });
      return;
    }
    items.forEach(showToast);
  }

  // --- polling -----------------------------------------------------------------------------
  function schedule(ms) {
    window.clearTimeout(timer);
    if (!stopped) timer = window.setTimeout(poll, ms);
  }

  function poll() {
    if (stopped) return;
    if (document.hidden) { schedule(interval); return; }
    fetch(feedUrl + "?since=" + encodeURIComponent(lastSeen), {
      credentials: "same-origin", cache: "no-store", headers: { "Accept": "application/json" },
    }).then(function (response) {
      // Signed out, suspended or session expired: stop quietly; the next navigation shows why.
      if (response.status === 401 || response.status === 403 || response.redirected) { stopped = true; return null; }
      if (!response.ok) throw new Error("HTTP " + response.status);
      return response.json();
    }).then(function (data) {
      if (!data) return;
      interval = POLL_MS;
      setCount(data.unread_count);
      renderPanel(data.items || []);
      var fresh = (data.new || []).filter(function (n) { return n.id > lastSeen && isInternal(n.open_url); });
      lastSeen = Math.max(lastSeen, data.latest_id || 0);
      writeSeen(lastSeen);
      toastNew(fresh);
    }).catch(function () {
      interval = Math.min(interval * 2, MAX_POLL_MS); // back off while the server is unreachable
    }).then(function () { schedule(interval); });
  }

  // Opening the bell shows everything the toasts were previewing, so clear them out of its way.
  root.addEventListener("show.bs.dropdown", function () {
    queue = [];
    visibleToasts.slice().forEach(function (t) { t.btDismiss(); });
  });

  document.addEventListener("visibilitychange", function () {
    if (!document.hidden) { interval = POLL_MS; schedule(0); }
  });

  setCount(Number(countEl.getAttribute("data-value") || countEl.textContent) || 0);
  schedule(0);
})();
