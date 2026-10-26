/* BiteTracker browser-console logging, for debugging.
 *
 * window.BTLog.success / error / warn / info write "[BiteTracker]" lines to the browser console.
 * On by default when Django DEBUG is true (base.html adds <meta name="bt-debug" content="1">).
 * In any build, BTLog.enable() / BTLog.disable() override that for this browser (localStorage
 * "bt-debug"); reload to apply. When off, every call is a no-op, so callers never check first.
 *
 * Loaded in <head>, before other scripts, so it also catches errors they throw. On its own it reports:
 * page load (status and time), Django messages, form errors, form submits, uncaught errors,
 * unhandled promise rejections and files (scripts, styles, images) that fail to load.
 * Form values are never logged: they can hold passwords.
 */
(function () {
  "use strict";
  var STORAGE_KEY = "bt-debug";
  var PREFIX = "[BiteTracker]";
  var LEVELS = {
    success: { method: "log", label: "✔", colour: "#15803d" },
    info: { method: "info", label: "ℹ", colour: "#2563eb" },
    warn: { method: "warn", label: "⚠", colour: "#b45309" },
    error: { method: "error", label: "✖", colour: "#dc2626" },
  };

  function isEnabled() {
    var saved = null;
    try { saved = localStorage.getItem(STORAGE_KEY); } catch (e) { /* storage blocked: use the page default */ }
    if (saved === "1" || saved === "0") return saved === "1";
    var meta = document.querySelector('meta[name="bt-debug"]');
    return !!meta && meta.content === "1";
  }

  var enabled = isEnabled();

  function logger(level) {
    var style = LEVELS[level];
    return function (message) {
      if (!enabled || !window.console) return;
      // Extra arguments (objects, errors) are passed through, so the console keeps them expandable.
      var args = ["%c" + PREFIX + " " + style.label + "%c " + message, "color:" + style.colour + ";font-weight:600", ""];
      console[style.method].apply(console, args.concat(Array.prototype.slice.call(arguments, 1)));
    };
  }

  function setOverride(value) {
    try { localStorage.setItem(STORAGE_KEY, value); } catch (e) { /* this page view only */ }
    enabled = value === "1";
    if (window.console) console.info(PREFIX + " console logging " + (enabled ? "on" : "off") + ". Reload to apply everywhere.");
  }

  var log = {
    success: logger("success"),
    info: logger("info"),
    warn: logger("warn"),
    error: logger("error"),
    enable: function () { setOverride("1"); },
    disable: function () { setOverride("0"); },
  };
  log.isEnabled = function () { return enabled; };
  window.BTLog = log;
  if (!enabled) return;

  // --- Uncaught errors ---------------------------------------------------------------
  // Capture phase, so failed <script>, <link> and <img> loads (which don't bubble) are seen too.
  window.addEventListener("error", function (event) {
    var target = event.target;
    if (target && target !== window && (target.src || target.href)) {
      log.error("Failed to load " + target.tagName.toLowerCase() + ": " + (target.src || target.href));
      return;
    }
    var where = event.filename ? " (" + event.filename + ":" + event.lineno + ":" + event.colno + ")" : "";
    log.error("Uncaught error: " + event.message + where, event.error || "");
  }, true);

  window.addEventListener("unhandledrejection", function (event) {
    log.error("Unhandled promise rejection:", event.reason);
  });

  // --- What the server sent back ---------------------------------------------------------
  var MESSAGE_LEVELS = { success: "success", error: "error", warning: "warn", info: "info", debug: "info" };

  function fieldName(feedback) {
    if (feedback.id && /_error$/.test(feedback.id)) return feedback.id.replace(/_error$/, "").replace(/^id_/, "");
    var field = feedback.parentElement && feedback.parentElement.querySelector("input[name], select[name], textarea[name]");
    return field ? field.name : "form";
  }

  document.addEventListener("DOMContentLoaded", function () {
    // Django messages (partials/_messages.html).
    document.querySelectorAll("[data-bt-message]").forEach(function (el) {
      log[MESSAGE_LEVELS[el.dataset.btMessage] || "info"]("Message: " + el.textContent.trim());
    });
    // Form errors: per-field feedback, and form-wide errors shown as danger alerts.
    document.querySelectorAll(".invalid-feedback").forEach(function (el) {
      var text = el.textContent.trim();
      if (text) log.error("Form error on \"" + fieldName(el) + "\": " + text);
    });
    document.querySelectorAll(".alert-danger[role=alert]:not([data-bt-message])").forEach(function (el) {
      log.error("Form error: " + el.textContent.trim().replace(/\s+/g, " "));
    });
  });

  // Form submits: method and destination only.
  document.addEventListener("submit", function (event) {
    var form = event.target;
    var method = (form.getAttribute("method") || "get").toUpperCase();
    log.info("Submitting form: " + method + " " + (form.getAttribute("action") || location.pathname));
  }, true);

  window.addEventListener("load", function () {
    var nav = window.performance && performance.getEntriesByType ? performance.getEntriesByType("navigation")[0] : null;
    var status = nav && nav.responseStatus;  // not reported by every browser
    var detail = [status ? "HTTP " + status : "", nav ? Math.round(nav.duration) + " ms" : ""].filter(Boolean).join(", ");
    var message = "Page loaded: " + location.pathname + location.search + (detail ? " (" + detail + ")" : "");
    if (status >= 400) log.error(message);
    else log.success(message);
  });
})();
