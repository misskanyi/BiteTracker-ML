/* BiteTracker theme switching.
 *
 * The initial theme is applied by partials/_theme_init.html before first paint.
 * This file wires the header toggle, follows OS changes while no explicit choice
 * is saved, and announces changes with a "bt:themechange" event (charts listen).
 * localStorage key "theme" is shared with Django admin's toggle. For signed-in users the
 * toggle also saves the choice to their account (Settings → Appearance), so it survives
 * logging in again on any device.
 */
(function () {
  "use strict";
  var root = document.documentElement;
  var STORAGE_KEY = "theme";
  var media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

  function stored() {
    try { return localStorage.getItem(STORAGE_KEY); } catch (e) { return null; }
  }

  function current() {
    return root.getAttribute("data-theme") === "dark" ? "dark" : "light";
  }

  function syncToggle() {
    var button = document.getElementById("bt-theme-toggle");
    if (!button) return;
    var dark = current() === "dark";
    button.setAttribute("aria-pressed", dark ? "true" : "false");
    button.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
    button.setAttribute("title", dark ? "Switch to light theme" : "Switch to dark theme");
  }

  var fadeTimer = null;

  // Cross-fade colours for one theme change only; a permanent transition on every element would slow the app.
  function crossFade() {
    root.classList.add("bt-theme-transition");
    window.clearTimeout(fadeTimer);
    fadeTimer = window.setTimeout(function () { root.classList.remove("bt-theme-transition"); }, 320);
  }

  function apply(theme) {
    crossFade();
    root.setAttribute("data-theme", theme);
    root.setAttribute("data-bs-theme", theme);
    syncToggle();
    document.dispatchEvent(new CustomEvent("bt:themechange", { detail: { theme: theme } }));
  }

  function csrfToken() {
    var match = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
    return match ? decodeURIComponent(match[1]) : "";
  }

  // Best effort: the page has already switched; a failed save only means the next login uses the old theme.
  function saveToAccount(theme) {
    var meta = document.querySelector('meta[name="bt-theme-url"]');
    if (!meta || !window.fetch) return;
    root.setAttribute("data-theme-pref", theme);
    fetch(meta.content, {
      method: "POST", credentials: "same-origin",
      headers: { "X-CSRFToken": csrfToken(), "Content-Type": "application/x-www-form-urlencoded" },
      body: "theme=" + encodeURIComponent(theme)
    }).catch(function () {});
  }

  document.addEventListener("DOMContentLoaded", function () {
    syncToggle();
    var button = document.getElementById("bt-theme-toggle");
    if (button) {
      button.addEventListener("click", function () {
        var next = current() === "dark" ? "light" : "dark";
        try { localStorage.setItem(STORAGE_KEY, next); } catch (e) { /* still switch for this page */ }
        apply(next);
        saveToAccount(next);
      });
    }
  });

  if (media) {
    var onChange = function (event) {
      var saved = stored();
      if (saved !== "light" && saved !== "dark") apply(event.matches ? "dark" : "light");
    };
    if (media.addEventListener) media.addEventListener("change", onChange);
    else if (media.addListener) media.addListener(onChange);
  }

  // Another tab (or Django admin) changed the theme.
  window.addEventListener("storage", function (event) {
    if (event.key !== STORAGE_KEY) return;
    var value = event.newValue;
    if (value === "light" || value === "dark") apply(value);
    else if (media) apply(media.matches ? "dark" : "light");
  });

  window.BiteTrackerTheme = { current: current };
})();
