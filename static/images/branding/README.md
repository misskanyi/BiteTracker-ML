# BiteTracker brand assets

The BiteTracker mark combines three student silhouettes (the children school feeding serves), a spoon
(meals and nutrition) and an open book (education). The bottom of the mark is always an open book with
straight page edges, page lines and a central spine; it must never be redrawn as a leaf.

These files are copied unchanged from `BiteTracker_Brand_Assets.zip`. Edit them in a vector editor and
replace the files here; do not hand-edit the SVG paths. Use the SVG files on the web; the PNG files are
raster copies of the same designs for places that cannot use SVG.

Every colour in the marks is a BiteTracker token from `static/css/tokens.css`:

| Part of the mark | Light version | Dark version |
|---|---|---|
| "Tracker", spoon, rays | `#7C3AED` (`--bt-primary`) | `#A78BFA` (dark `--bt-primary`) |
| Students, outlines | `#5B21B6` | `#C4B5FD` |
| Book pages | `#C4B5FD` | `#A78BFA` |
| "Bite" | `#1F1B2D` (`--bt-text`) | `#F5F3FF` (dark `--bt-text`) |
| Tagline | `#5F576E` (`--bt-text-muted`) | `#B8B0C9` (dark `--bt-text-muted`) |
| App tile background | `#EDE9FE` (`--bt-primary-soft`) | `#241E35` (dark `--bt-surface`) |

## Folders

| Folder | Files | Used for |
|---|---|---|
| `logos/` | `logo-horizontal-light/-dark` | Expanded sidebar |
| | `logo-stacked-light/-dark` | Login page |
| | `wordmark-light/-dark`, `wordmark-tagline-light/-dark` | Not used yet; narrow headers without the symbol |
| `icons/` | `icon-light/-dark` | Collapsed sidebar, mobile top bar, Django admin header, loaders |
| | `app-light/-dark` | `app-light.png` is the Apple touch icon; both are app-tile designs |
| | `icon-white` | Not used; identical to `icon-dark` (see below) |
| `email/` | `logo-icon-96.png` | Account-email header, attached inline (Content-ID). A 96×96 PNG resize of `icons/icon-light.png`, shown at 48×48; regenerate it from that file if the icon changes |
| `favicons/` | `favicon-simple` | Browser tab (SVG, with the 256px PNG as fallback) |
| | `favicon` | Not used; the full mark is too detailed at 16–32px |

## Using a mark in a template

Always go through `templates/partials/_brand.html`. It renders the light and dark files together and CSS
(`.bt-brand-mark` in `bitetracker.css` and `admin.css`) shows the one matching `html[data-theme]`, so the
mark follows the theme toggle without a flash or JavaScript.

```django
{% include "partials/_brand.html" with variant="horizontal" %}  {# or "stacked", "icon" #}
{% include "partials/_brand.html" with variant="icon" decorative=True %}  {# alt="" next to a visible label #}
```

Size marks by width only (`.bt-brand-mark--horizontal`, `--stacked`, `--icon`); height follows each file's
aspect ratio, so marks are never stretched or cropped. Use the full logo once per screen (the sidebar);
do not repeat it in cards or page sections.

## Loading indicators

`templates/partials/_loader.html` is the only branded loader. The mark never rotates.

- `variant="pulse"`: the icon gently pulses in opacity and scale. Used while a slow page is being prepared.
- `variant="ring"`: the icon stays still inside a rotating violet ring. Used for forecast generation and
  CSV upload and validation. It is indeterminate because the server does not report progress; never show
  an invented percentage.

Loaders appear only after a short delay, are removed when the page changes or the user goes back, and
stop moving under `prefers-reduced-motion: reduce`. Short actions (logging in) use the small spinner inside
the button instead. Charts keep their existing shimmer placeholder.

## Known issues in the source files

- The wordmark text is live SVG text in Arial. Inside an `<img>` the page font (Outfit) is not available,
  so "BiteTracker" may render slightly differently across platforms. Convert the text to outlines for exact
  fidelity.
- `wordmark-tagline-light/-dark.svg` set no font on the tagline, so some browsers may use a serif font.
- `icon-white.svg` is not white; it is an exact copy of `icon-dark.svg`.
- `favicon.svg` is an exact copy of `app-light.svg`.
- The PNG previews of the wide logos sit on square canvases with large empty margins.
- At sidebar size the horizontal logo's tagline is very small and decorative only.
