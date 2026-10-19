"""UI shell: theme bootstrapping, sidebar navigation and permissions, header controls, chart colour roles.

What this file checks:
- The theme is applied by an inline script before any stylesheet loads (no flash of the wrong theme).
- Sidebar links follow permissions: Data upload needs add_datasetupload, Schools and Django admin need staff.
- The active page is marked with aria-current, and logout stays a CSRF-protected POST form.
- Anonymous pages (login) get no application shell but keep the data-status banner.
- Every chart trace carries a theme colour role and the chart data itself is unchanged.
"""

from __future__ import annotations

import base64
import re

import numpy as np
import pandas as pd
import pytest
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
from django.contrib.staticfiles import finders
from django.urls import reverse

from web.core import charts

PASSWORD = "Str0ng-test-passw0rd!"


@pytest.fixture
def planner(db):
    return get_user_model().objects.create_user(username="planner", password=PASSWORD)


@pytest.fixture
def staff(db):
    user = get_user_model().objects.create_user(username="staffer", password=PASSWORD, is_staff=True)
    user.user_permissions.add(Permission.objects.get(codename="view_school"))
    return user


def _page(client, user, name="core:forecast_history") -> str:
    client.force_login(user)
    response = client.get(reverse(name))
    assert response.status_code == 200
    return response.content.decode()


def _sidebar(html: str) -> str:
    match = re.search(r'<aside class="bt-sidebar.*?</aside>', html, re.S)
    assert match, "sidebar not rendered"
    return match.group(0)


# --- Theme ---------------------------------------------------------------------------------


def test_theme_is_set_before_stylesheets_load(client, planner):
    html = _page(client, planner)
    head = html.split("</head>")[0]
    assert head.index('localStorage.getItem("theme")') < head.index("bootstrap.min.css")
    assert head.index('localStorage.getItem("theme")') < head.index("css/bitetracker.css")


def test_no_template_comment_leaks_into_page(client, planner):
    assert "{#" not in _page(client, planner)
    assert "{#" not in client.get(reverse("accounts:login")).content.decode()


def test_theme_toggle_is_an_accessible_button(client, planner):
    html = _page(client, planner)
    assert re.search(r'<button type="button" class="bt-icon-btn[^"]*" id="bt-theme-toggle" aria-label="[^"]+" aria-pressed=', html)


def test_shell_uses_shared_glass_components(client, planner):
    html = _page(client, planner)
    assert '<aside class="bt-sidebar bt-glass-panel' in html
    assert '<header class="bt-topbar bt-glass-nav">' in html


def test_theme_init_also_runs_on_login_page(client, db):
    html = client.get(reverse("accounts:login")).content.decode()
    assert 'localStorage.getItem("theme")' in html


# --- Sidebar and permissions ---------------------------------------------------------------


def test_planner_sees_main_navigation_only(client, planner):
    sidebar = _sidebar(_page(client, planner))
    for url in ("core:dashboard", "core:history", "core:forecast", "core:forecast_history", "core:performance"):
        assert f'href="{reverse(url)}"' in sidebar
    assert reverse("schools:upload") not in sidebar
    assert reverse("admin:index") not in sidebar
    assert reverse("admin:schools_school_changelist") not in sidebar


def test_upload_link_requires_upload_permission(client, planner):
    planner.user_permissions.add(Permission.objects.get(codename="add_datasetupload"))
    sidebar = _sidebar(_page(client, planner))
    assert f'href="{reverse("schools:upload")}"' in sidebar
    assert reverse("admin:index") not in sidebar


def test_staff_sees_schools_and_admin_links(client, staff):
    sidebar = _sidebar(_page(client, staff))
    assert f'href="{reverse("admin:schools_school_changelist")}"' in sidebar
    assert f'href="{reverse("admin:index")}"' in sidebar


def test_staff_without_school_permission_gets_no_schools_link(client, db):
    user = get_user_model().objects.create_user(username="helper", password=PASSWORD, is_staff=True)
    sidebar = _sidebar(_page(client, user))
    assert reverse("admin:schools_school_changelist") not in sidebar
    assert f'href="{reverse("admin:index")}"' in sidebar


@pytest.mark.parametrize("url_name, label", [
    ("core:forecast_history", "Forecast history"),
    ("core:dashboard", "Dashboard"),
])
def test_active_page_is_marked(client, planner, url_name, label):
    sidebar = _sidebar(_page(client, planner, url_name))
    active = re.findall(r'<a class="bt-nav__link is-active" href="([^"]+)"\s+aria-current="page"', sidebar)
    assert active == [reverse(url_name)]
    assert f'aria-current="page">{label}</li>' in _page(client, planner, url_name)  # breadcrumb


def test_sidebar_collapse_control_and_mobile_drawer(client, planner):
    html = _page(client, planner)
    assert 'id="bt-collapse-toggle"' in html and 'aria-controls="bt-sidebar"' in html
    assert 'data-bs-toggle="offcanvas" data-bs-target="#bt-sidebar"' in html
    assert 'aria-label="Open navigation"' in html


def test_logout_controls_are_csrf_protected_post_forms(client, planner):
    html = _page(client, planner)
    forms = re.findall(r'<form method="post" action="%s">(.*?)</form>' % re.escape(reverse("accounts:logout")), html, re.S)
    assert len(forms) == 2  # sidebar footer and account menu
    assert all("csrfmiddlewaretoken" in form for form in forms)
    assert "Signed in as planner" in html


def test_login_page_has_no_app_shell(client, db):
    html = client.get(reverse("accounts:login")).content.decode()
    assert 'class="bt-sidebar' not in html and 'id="bt-theme-toggle"' not in html
    assert 'data-testid="synthetic-banner"' in html


def test_font_is_self_hosted(client, planner):
    html = _page(client, planner)
    assert "fonts.googleapis.com" not in html and "fonts.gstatic.com" not in html
    assert "css/fonts.css" in html and 'rel="preload"' in html


def test_admin_uses_the_same_font(client, staff):
    client.force_login(staff)
    html = client.get(reverse("admin:index")).content.decode()
    assert "css/fonts.css" in html and '--font-family-primary: "Outfit"' in html
    assert 'id="site-name"' in html  # stock admin branding block still rendered


def test_admin_matches_app_theme(client, staff):
    """Admin shares tokens, glass skin and the app's light/dark toggle instead of Django's own."""
    client.force_login(staff)
    html = client.get(reverse("admin:schools_school_changelist")).content.decode()
    assert "css/tokens.css" in html and "css/admin.css" in html
    assert html.index('localStorage.getItem("theme")') < html.index("css/admin.css")  # no theme flash
    assert "admin/css/dark_mode.css" not in html and "admin/js/theme.js" not in html
    assert 'id="bt-theme-toggle"' in html and "js/theme.js" in html
    assert "BiteTracker administration" in html
    assert f'href="{reverse("core:dashboard")}"' in html and "Back to BiteTracker" in html


def test_admin_login_matches_app_theme(client, db):
    html = client.get(reverse("admin:login")).content.decode()
    assert "css/admin.css" in html and 'id="bt-theme-toggle"' in html


@pytest.mark.parametrize("path", [
    "css/bitetracker.css", "css/tokens.css", "css/admin.css", "css/fonts.css", "js/theme.js", "js/layout.js", "js/charts.js",
    "fonts/outfit/outfit-latin-wght-normal.woff2", "fonts/outfit/outfit-latin-ext-wght-normal.woff2", "fonts/outfit/OFL.txt",
])
def test_static_assets_exist(path):
    assert finders.find(path)


# --- Chart colour roles --------------------------------------------------------------------


def _history(n_schools: int = 1) -> pd.DataFrame:
    rows = []
    for s in range(n_schools):
        for d, meals in enumerate([100.0, 120.5, 90.0]):
            rows.append({"date": pd.Timestamp("2025-03-03") + pd.Timedelta(days=d), "school_name": f"School {s}",
                         "meals_served": meals + s, "students_present": 110 + s})
    return pd.DataFrame(rows)


def test_every_chart_trace_has_a_known_colour_role():
    perf = {
        "selected_model": "Random Forest", "test": {"mae": 47.8, "rmse": 87.0},
        "naive_baselines": [{"name": "Persistence", "test_mae": 40.6, "test_rmse": 74.0}],
    }
    specs = [
        charts.school_history_chart(_history(), "School 0"),
        charts.all_schools_history_chart(_history(3)),
        charts.forecast_chart(_history(), [{"date": "2025-03-06", "predicted_meals": 101, "recommended_meals": 112}], "School 0"),
        charts.model_vs_baselines_chart(perf),
        charts.validation_ranking_chart([{"Model": "Random Forest", "Validation MAE": 78.4, "Selected": True},
                                         {"Model": "Decision Tree", "Validation MAE": 80.3}]),
    ]
    for spec in specs:
        for trace in spec["data"]:
            roles = trace["meta"] if isinstance(trace["meta"], list) else [trace["meta"]]
            assert all(role in charts.ROLE_COLOURS for role in roles), trace["meta"]


def test_school_colours_follow_the_school_not_the_rank():
    spec = charts.all_schools_history_chart(_history(7))
    roles = [t["meta"] for t in spec["data"]]
    assert roles[:5] == charts.SCHOOL_ROLES
    assert roles[5:] == ["other", "other"]  # never a reused hue


def _values(array) -> list:
    """Plotly 6 serialises numeric arrays as {"dtype", "bdata"} (base64); plain lists pass through."""
    if isinstance(array, dict):
        return np.frombuffer(base64.b64decode(array["bdata"]), dtype=array["dtype"]).tolist()
    return list(array)


def test_chart_restyle_keeps_data_values():
    spec = charts.school_history_chart(_history(), "School 0")
    assert _values(spec["data"][0]["y"]) == [100.0, 120.5, 90.0]  # capped/fractional values are plotted unchanged


def test_baseline_chart_still_shows_baseline_beating_model_on_test_mae():
    perf = {
        "selected_model": "Random Forest", "test": {"mae": 47.8, "rmse": 87.0},
        "naive_baselines": [{"name": "Persistence", "test_mae": 40.6, "test_rmse": 74.0}],
    }
    mae = charts.model_vs_baselines_chart(perf)["data"][0]
    assert mae["meta"] == "mae" and _values(mae["y"]) == [47.8, 40.6]
