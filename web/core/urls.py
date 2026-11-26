from django.urls import path

from web.core import actual_views, views

app_name = "core"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("history/", views.history, name="history"),
    path("forecast/", views.forecast, name="forecast"),
    path("forecast/runs/", views.forecast_history, name="forecast_history"),
    path("forecast/runs/<int:pk>/", views.forecast_run, name="forecast_run"),
    path("forecast/runs/<int:pk>/actuals/", actual_views.record_actual, name="record_actual"),
    path("actuals/", actual_views.actuals_overview, name="actuals"),
    path("actuals/export.csv", actual_views.followup_export, name="followup_export"),
    path("actuals/<int:pk>/", actual_views.actual_detail, name="actual_detail"),
    path("actuals/<int:pk>/review/", actual_views.actual_review, name="actual_review"),
    path("performance/", views.performance, name="performance"),
    path("donors/", views.donors, name="donors"),
    path("privacy/", views.privacy, name="privacy"),
]
