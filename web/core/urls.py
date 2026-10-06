from django.urls import path

from web.core import views

app_name = "core"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("history/", views.history, name="history"),
    path("forecast/", views.forecast, name="forecast"),
    path("forecast/runs/", views.forecast_history, name="forecast_history"),
    path("forecast/runs/<int:pk>/", views.forecast_run, name="forecast_run"),
    path("performance/", views.performance, name="performance"),
]
