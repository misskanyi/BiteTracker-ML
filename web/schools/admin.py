from django.contrib import admin

from web.schools.models import DatasetUpload, ForecastRun, School


@admin.register(School)
class SchoolAdmin(admin.ModelAdmin):
    list_display = ("school_id", "name", "county", "sub_county", "verified_enrolment", "is_active", "updated_at")
    list_filter = ("is_active", "county", "data_source")
    search_fields = ("school_id", "name", "county", "sub_county")
    readonly_fields = ("created_at", "updated_at")


class ReadOnlyRecordAdmin(admin.ModelAdmin):
    """Audit records are created by the application, so admins can inspect and delete but not add or edit them."""

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False


@admin.register(DatasetUpload)
class DatasetUploadAdmin(ReadOnlyRecordAdmin):
    list_display = (
        "original_filename", "uploaded_by", "uploaded_at", "validation_status",
        "forecast_readiness", "quality_score", "n_records", "processing_status",
    )
    list_filter = ("validation_status", "forecast_readiness", "processing_status")
    search_fields = ("original_filename", "uploaded_by__username")
    date_hierarchy = "uploaded_at"


@admin.register(ForecastRun)
class ForecastRunAdmin(ReadOnlyRecordAdmin):
    list_display = (
        "created_at", "school_code", "school_name", "horizon_days", "origin_date",
        "day1_predicted_meals", "day1_recommended_meals", "day1_risk_level", "model_label", "user",
    )
    list_filter = ("horizon_days", "day1_risk_level", "model_label", "results_label")
    search_fields = ("school_code", "school_name", "user__username")
    date_hierarchy = "created_at"
