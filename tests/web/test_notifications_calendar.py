"""In-app notifications and the School Calendar.

What this file checks:
- Real events create notifications: forecast success and failure, dataset upload
  success and validation failure, calendar changes.
- Preferences switch optional notifications off; security notices always arrive.
- Unread counts, mark-as-read, mark-all, opening a notification; nobody can read or
  change another user's notifications; duplicate submissions do not double-notify.
- Calendar: month and list views, create/edit/delete, date validation, overlap
  confirmation, school filtering, scope enforcement, notifications and audit logging.
"""

from __future__ import annotations

import io
from datetime import date

import pandas as pd
import pytest
from django.contrib.auth.models import Permission
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from services.model_service import ModelNotFoundError
from services import forecast_service
from tests.support.web_fixtures import operator
from web.accounts.models import UserProfile
from web.audit.models import AuditLog
from web.notifications.models import Notification
from web.notifications.services import notify
from web.schoolcalendar.models import CalendarEvent
from web.schools.models import School

pytestmark = pytest.mark.django_db
Type = Notification.Type


@pytest.fixture
def media_root(settings, tmp_path):
    settings.MEDIA_ROOT = tmp_path / "uploads"
    return settings.MEDIA_ROOT


@pytest.fixture
def amina(client):
    user = operator("amina")
    client.force_login(user)
    return user


def _event_data(school_obj: School, **overrides) -> dict:
    data = {"title": "Mid-term break", "event_type": "mid_term_break", "school": school_obj.pk,
            "start_date": "2025-02-26", "end_date": "2025-03-02", "feeding_status": "no_meals", "description": ""}
    data.update(overrides)
    return data


# --- Notification triggers ----------------------------------------------------------------------


def test_successful_forecast_notifies_the_user(client, amina, stub_artifacts):
    client.post(reverse("core:forecast"), {"school": "SCH001", "horizon": 1})
    n = Notification.objects.get(recipient=amina)
    assert n.type == Type.FORECAST_COMPLETED and n.url.startswith("/forecast/runs/")
    assert n.school.school_id == "SCH001"


def test_failed_forecast_notifies_the_user(client, amina, stub_artifacts, monkeypatch):
    def fail(*args, **kwargs):
        raise ModelNotFoundError("missing")
    monkeypatch.setattr(forecast_service, "generate_forecast", fail)
    client.post(reverse("core:forecast"), {"school": "SCH001", "horizon": 1})
    assert Notification.objects.get(recipient=amina).type == Type.FORECAST_FAILED


def test_forecast_notification_respects_preference(client, amina, stub_artifacts):
    UserProfile.objects.filter(user=amina).update(notify_forecast=False)
    client.post(reverse("core:forecast"), {"school": "SCH001", "horizon": 1})
    assert not Notification.objects.exists()


def _upload(client, content: bytes, name="data.csv"):
    return client.post(reverse("schools:upload"), {"file": SimpleUploadedFile(name, content, content_type="text/csv"),
                                                   "letter": SimpleUploadedFile("letter.pdf", b"%PDF-1.4\n")})


@pytest.fixture
def uploader(client, amina):
    amina.user_permissions.add(Permission.objects.get(codename="add_datasetupload"),
                               Permission.objects.get(codename="view_datasetupload"))
    return amina


def test_valid_upload_notifies_and_is_audited(client, uploader, media_root, stub_processed_frame):
    buffer = io.StringIO()
    stub_processed_frame[["date", "school_id", "school_name", "meals_served"]].to_csv(buffer, index=False)
    _upload(client, buffer.getvalue().encode())
    assert Notification.objects.get(recipient=uploader).type == Type.DATASET_UPLOADED
    assert AuditLog.objects.get(action=AuditLog.Action.DATASET_UPLOADED).details["status"] == "valid"


def test_invalid_upload_notifies_validation_errors(client, uploader, media_root):
    _upload(client, pd.DataFrame({"date": ["2024-01-01"]}).to_csv(index=False).encode())
    n = Notification.objects.get(recipient=uploader)
    assert n.type == Type.DATASET_INVALID and "problem" in n.message


def test_invalid_upload_notification_respects_preference(client, uploader, media_root):
    UserProfile.objects.filter(user=uploader).update(notify_dataset_validation=False)
    _upload(client, b"date\n2024-01-01\n")
    assert not Notification.objects.exists()


def test_security_notices_ignore_preferences(db):
    user = operator("quiet")
    profile = user.profile
    profile.notify_forecast = profile.notify_calendar = profile.notify_announcements = False
    profile.save()
    assert notify([user], Type.SECURITY, "Password changed")
    assert not notify([user], Type.ANNOUNCEMENT, "News")


def test_duplicate_notifications_are_not_created(db):
    user = operator("dup")
    notify([user, user], Type.ANNOUNCEMENT, "Same", url="/x/")
    notify([user], Type.ANNOUNCEMENT, "Same", url="/x/")
    assert Notification.objects.count() == 1


def test_school_notifications_skip_users_without_access(db):
    insider, outsider = operator("insider", codes=("SCH001",)), operator("outsider", codes=("SCH002",))
    notify([insider, outsider], Type.CALENDAR, "Event", school=School.objects.get(school_id="SCH001"))
    assert list(Notification.objects.values_list("recipient__username", flat=True)) == ["insider"]


# --- Bell, read state and isolation ---------------------------------------------------------------


def test_bell_shows_unread_count_and_empty_state(client, amina):
    html = client.get(reverse("accounts:profile")).content.decode()
    assert 'data-testid="notification-bell"' in html
    assert 'data-testid="unread-count" data-notify-count hidden>0</span>' in html  # nothing unread: badge hidden
    assert "You're all caught up" in html
    notify([amina], Type.ANNOUNCEMENT, "One")
    notify([amina], Type.ANNOUNCEMENT, "Two")
    html = client.get(reverse("accounts:profile")).content.decode()
    assert '<span class="bt-bell__count" data-testid="unread-count" data-notify-count>2</span>' in html


def test_mark_read_mark_all_and_open(client, amina):
    first, second = (notify([amina], Type.ANNOUNCEMENT, title, url="/calendar/")[0] for title in ("A", "B"))
    client.post(reverse("notifications:mark_read", args=[first.pk]))
    first.refresh_from_db()
    assert first.is_read and first.read_at
    response = client.post(reverse("notifications:open", args=[second.pk]))
    assert response.url == "/calendar/"
    notify([amina], Type.ANNOUNCEMENT, "C")
    client.post(reverse("notifications:mark_all_read"))
    assert not Notification.objects.filter(recipient=amina, is_read=False).exists()


def test_open_never_redirects_off_site(client, amina):
    n = Notification.objects.create(recipient=amina, type=Type.ANNOUNCEMENT, title="x", url="https://evil.example/")
    assert client.post(reverse("notifications:open", args=[n.pk])).url == reverse("notifications:list")


def test_users_cannot_touch_other_users_notifications(client, amina):
    other = operator("other")
    theirs = notify([other], Type.ANNOUNCEMENT, "Private")[0]
    assert client.post(reverse("notifications:mark_read", args=[theirs.pk])).status_code == 404
    assert client.post(reverse("notifications:open", args=[theirs.pk])).status_code == 404
    assert "Private" not in client.get(reverse("notifications:list")).content.decode()
    theirs.refresh_from_db()
    assert not theirs.is_read


def test_notification_actions_require_post(client, amina):
    n = notify([amina], Type.ANNOUNCEMENT, "x")[0]
    assert client.get(reverse("notifications:mark_read", args=[n.pk])).status_code == 405


# --- Calendar ----------------------------------------------------------------------------------------


@pytest.fixture
def sch1(db):
    return School.objects.get_or_create(school_id="SCH001", defaults={"name": "Greenfield"})[0]


def test_create_event_notifies_colleagues_not_actor_and_audits(client, amina, sch1):
    colleague, stranger = operator("colleague", codes=("SCH001",)), operator("stranger", codes=("SCH002",))
    response = client.post(reverse("schoolcalendar:event_create"), _event_data(sch1))
    assert response.status_code == 302
    event = CalendarEvent.objects.get()
    assert event.created_by == amina and event.updated_by == amina
    recipients = set(Notification.objects.values_list("recipient__username", flat=True))
    assert recipients == {"colleague"}
    assert AuditLog.objects.get(action=AuditLog.Action.CALENDAR_CREATED).school == sch1
    assert stranger.pk  # assigned elsewhere: no notification


def test_calendar_notifications_respect_preference(client, amina, sch1):
    colleague = operator("colleague", codes=("SCH001",))
    UserProfile.objects.filter(user=colleague).update(notify_calendar=False)
    client.post(reverse("schoolcalendar:event_create"), _event_data(sch1))
    assert not Notification.objects.exists()


def test_end_date_before_start_is_rejected(client, amina, sch1):
    response = client.post(reverse("schoolcalendar:event_create"), _event_data(sch1, end_date="2025-02-01"))
    assert response.status_code == 200 and "cannot be before the start date" in response.content.decode()
    assert not CalendarEvent.objects.exists()


def test_required_fields_are_validated(client, amina, sch1):
    response = client.post(reverse("schoolcalendar:event_create"), _event_data(sch1, title="", school=""))
    assert response.status_code == 200 and not CalendarEvent.objects.exists()


def test_operator_cannot_create_event_for_unassigned_school(client, db):
    other_school = School.objects.create(school_id="SCH777")
    client.force_login(operator("narrow", codes=("SCH001",)))
    response = client.post(reverse("schoolcalendar:event_create"), _event_data(other_school))
    assert response.status_code == 200 and "Select a valid choice" in response.content.decode()
    assert not CalendarEvent.objects.exists()


def test_operator_cannot_create_programme_wide_event(client, amina, sch1):
    client.post(reverse("schoolcalendar:event_create"), _event_data(sch1, school=""))
    assert not CalendarEvent.objects.exists()


def test_overlap_with_different_feeding_status_needs_confirmation(client, amina, sch1):
    CalendarEvent.objects.create(title="Exams", event_type="examination", school=sch1, start_date=date(2025, 3, 1),
                                 end_date=date(2025, 3, 5), feeding_status="meals_expected")
    response = client.post(reverse("schoolcalendar:event_create"), _event_data(sch1))
    html = response.content.decode()
    assert response.status_code == 200 and "overlap events with a different feeding status" in html
    assert 'name="confirm_overlap"' in html
    response = client.post(reverse("schoolcalendar:event_create"), _event_data(sch1, confirm_overlap="on"))
    assert response.status_code == 302 and CalendarEvent.objects.count() == 2


def test_edit_and_delete_event_with_audit(client, amina, sch1):
    client.post(reverse("schoolcalendar:event_create"), _event_data(sch1))
    event = CalendarEvent.objects.get()
    client.post(reverse("schoolcalendar:event_update", args=[event.pk]), _event_data(sch1, title="Half term"))
    event.refresh_from_db()
    assert event.title == "Half term"
    log = AuditLog.objects.get(action=AuditLog.Action.CALENDAR_UPDATED)
    assert log.details["changes"]["title"] == {"from": "Mid-term break", "to": "Half term"}
    assert client.get(reverse("schoolcalendar:event_delete", args=[event.pk])).status_code == 200  # confirmation page
    assert CalendarEvent.objects.exists()
    client.post(reverse("schoolcalendar:event_delete", args=[event.pk]))
    assert not CalendarEvent.objects.exists()
    assert AuditLog.objects.filter(action=AuditLog.Action.CALENDAR_DELETED).exists()


def test_events_of_other_schools_are_invisible_and_uneditable(client, db, sch1):
    hidden = CalendarEvent.objects.create(title="Secret", event_type="other", school=sch1,
                                          start_date=date(2025, 1, 6), end_date=date(2025, 1, 6))
    client.force_login(operator("narrow", codes=("SCH002",)))
    for name in ("event_detail", "event_update", "event_delete"):
        assert client.get(reverse(f"schoolcalendar:{name}", args=[hidden.pk])).status_code == 404
    assert client.post(reverse("schoolcalendar:event_delete", args=[hidden.pk])).status_code == 404
    html = client.get(reverse("schoolcalendar:calendar") + "?view=list").content.decode()
    assert "Secret" not in html
    assert client.get(reverse("schoolcalendar:calendar"), {"school": sch1.pk}).status_code == 403


def test_programme_wide_events_are_visible_but_read_only_to_operators(client, amina):
    event = CalendarEvent.objects.create(title="Madaraka Day", event_type="public_holiday", school=None,
                                         start_date=date(2025, 6, 1), end_date=date(2025, 6, 1), feeding_status="no_meals")
    assert client.get(reverse("schoolcalendar:event_detail", args=[event.pk])).status_code == 200
    assert client.get(reverse("schoolcalendar:event_update", args=[event.pk])).status_code == 403


def test_month_view_navigation_and_list_filters(client, amina, sch1):
    sch2 = School.objects.get(school_id="SCH002")
    CalendarEvent.objects.create(title="Opening", event_type="term_opening", school=sch1, start_date=date(2025, 1, 6),
                                 end_date=date(2025, 1, 6), feeding_status="meals_expected")
    CalendarEvent.objects.create(title="Sports day", event_type="special_event", school=sch2, start_date=date(2025, 1, 20),
                                 end_date=date(2025, 1, 20))
    CalendarEvent.objects.create(title="Old exams", event_type="examination", school=sch1, start_date=date(2024, 11, 4),
                                 end_date=date(2024, 11, 8))
    html = client.get(reverse("schoolcalendar:calendar"), {"month": "2025-01"}).content.decode()
    assert 'data-testid="month-grid"' in html and "January 2025" in html
    assert "Opening" in html and "Sports day" in html and "month=2024-12" in html and "month=2025-02" in html
    html = client.get(reverse("schoolcalendar:calendar"), {"month": "2025-01", "school": sch1.pk}).content.decode()
    assert "Opening" in html and "Sports day" not in html
    html = client.get(reverse("schoolcalendar:calendar"), {"view": "list", "year": 2024}).content.decode()
    assert "Old exams" in html and "Opening" not in html
    html = client.get(reverse("schoolcalendar:calendar"), {"view": "list", "type": "term_opening"}).content.decode()
    assert "Opening" in html and "Old exams" not in html


def test_calendar_is_not_a_model_input():
    """The forecasting code must not read calendar events (they are planning references only)."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    for folder in ("src", "services"):
        for path in (root / folder).rglob("*.py"):
            assert "CalendarEvent" not in path.read_text() and "schoolcalendar" not in path.read_text(), path


# --- Removing notifications -------------------------------------------------------------------------


def test_delete_removes_one_notification(client, amina):
    keep, remove = (notify([amina], Type.ANNOUNCEMENT, title)[0] for title in ("Keep", "Remove"))
    response = client.post(reverse("notifications:delete", args=[remove.pk]), {"next": reverse("notifications:list")})
    assert response.url == reverse("notifications:list")
    assert list(Notification.objects.filter(recipient=amina)) == [keep]


def test_clear_all_removes_read_and_unread(client, amina):
    for title in ("A", "B", "C"):
        notify([amina], Type.ANNOUNCEMENT, title)
    Notification.objects.filter(recipient=amina, title="A").update(is_read=True)
    response = client.post(reverse("notifications:clear_all"), follow=True)
    assert not Notification.objects.filter(recipient=amina).exists()
    assert "Cleared 3 notifications." in response.content.decode()


def test_clear_all_and_delete_are_offered_with_confirmation(client, amina):
    n = notify([amina], Type.ANNOUNCEMENT, "Hello")[0]
    html = client.get(reverse("notifications:list")).content.decode()
    assert f'action="{reverse("notifications:clear_all")}" data-confirm=' in html
    assert f'action="{reverse("notifications:delete", args=[n.pk])}"' in html
    Notification.objects.all().delete()
    assert 'data-testid="clear-all"' not in client.get(reverse("notifications:list")).content.decode()


def test_users_cannot_delete_other_users_notifications(client, amina):
    other = operator("other")
    theirs = notify([other], Type.ANNOUNCEMENT, "Private")[0]
    assert client.post(reverse("notifications:delete", args=[theirs.pk])).status_code == 404
    notify([amina], Type.ANNOUNCEMENT, "Mine")
    client.post(reverse("notifications:clear_all"))
    assert Notification.objects.filter(pk=theirs.pk).exists()
    assert not Notification.objects.filter(recipient=amina).exists()


def test_delete_and_clear_require_post(client, amina):
    n = notify([amina], Type.ANNOUNCEMENT, "x")[0]
    assert client.get(reverse("notifications:delete", args=[n.pk])).status_code == 405
    assert client.get(reverse("notifications:clear_all")).status_code == 405
    assert Notification.objects.filter(pk=n.pk).exists()


def test_delete_never_redirects_off_site(client, amina):
    n = notify([amina], Type.ANNOUNCEMENT, "x")[0]
    response = client.post(reverse("notifications:delete", args=[n.pk]), {"next": "https://evil.example/"})
    assert response.url == reverse("notifications:list")
