"""Deployment preparation: data-source banner from model metadata, BITETRACKER_STATE_DIR and /health/.

What this file checks:
- The data banner follows the saved model metadata's `data_source`, not whether the
  Google Sheet link happens to be configured on this machine.
- BITETRACKER_STATE_DIR moves the database, model, metadata, processed CSV, metrics and
  uploads out of the code directory, and leaving it unset keeps the old local paths.
- /health/ needs no login, reports only ok/fail per check (no paths or error text), and
  answers 503 when an artifact is missing or, in production, when the model is synthetic.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from django.urls import reverse

from config import settings as ml_settings
from services import status_service
from src.feature_engineering import FOOD4EDUCATION_FEATURE_COLUMNS, FOOD4EDUCATION_SOURCE, SYNTHETIC_SOURCE

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --- Data-source banner -----------------------------------------------------------------------


def test_status_real_from_metadata_without_google_sheet(stub_artifacts, make_metadata):
    """Production has no Google Sheet link and the default (synthetic) dataset path; metadata still wins."""
    assert ml_settings.USE_GOOGLE_SHEET is False
    stub_artifacts.write_metadata(
        make_metadata(results_label="FOOD4EDUCATION RESEARCH RESULTS", data_source=FOOD4EDUCATION_SOURCE)
    )
    status = status_service.get_data_status()
    assert status["kind"] == "real"
    assert status["label"] == "FOOD4EDUCATION RESEARCH RESULTS"
    assert status["dataset_name"] == ml_settings.GOOGLE_SHEET_DATASET_NAME
    assert status["data_source"] == FOOD4EDUCATION_SOURCE


def test_status_synthetic_from_metadata_even_with_google_sheet(stub_artifacts, make_metadata, monkeypatch):
    monkeypatch.setattr(ml_settings, "USE_GOOGLE_SHEET", True)
    stub_artifacts.write_metadata(make_metadata(results_label="SOME RESULTS", data_source=SYNTHETIC_SOURCE))
    assert status_service.get_data_status()["kind"] == "synthetic"


def test_status_synthetic_label_beats_real_source(stub_artifacts, make_metadata):
    stub_artifacts.write_metadata(
        make_metadata(results_label="SYNTHETIC DEVELOPMENT RESULTS", data_source=FOOD4EDUCATION_SOURCE)
    )
    assert status_service.get_data_status()["kind"] == "synthetic"


def test_status_unknown_for_unrecognised_source(stub_artifacts, make_metadata):
    stub_artifacts.write_metadata(make_metadata(results_label="REAL RESULTS", data_source="mystery"))
    status = status_service.get_data_status()
    assert status["kind"] == "unknown"
    assert status["label"] == status_service.UNKNOWN_LABEL


@pytest.mark.django_db
def test_banner_real_without_google_sheet(client, django_user_model, stub_artifacts, make_metadata):
    stub_artifacts.write_metadata(
        make_metadata(results_label="FOOD4EDUCATION RESEARCH RESULTS", data_source=FOOD4EDUCATION_SOURCE)
    )
    client.force_login(django_user_model.objects.create_user("viewer", password="Str0ng-test-passw0rd!"))
    html = client.get(reverse("core:dashboard")).content.decode()
    assert "data-banner--real" in html and "FOOD4EDUCATION RESEARCH RESULTS" in html
    assert "SYNTHETIC DEVELOPMENT DATA" not in html


# --- BITETRACKER_STATE_DIR --------------------------------------------------------------------

_PATHS_CODE = (
    "import json, django; django.setup();"
    "from django.conf import settings as d; from config import settings as m;"
    "print(json.dumps({'db': str(d.DATABASES['default']['NAME']), 'media': str(d.MEDIA_ROOT),"
    "'models': str(m.MODELS_DIR), 'processed': str(m.PROCESSED_DATA_DIR), 'metrics': str(m.METRICS_DIR),"
    "'uploads': str(m.UPLOADS_DIR), 'sample': str(m.SYNTHETIC_DATA_PATH), 'static': str(d.STATIC_ROOT)}))"
)


def _paths(**env_overrides) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DJANGO_", "BITETRACKER_STATE_DIR"))}
    env.update({"DJANGO_SETTINGS_MODULE": "bitetracker_web.settings", **env_overrides})
    result = subprocess.run([sys.executable, "-c", _PATHS_CODE], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_state_dir_unset_keeps_local_paths():
    paths = _paths()
    root = PROJECT_ROOT.resolve()
    assert Path(paths["db"]) == root / "db.sqlite3"
    assert Path(paths["models"]) == root / "models"
    assert Path(paths["processed"]) == root / "data" / "processed"
    assert Path(paths["metrics"]) == root / "reports" / "metrics"
    assert Path(paths["media"]) == Path(paths["uploads"]) == root / "data" / "uploads"


def test_state_dir_moves_all_state(tmp_path):
    state = tmp_path / "state"
    paths = _paths(BITETRACKER_STATE_DIR=str(state))
    state = state.resolve()
    assert Path(paths["db"]) == state / "db.sqlite3"
    assert Path(paths["models"]) == state / "models"
    assert Path(paths["processed"]) == state / "data" / "processed"
    assert Path(paths["metrics"]) == state / "reports" / "metrics"
    assert Path(paths["media"]) == Path(paths["uploads"]) == state / "data" / "uploads"
    # Code-owned files stay with the code.
    assert Path(paths["sample"]).is_relative_to(PROJECT_ROOT.resolve())
    assert Path(paths["static"]).is_relative_to(PROJECT_ROOT.resolve())


def test_explicit_sqlite_path_still_wins(tmp_path):
    paths = _paths(BITETRACKER_STATE_DIR=str(tmp_path / "state"), DJANGO_SQLITE_PATH=str(tmp_path / "other.sqlite3"))
    assert Path(paths["db"]) == tmp_path / "other.sqlite3"


# --- /health/ -----------------------------------------------------------------------------------


@pytest.mark.django_db
def test_health_ok_without_login(client, settings, stub_artifacts):
    settings.DEBUG = True  # local development: the synthetic stub model is acceptable
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "checks": {"database": "ok", "model": "ok", "processed_data": "ok"},
    }
    assert "no-cache" in response.headers.get("Cache-Control", "")


@pytest.mark.django_db
def test_health_fails_without_model_and_hides_details(client, stub_artifacts):
    stub_artifacts.model_path.unlink()
    response = client.get("/health/")
    assert response.status_code == 503
    body = response.content.decode()
    assert response.json()["checks"]["model"] == "fail"
    assert "/" not in json.dumps(response.json()["checks"])
    assert ".joblib" not in body and str(stub_artifacts.model_path.parent) not in body


@pytest.mark.django_db
def test_health_fails_without_processed_data(client, stub_artifacts):
    stub_artifacts.processed_path.unlink()
    response = client.get("/health/")
    assert response.status_code == 503
    assert response.json()["checks"]["processed_data"] == "fail"


@pytest.mark.django_db
def test_health_rejects_synthetic_model_in_production(client, settings, stub_artifacts):
    settings.DEBUG = False
    response = client.get("/health/")
    assert response.status_code == 503
    assert response.json()["checks"]["real_data_model"] == "fail"


@pytest.mark.django_db
def test_health_rejects_synthetic_source_or_legacy_metadata_in_production(client, settings, stub_artifacts, make_metadata):
    settings.DEBUG = False
    stub_artifacts.write_metadata(
        make_metadata(results_label="FOOD4EDUCATION RESEARCH RESULTS", data_source=SYNTHETIC_SOURCE)
    )
    assert client.get("/health/").status_code == 503
    # Legacy metadata without data_source counts as synthetic (as model_service assumes).
    stub_artifacts.write_metadata(make_metadata(results_label="FOOD4EDUCATION RESEARCH RESULTS"))
    assert client.get("/health/").status_code == 503


@pytest.mark.django_db
def test_health_accepts_food4education_model_in_production(client, settings, stub_artifacts, make_model, make_metadata):
    settings.DEBUG = False
    features = list(FOOD4EDUCATION_FEATURE_COLUMNS)
    stub_artifacts.write_model(make_model(features))
    stub_artifacts.feature_columns_path.write_text(json.dumps(features))
    stub_artifacts.write_metadata(
        make_metadata(
            results_label="FOOD4EDUCATION RESEARCH RESULTS", data_source=FOOD4EDUCATION_SOURCE, features_used=features
        )
    )
    response = client.get("/health/")
    assert response.status_code == 200, response.json()
    assert response.json()["checks"]["real_data_model"] == "ok"


@pytest.mark.django_db
def test_health_only_allows_safe_methods(client, stub_artifacts):
    assert client.post("/health/").status_code == 405


# --- Deployment shell scripts --------------------------------------------------------------------

DEPLOY_SCRIPTS = sorted((PROJECT_ROOT / "deploy" / "scripts").glob("*.sh"))


@pytest.mark.parametrize("script", DEPLOY_SCRIPTS, ids=lambda p: p.name)
def test_deploy_scripts_have_valid_bash_syntax(script):
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def _prune(tmp_path, keep, *, current=None, previous="", release=""):
    """Run common.sh's prune_releases under deploy.sh's strict mode; return (result, remaining names)."""
    releases_dir = tmp_path / "releases"
    releases_dir.mkdir(exist_ok=True)
    current_link = tmp_path / "current"
    if current is not None and not current_link.exists():
        current_link.symlink_to(releases_dir / current)
    script = (
        "set -euo pipefail\n"
        f'. "{PROJECT_ROOT / "deploy" / "scripts" / "common.sh"}"\n'
        'prune_releases "$KEEP"\n'
        'echo "prune-ok"\n'
    )
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "APP_ROOT": str(tmp_path),  # common.sh derives RELEASES_DIR and CURRENT_LINK from it
        "KEEP": str(keep),
        "PREVIOUS": str(releases_dir / previous) if previous else "",
        "RELEASE": str(releases_dir / release) if release else "",
    }
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    return result, sorted(p.name for p in releases_dir.iterdir())


def _make_releases(tmp_path, *names):
    """Create release directories, oldest first (distinct modification times)."""
    releases_dir = tmp_path / "releases"
    releases_dir.mkdir(exist_ok=True)
    for age, name in enumerate(reversed(names)):
        d = releases_dir / name
        d.mkdir()
        stamp = 1_700_000_000 - age * 3600
        os.utime(d, (stamp, stamp))


def test_prune_releases_empty_dir_first_deployment(tmp_path):
    """Regression: first deploy aborted with exit 2 because `ls releases/*/` failed on an empty dir."""
    for keep in (0, 2, 3):  # deploy.sh calls it with KEEP_RELEASES - 1 and KEEP_RELEASES
        result, remaining = _prune(tmp_path, keep, release="new")
        assert result.returncode == 0, result.stderr
        assert "prune-ok" in result.stdout
        assert remaining == []


def test_prune_releases_single_release_is_kept(tmp_path):
    _make_releases(tmp_path, "r1")
    result, remaining = _prune(tmp_path, 0, current="r1", previous="r1", release="r2")
    assert result.returncode == 0, result.stderr
    assert remaining == ["r1"]


def test_prune_releases_keeps_newest_and_protected(tmp_path):
    _make_releases(tmp_path, "r1", "r2", "r3", "r4", "r5")  # r1 oldest, r5 newest
    # r1 is active and the rollback target although it is the oldest: it must survive.
    result, remaining = _prune(tmp_path, 2, current="r1", previous="r1", release="r6")
    assert result.returncode == 0, result.stderr
    assert remaining == ["r1", "r4", "r5"]


def test_prune_releases_never_removes_release_being_deployed(tmp_path):
    _make_releases(tmp_path, "new", "r2", "r3")  # the release being deployed is the oldest dir
    result, remaining = _prune(tmp_path, 1, current="r3", previous="r3", release="new")
    assert result.returncode == 0, result.stderr
    assert remaining == ["new", "r3"]


def test_deploy_passes_backup_label_as_separate_argument():
    """Regression: `"$SCRIPT_DIR/backup.sh""pre-deploy-..."` ran a non-existent file and aborted every deploy."""
    deploy = (PROJECT_ROOT / "deploy" / "scripts" / "deploy.sh").read_text()
    assert 'bash "$SCRIPT_DIR/backup.sh" "pre-deploy-$RELEASE_ID"' in deploy
    assert 'backup.sh""' not in deploy
    # backup.sh must accept every character deploy.sh allows in a release id.
    backup = (PROJECT_ROOT / "deploy" / "scripts" / "backup.sh").read_text()
    assert "^[A-Za-z0-9._-]+$" in backup
