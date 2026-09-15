import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT = ROOT / "deployment" / "azure"


def test_azure_shell_scripts_have_valid_bash_syntax():
    scripts = sorted(DEPLOYMENT.glob("*.sh"))
    assert scripts
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_release_deployment_has_smoke_checked_rollback_and_locking():
    script = (DEPLOYMENT / "deploy_release.sh").read_text(encoding="utf-8")

    assert "ms-vista-deployment.lock" in script
    assert "scripts/production_smoke.py" in script
    assert "--attempts 6" in script
    assert "restored and verified image tag" in script
    assert "CRITICAL: deployment and automatic rollback both failed" in script
    assert "RELEASE_COMMIT_FILE" in script
    assert "^[0-9a-f]{40}$" in script
    assert "MS_VISTA_DEPLOY_DIR" in script


def test_watchdog_requires_repeated_failures_and_has_recovery_cooldown():
    script = (DEPLOYMENT / "ms-vista-watchdog.sh").read_text(encoding="utf-8")
    timer = (DEPLOYMENT / "ms-vista-watchdog.timer").read_text(encoding="utf-8")

    assert "WATCHDOG_FAILURE_THRESHOLD:-3" in script
    assert "WATCHDOG_RECOVERY_COOLDOWN_SECONDS:-900" in script
    assert "ms-vista-deployment.lock" in script
    assert "OnUnitActiveSec=1min" in timer


def test_backup_verifier_performs_an_offline_restore_drill():
    script = (DEPLOYMENT / "verify_latest_backup.sh").read_text(encoding="utf-8")
    timer = (DEPLOYMENT / "ms-vista-backup-verify.timer").read_text(encoding="utf-8")

    assert "python -m backend.backup_cli verify" in script
    assert "python -m backend.backup_cli restore" in script
    assert "--destination" in script
    assert "BACKUP_VERIFY_ALLOWED_FSTYPES" in script
    assert "OnCalendar=" in timer


def test_boot_service_never_builds_images_and_waits_for_health():
    service = (DEPLOYMENT / "ms-vista.service").read_text(encoding="utf-8")

    assert "--no-build" in service
    assert "--wait" in service
    assert "--wait-timeout 180" in service


def test_compose_exposes_release_controls_without_weakening_safe_defaults():
    compose = (DEPLOYMENT / "compose.yaml").read_text(encoding="utf-8")
    env_example = (DEPLOYMENT / "production.env.example").read_text(encoding="utf-8")

    assert "LIMITER_STORAGE_URI: ${LIMITER_STORAGE_URI:-memory://}" in compose
    assert "REQUIRE_OFFICIAL_SPATIAL: ${REQUIRE_OFFICIAL_SPATIAL:-false}" in compose
    assert (
        "MAX_SPATIAL_SUBMISSION_BYTES: "
        "${MAX_SPATIAL_SUBMISSION_BYTES:-67108864}"
    ) in compose
    assert "LIMITER_STORAGE_URI=memory://" in env_example
    assert "MAX_SPATIAL_SUBMISSION_BYTES=67108864" in env_example
