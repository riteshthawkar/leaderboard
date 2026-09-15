import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT = ROOT / "deployment" / "oci"


def test_oci_shell_scripts_have_valid_bash_syntax():
    scripts = sorted(DEPLOYMENT.glob("*.sh")) + sorted(
        (DEPLOYMENT.parent / "oci-micro").glob("*.sh")
    )
    assert scripts
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_oci_compose_keeps_services_private_and_persistent():
    compose = (DEPLOYMENT / "compose.yaml").read_text(encoding="utf-8")

    assert "api:" in compose
    assert "frontend:" in compose
    assert "caddy:" in compose
    assert "sqlite:////data/leaderboard.db" in compose
    assert "/srv/ms-vista/data" in compose
    assert "/mnt/ms-vista-backups" in compose
    assert '"80:80"' in compose
    assert '"443:443"' in compose
    assert '"7860:7860"' not in compose
    assert '"8080:8080"' not in compose
    assert "no-new-privileges:true" in compose


def test_oci_units_use_the_oci_stack_and_separate_backup_mount():
    app_service = (DEPLOYMENT / "ms-vista.service").read_text(encoding="utf-8")
    watchdog = (DEPLOYMENT / "ms-vista-watchdog.service").read_text(
        encoding="utf-8"
    )
    backup = (DEPLOYMENT / "ms-vista-backup-verify.service").read_text(
        encoding="utf-8"
    )

    assert "/srv/ms-vista/app/deployment/oci" in app_service
    assert "RequiresMountsFor=/srv/ms-vista /mnt/ms-vista-backups" in app_service
    assert "--no-build" in app_service
    assert "MS_VISTA_DEPLOY_DIR=/srv/ms-vista/app/deployment/oci" in watchdog
    assert "MS_VISTA_BACKUP_DIR=/mnt/ms-vista-backups" in backup


def test_oci_public_environment_uses_bearer_compatible_auth():
    env_example = (DEPLOYMENT / "production.env.example").read_text(
        encoding="utf-8"
    )

    assert "DEPLOYMENT_MODE=public" in env_example
    assert "AUTH_TRANSPORT=dual" in env_example
    assert "FRONTEND_BASE_URL=https://riteshthawkar.github.io/leaderboard" in env_example
    assert "CORS_ORIGINS=https://riteshthawkar.github.io,https://api.example.org" in env_example
    assert "BACKUP_VERIFY_ALLOWED_FSTYPES=ext4,xfs" in env_example
    assert "REQUIRE_OFFICIAL_SPATIAL=false" in env_example


def test_oci_host_check_rejects_same_filesystem_backups():
    script = (DEPLOYMENT / "check_host.sh").read_text(encoding="utf-8")

    assert "data_device" in script
    assert "backup_device" in script
    assert '[[ ${data_device} != "${backup_device}" ]]' in script
    assert "ext4" in script
    assert "xfs" in script


def test_micro_profile_is_explicit_and_keeps_production_guards():
    micro = DEPLOYMENT.parent / "oci-micro"
    compose = (micro / "compose.yaml").read_text(encoding="utf-8")
    check = (micro / "check_host.sh").read_text(encoding="utf-8")
    production_check = (DEPLOYMENT / "check_host.sh").read_text(encoding="utf-8")

    assert compose.count("file: ../oci/compose.yaml") == 3
    assert "GUNICORN_THREADS: ${MICRO_GUNICORN_THREADS:-1}" in compose
    assert "mem_limit: ${API_MEMORY_LIMIT:-512m}" in compose
    assert "MAX_SPATIAL_SUBMISSION_BYTES: ${MICRO_MAX_SPATIAL_SUBMISSION_BYTES:-33554432}" in compose
    assert "DISABLE_SUBMISSION_AUTH" not in compose
    assert "AUTO_BACKUP_ENABLED" not in compose
    assert "SwapTotal" in check
    assert "../oci/check_host.sh" in check
    assert "MIN_MEMORY_KIB:-3670016" in production_check


def test_deployment_validates_capacity_before_changing_the_release():
    deploy = (DEPLOYMENT.parent / "azure/deploy_release.sh").read_text()
    assert deploy.index('bash "${DEPLOY_DIR}/check_capacity.sh"') < deploy.index('sed -i "s/^MS_VISTA_IMAGE_TAG=')
    preflight = (DEPLOYMENT / "check_capacity.sh").read_text()
    assert "config --format json" in preflight
    assert "check_deployment_capacity.py" in preflight
    assert 'bash "${DEPLOY_DIR}/check_host.sh"' in preflight
