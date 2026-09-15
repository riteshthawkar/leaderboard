import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "capacity", Path(__file__).resolve().parents[2] / "scripts/check_deployment_capacity.py"
)
capacity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capacity)
MIB = capacity.MIB


def config():
    return {"services": {
        "api": {"mem_limit": "512m", "memswap_limit": "768m", "environment": {
            "WEB_CONCURRENCY": "1", "GUNICORN_THREADS": "1",
            "MAX_SPATIAL_ARCHIVE_BYTES": str(10 * MIB),
            "MAX_SPATIAL_MULTIPART_BYTES": str(12 * MIB),
            "MAX_SPATIAL_SUBMISSION_BYTES": str(32 * MIB),
        }},
        "frontend": {"mem_limit": "32m", "memswap_limit": "64m"},
        "caddy": {"mem_limit": "96m", "memswap_limit": "128m", "environment": {"GOMEMLIMIT": "64MiB"}},
    }}


def test_micro_profile_fits_without_treating_swap_as_ram():
    result = capacity.validate_capacity(config(), ram=960 * MIB, swap=2048 * MIB)
    assert result["container_memory_mib"] == {"api": 512, "frontend": 32, "caddy": 96}
    assert result["api_threads"] == 1


def test_vertical_upgrade_uses_same_services_and_environment_only():
    upgraded = config()
    upgraded["services"]["api"].update(mem_limit="2g", memswap_limit="2560m")
    upgraded["services"]["api"]["environment"]["GUNICORN_THREADS"] = "4"
    assert capacity.validate_capacity(upgraded, ram=3900 * MIB, swap=2048 * MIB)["api_threads"] == 4
    with pytest.raises(ValueError, match="insufficient RAM"):
        capacity.validate_capacity(upgraded, ram=960 * MIB, swap=10000 * MIB)


@pytest.mark.parametrize("value", [0, -1, "0m", "unlimited", "", None, "secret=value"])
def test_invalid_memory_limits_are_rejected_without_echoing_values(value):
    with pytest.raises(ValueError, match="positive, finite"):
        capacity.memory_bytes(value)


@pytest.mark.parametrize("key,value", [("WEB_CONCURRENCY", "2"), ("GUNICORN_THREADS", "4"), ("MAX_SPATIAL_ARCHIVE_BYTES", str(300 * MIB))])
def test_rejects_unsafe_worker_or_upload_configuration(key, value):
    document = config()
    document["services"]["api"]["environment"][key] = value
    with pytest.raises(ValueError):
        capacity.validate_capacity(document, ram=960 * MIB, swap=2048 * MIB)


def test_missing_swap_and_go_headroom_are_rejected():
    with pytest.raises(ValueError, match="swap"):
        capacity.validate_capacity(config(), ram=960 * MIB, swap=0)
    document = config()
    document["services"]["caddy"]["environment"]["GOMEMLIMIT"] = "96MiB"
    with pytest.raises(ValueError, match="Go memory"):
        capacity.validate_capacity(document, ram=960 * MIB, swap=2048 * MIB)
