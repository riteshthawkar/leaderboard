from pathlib import Path
import json

import pytest

import production_smoke  # noqa: E402


def test_public_task_tree_contains_no_answer_keys():
    root = Path(__file__).resolve().parents[2]
    exposed = sorted(root.glob("tasks/**/ground_truth*"))

    assert exposed == [], (
        "Private answer keys must live under the ignored Ground_truths directory, "
        f"not the public tasks tree: {exposed}"
    )
    dockerignore = (root / ".dockerignore").read_text(encoding="utf-8")
    assert "tasks/**/ground_truth*" in dockerignore


def test_production_smoke_validates_public_controls(monkeypatch):
    readiness = {
        "status": "healthy",
        "components": {"spatial_bundle": "demo"},
        "details": {
            "deployment": {"public_deployment_ready": True},
            "auth": {"admin_ready": True},
            "backup": {
                "mirror_required": True,
                "mirror_configured": True,
                "mirror_separate_filesystem": True,
            },
        },
    }

    def fake_json(url, *, origin=None):
        assert origin == "https://app.example.com"
        if url.endswith("/api/health/live"):
            return 200, {}, {"status": "alive"}
        if url.endswith("/api/readiness"):
            return 200, {"Access-Control-Allow-Origin": origin}, readiness
        if url.endswith("/api/auth/providers"):
            return 200, {}, {"providers": [{"id": "microsoft"}]}
        raise AssertionError(url)

    monkeypatch.setattr(production_smoke, "_json", fake_json)
    monkeypatch.setattr(
        production_smoke,
        "_request",
        lambda _url: (
            200,
            {
                "Content-Security-Policy": "default-src 'self'",
                "Strict-Transport-Security": "max-age=31536000",
                "Permissions-Policy": "camera=()",
            },
            b"<html></html>",
        ),
    )

    result = production_smoke.run(
        "https://api.example.com",
        "https://app.example.com",
        allow_http=False,
        require_spatial=False,
    )

    assert result["status"] == "passed"
    assert not result["failed_checks"]


def test_pages_csp_requires_exact_api_origin_and_rejects_broad_https():
    valid = b'''<meta http-equiv="Content-Security-Policy" content="default-src 'self'; object-src 'none'; script-src 'self'; connect-src 'self' https://api.example.com">'''
    assert production_smoke._pages_csp_valid(valid, "https://api.example.com")
    assert not production_smoke._pages_csp_valid(valid, "https://wrong.example.com")
    assert not production_smoke._pages_csp_valid(valid.replace(b"https://api.example.com", b"https:"), "https://api.example.com")
    assert not production_smoke._pages_csp_valid(b"<html></html>", "https://api.example.com")


def test_startup_retry_requires_a_complete_success(monkeypatch, capsys):
    outcomes = iter([
        TimeoutError("startup timeout"),
        {"status": "failed", "failed_checks": ["api_readiness"]},
        {"status": "passed", "failed_checks": []},
    ])
    delays = []

    def check(*_args, **kwargs):
        assert kwargs["require_spatial"] is True
        result = next(outcomes)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(production_smoke, "run", check)
    monkeypatch.setattr(production_smoke.time, "sleep", delays.append)
    status = production_smoke.main([
        "--api-url", "https://api.example.com", "--frontend-url", "https://app.example.com",
        "--attempts", "6", "--require-spatial",
    ])
    assert status == 0
    output = capsys.readouterr()
    assert json.loads(output.out) == {"status": "passed", "failed_checks": [], "attempts": 3}
    assert delays == [5, 5]
    assert "startup timeout" in output.err
    assert "api_readiness" in output.err


def test_startup_retries_do_not_hide_persistent_failure(monkeypatch, capsys):
    delays = []
    monkeypatch.setattr(production_smoke, "run", lambda *_args, **_kwargs: {
        "status": "failed", "failed_checks": ["verified_admin"],
    })
    monkeypatch.setattr(production_smoke.time, "sleep", delays.append)
    assert production_smoke.main([
        "--api-url", "https://api.example.com", "--frontend-url", "https://app.example.com",
        "--attempts", "6",
    ]) == 1
    assert delays == [5] * 5
    assert json.loads(capsys.readouterr().out) == {
        "status": "failed", "failed_checks": ["verified_admin"], "attempts": 6,
    }


@pytest.mark.parametrize("attempts", ["0", "7"])
def test_startup_attempts_are_bounded(attempts):
    with pytest.raises(SystemExit) as error:
        production_smoke.main([
            "--api-url", "https://api.example.com", "--frontend-url", "https://app.example.com",
            "--attempts", attempts,
        ])
    assert error.value.code == 2
