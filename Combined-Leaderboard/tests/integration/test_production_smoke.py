from pathlib import Path

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
