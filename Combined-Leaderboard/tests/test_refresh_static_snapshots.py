import pytest

from scripts.refresh_static_snapshots import _model_report_endpoints


def test_model_report_endpoints_cover_every_listed_model():
    payload = {
        "leaderboard": [
            {"model_name": "Qwen3.6-27B"},
            {"model_name": "Gemma 3 27B IT"},
            {"model_name": "Model/Variant"},
        ]
    }

    assert _model_report_endpoints(payload) == [
        "/api/model/Qwen3.6-27B/report",
        "/api/model/Gemma%203%2027B%20IT/report",
        "/api/model/Model%2FVariant/report",
    ]


def test_model_report_endpoints_reject_duplicate_models():
    payload = {
        "leaderboard": [
            {"model_name": "Qwen3.6-27B"},
            {"model_name": "Qwen3.6-27B"},
        ]
    }

    with pytest.raises(RuntimeError, match="repeats model"):
        _model_report_endpoints(payload)
