import importlib
import json
from datetime import datetime, timezone

import pytest

from config import EVAL_CONDITIONS, SPATIAL_DATASET_KEYS
from leaderboard_store import LeaderboardStore
from models.tasks import TaskScore
from prepare_track3_v2 import write_contract
from spatial_submission import _load_public_spatial_contract, load_official_benchmark_manifest


def test_submitted_catalog_requires_explicit_admission_and_exact_hashes(tmp_path):
    catalog = [{"question_id": f"{dataset}:sample", "dataset_key": dataset,
                "evaluation_group": f"{dataset}:sample", "answer_type": "mcq", "conditions": EVAL_CONDITIONS}
               for dataset in SPATIAL_DATASET_KEYS]
    manifest = write_contract(tmp_path / "contract", catalog, "a" * 64)
    template = (tmp_path / "contract/submission_template.jsonl").read_bytes()
    questions = (tmp_path / "contract/questions.jsonl").read_bytes()
    with pytest.raises(ValueError, match="schema_version"):
        load_official_benchmark_manifest(manifest)
    with pytest.raises(ValueError, match="schema_version"):
        _load_public_spatial_contract(manifest, template, questions)
    _, loaded, expected = _load_public_spatial_contract(manifest, template, questions, allow_submitted_cohort=True)
    assert len(loaded) == 13 and len(expected) == 78
    with pytest.raises(ValueError, match="hash"):
        _load_public_spatial_contract(manifest, template, questions + b"\n", allow_submitted_cohort=True)
    modified = json.loads(manifest)
    modified["official_protocol_attested"] = True
    with pytest.raises(ValueError, match="policy"):
        _load_public_spatial_contract(json.dumps(modified).encode(), template, questions, allow_submitted_cohort=True)


def _score(name, version, digest, accuracy):
    return TaskScore(task_id="spatial", submission_id=f"score-{name}", submitted_at=datetime.now(timezone.utc), model_name=name, accuracy=accuracy, total_samples=1,
                     correct_samples=0, metadata={"spatial_run": {"benchmark_version": version,
                     "benchmark_manifest_sha256": digest, "missing_output_rows": 3}})


def test_rankings_never_mix_versions_or_distinct_manifests(tmp_path):
    store = LeaderboardStore(tmp_path / "store.json")
    store.add_result(_score("Official", "paper-aligned-v5-2026-07", "a" * 64, 0.5))
    store.add_result(_score("New", "v2", "b" * 64, 0.8))
    store.add_result(_score("Different samples", "v2", "c" * 64, 0.9))
    assert len(store.spatial_cohorts()) == 3
    assert [row["model_name"] for row in store.spatial_leaderboard()] == ["Official"]
    selected = store.spatial_leaderboard(cohort="b" * 64)
    assert [row["model_name"] for row in selected] == ["New"]
    assert selected[0]["rank"] == 1 and selected[0]["missing_output_rows"] == 3
    assert len(store.public_submission_ids()) == 3
    with pytest.raises(ValueError, match="Unknown"):
        store.spatial_leaderboard(cohort="all")


def test_api_cohort_filter_and_unknown_cohort_fail_closed(tmp_path, monkeypatch):
    app_module = importlib.import_module("web.app")
    store = LeaderboardStore(tmp_path / "store.json")
    store.add_result(_score("Official", "paper-aligned-v5-2026-07", "a" * 64, 0.5))
    store.add_result(_score("New", "v2", "b" * 64, 0.8))
    monkeypatch.setattr(app_module, "leaderboard_store", store)
    with app_module.app.test_client() as client:
        result = client.get("/api/leaderboard/spatial").get_json()
        assert result["count"] == 1 and len(result["cohorts"]) == 2
        assert result["leaderboard"][0]["model_name"] == "Official"
        for url in ("/api/leaderboard/spatial?cohort=", "/api/leaderboard?task=spatial&cohort="):
            selected = client.get(url + "b" * 64)
            assert selected.status_code == 200
            assert selected.get_json()["leaderboard"][0]["model_name"] == "New"
            assert client.get(url + "all").status_code == 400


def test_stored_revalidation_allows_only_explicit_model_repository_mapping():
    from types import SimpleNamespace
    app_module = importlib.import_module("web.app")
    package = SimpleNamespace(manifest={"model": {"name": "source/snapshot"}})
    stored = {"model_name": "Display Name", "model_meta": {"model_repository": "source/snapshot"}}
    assert app_module._stored_artifact_model_name(stored, package) == "source/snapshot"
    stored["model_meta"] = {}
    with pytest.raises(ValueError, match="not bound"):
        app_module._stored_artifact_model_name(stored, package)
