import asyncio
import base64
import json
from pathlib import Path

import pandas as pd
import pytest

from spatial_harness.run_track3_vllm import (
    BASE_SYSTEM_PROMPT,
    COT_SYSTEM_PROMPT,
    MODES,
    _append_jsonl,
    _import_compatible_reuse,
    _infer_one,
    _load_existing,
    _message_content,
    build_question,
    build_records,
    circular_rotations,
    data_url,
    parse_args,
    prediction_payload,
    run_inference,
    system_prompt,
)


def _write_tsv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def test_build_question_supports_vqa():
    question = build_question("How many cubes?", {}, "OmniSpatial")
    assert "Answer the question directly with a short final answer." in question
    assert "Options:" not in question


def test_long_base64_image_is_not_probed_as_a_file_path():
    payload = base64.b64encode(b"embedded-image" * 100).decode("ascii")
    assert len(payload) > 255
    assert data_url(payload).startswith("data:image/png;base64,")


def test_circular_rotations_move_correct_text_through_every_letter():
    rotations = list(circular_rotations({"A": "red", "B": "blue", "C": "green"}, "B"))
    assert len(rotations) == 3
    assert [correct for _options, correct in rotations] == ["B", "C", "A"]
    assert all(options[correct] == "blue" for options, correct in rotations)


def test_build_records_routes_vqa_and_all_three_image_modes(tmp_path: Path):
    encoded_image = "a" * 80
    _write_tsv(
        tmp_path / "SpatialBench.tsv",
        [
            {
                "index": 1,
                "image": encoded_image,
                "question": "Where?",
                "A": "left",
                "B": "right",
                "answer": "A",
                "answer_type": "mcq",
            },
            {
                "index": 2,
                "image": "1",
                "question": "How many?",
                "answer": "3",
                "answer_type": "vqa",
            },
        ],
    )

    main = build_records(tmp_path, "SpatialBench", "main")
    noimage = build_records(tmp_path, "SpatialBench", "noimage")
    noimgpp = build_records(tmp_path, "SpatialBench", "noimgpp")

    assert MODES == ("main", "noimage", "noimgpp")
    assert len(main) == 3
    assert {record["answer_type"] for record in main} == {"mcq", "vqa"}
    assert {record["group"] for record in main if record["answer_type"] == "mcq"} == {"1"}
    assert main[0]["imgs"] == [encoded_image]
    assert main[2]["imgs"] == [encoded_image]
    assert main[2]["gt"] == "3"
    assert len(noimage) == 2
    assert all(record["gray"] is True for record in noimage)
    assert noimage[0]["gt"] == "A"
    assert noimage[0]["options"] == {"A": "left", "B": "right"}
    assert noimage[1]["answer_type"] == "vqa"
    assert noimage[1]["gt"] == "3"
    assert len(noimgpp) == 1
    assert noimgpp[0]["answer_type"] == "mcq"
    assert noimgpp[0]["options"]["C"] == "Cannot determine from the image"
    assert noimgpp[0]["cannot_label"] == "C"

    metadata_only = build_records(
        tmp_path,
        "SpatialBench",
        "main",
        include_payload=False,
    )
    assert len(metadata_only) == len(main)
    assert all(
        "question" not in record
        and "options" not in record
        and "imgs" not in record
        and "gray" not in record
        for record in metadata_only
    )


def test_published_system_prompts_are_not_user_suffixes():
    assert system_prompt("noncot") == BASE_SYSTEM_PROMPT
    assert system_prompt("cot") == COT_SYSTEM_PROMPT
    assert "<think> </think>" in COT_SYSTEM_PROMPT
    content = _message_content(
        {"imgs": [], "gray": False, "question": "Question:Where?"},
        "cot",
    )
    assert content == [{"type": "text", "text": "Question:Where?"}]


def test_3dsr_base_only_deduplicates_flip_rows(tmp_path: Path):
    rows = []
    for index, qid in enumerate(("q1", "q1-flip-1")):
        rows.append(
            {
                "index": index,
                "qid": qid,
                "image": "a" * 80,
                "question": "Where?",
                "A": "left",
                "B": "right",
                "answer": "A",
            }
        )
    _write_tsv(tmp_path / "3DSRBench.tsv", rows)
    assert len(build_records(tmp_path, "3DSRBench", "main")) == 1


def test_3dsr_official_index_format_deduplicates_flip_rows(tmp_path: Path):
    _write_tsv(
        tmp_path / "3DSRBench.tsv",
        [
            {
                "index": index,
                "image": "a" * 80,
                "question": "Where?",
                "A": "left",
                "B": "right",
                "answer": "A",
            }
            for index in ("VIN6MS3J", "VIN6MS3J-flip")
        ],
    )
    records = build_records(tmp_path, "3DSRBench", "main")
    assert len(records) == 1
    assert records[0]["index"] == "VIN6MS3J"


def test_prediction_payload_persists_v2_routing_fields():
    payload = prediction_payload(
        {
            "dataset": "SpatialBench",
            "index": "1_r0",
            "group": "1",
            "answer_type": "mcq",
            "options": {"A": "left"},
            "gt": "A",
            "cannot_label": None,
            "output": "A",
            "finish_reason": "stop",
            "completion_tokens": 17,
        },
        "main",
        "cot",
    )
    assert payload["group"] == "1"
    assert payload["answer_type"] == "mcq"
    assert payload["mode"] == "main"
    assert payload["pmode"] == "cot"
    assert payload["finish_reason"] == "stop"
    assert payload["completion_tokens"] == 17


def test_append_checkpoint_is_resume_readable_and_last_write_wins(tmp_path: Path):
    path = tmp_path / "pred.checkpoint.jsonl"
    original = {
        "dataset": "BLINK",
        "index": "1",
        "mode": "main",
        "pmode": "noncot",
        "output": "",
        "error": "temporary",
    }
    recovered = {**original, "output": "A"}
    recovered.pop("error")

    _append_jsonl(path, [original])
    _append_jsonl(path, [recovered])

    loaded = _load_existing(path)
    assert loaded[("BLINK", "1", "main", "noncot")]["output"] == "A"


def test_token_budgets_default_to_server_context_remainder_for_both_modes():
    args = parse_args(["--model", "model", "--endpoints", "http://localhost:8000/v1"])
    assert args.max_tokens_noncot == 0
    assert args.max_tokens_cot == 0


def test_context_remainder_omits_max_tokens_from_request():
    captured = {}

    class Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            choice = type(
                "Choice",
                (),
                {
                    "message": type("Message", (), {"content": "A"})(),
                    "finish_reason": "stop",
                },
            )()
            return type(
                "Response",
                (),
                {
                    "choices": [choice],
                    "usage": type("Usage", (), {"completion_tokens": 1})(),
                },
            )()

    client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": Completions()})()},
    )()
    item = {
        "imgs": [],
        "gray": False,
        "question": "Question:Where?",
    }
    asyncio.run(
        _infer_one(
            client,
            asyncio.Semaphore(1),
            item,
            "test/model",
            "noncot",
            0,
            0,
            1,
            0,
            0,
            {},
        )
    )
    assert "max_tokens" not in captured
    assert item["finish_reason"] == "stop"


def test_compatible_reuse_imports_only_naturally_stopped_rows(tmp_path: Path):
    source = tmp_path / "v4"
    target = tmp_path / "v5"
    source.mkdir()
    target.mkdir()
    target_contract = {
        "model": "test/model",
        "model_revision": "revision",
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "chat_template_kwargs": {"noncot": {}, "cot": {}},
        "prompt_sha256": {"noncot": "a", "cot": "b"},
        "dataset_sha256": {"BLINK": "c"},
        "datasets": ["BLINK"],
        "modes": ["main"],
        "prompt_modes": ["noncot"],
        "limit": 0,
    }
    source_config = {
        **target_contract,
        "schema_version": 4,
        "harness_contract": "ms-vista-track3-paper-aligned-v4",
        "max_tokens_noncot": 16384,
        "max_tokens_cot": 16384,
    }
    (source / "run_config.json").write_text(
        json.dumps(source_config),
        encoding="utf-8",
    )
    source_rows = [
        {
            "dataset": "BLINK",
            "index": "stop",
            "mode": "main",
            "pmode": "noncot",
            "output": "A",
            "finish_reason": "stop",
        },
        {
            "dataset": "BLINK",
            "index": "length",
            "mode": "main",
            "pmode": "noncot",
            "output": "reasoning",
            "finish_reason": "length",
        },
    ]
    (source / "pred_main_noncot.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in source_rows),
        encoding="utf-8",
    )

    counts = _import_compatible_reuse(source, target, target_contract)
    imported = _load_existing(target / "pred_main_noncot.checkpoint.jsonl")

    assert counts == {"main_noncot": 1}
    assert set(imported) == {("BLINK", "stop", "main", "noncot")}
    row = next(iter(imported.values()))
    assert row["reuse_provenance"]["eligibility"] == "greedy_natural_stop"


def test_prompt_modes_parse_separate_thinking_kwargs():
    args = parse_args(
        [
            "--model",
            "Qwen/Qwen3.6-27B",
            "--endpoints",
            "http://localhost:8031/v1",
            "--chat-template-kwargs-noncot",
            '{"enable_thinking":false}',
            "--chat-template-kwargs-cot",
            '{"enable_thinking":true}',
        ]
    )
    assert args.chat_template_kwargs_noncot == {"enable_thinking": False}
    assert args.chat_template_kwargs_cot == {"enable_thinking": True}


def test_records_are_prepared_before_endpoint_validation(tmp_path: Path, monkeypatch):
    _write_tsv(
        tmp_path / "BLINK.tsv",
        [
            {
                "index": 0,
                "image": "a" * 80,
                "question": "Where?",
                "A": "left",
                "B": "right",
                "answer": "A",
            }
        ],
    )
    args = parse_args(
        [
            "--model",
            "test/model",
            "--endpoints",
            "http://localhost:8031/v1",
            "--datasets",
            "BLINK",
            "--lmudata",
            str(tmp_path),
            "--out",
            str(tmp_path / "results"),
            "--modes",
            "main",
            "--prompt-modes",
            "noncot",
        ]
    )
    events = []

    class Client:
        pass

    monkeypatch.setattr(
        "spatial_harness.run_track3_vllm.AsyncOpenAI", lambda **_kwargs: Client()
    )
    original_build_records = build_records

    def tracked_build_records(*build_args, **build_kwargs):
        events.append("build_records")
        return original_build_records(*build_args, **build_kwargs)

    async def tracked_wait(_clients, _model, _startup_timeout):
        events.append("validate_endpoints")

    async def fake_infer(_client, _semaphore, item, *_args, **_kwargs):
        item["output"] = "A"

    monkeypatch.setattr(
        "spatial_harness.run_track3_vllm.build_records", tracked_build_records
    )
    monkeypatch.setattr(
        "spatial_harness.run_track3_vllm._wait_for_endpoints", tracked_wait
    )
    monkeypatch.setattr("spatial_harness.run_track3_vllm._infer_one", fake_infer)

    asyncio.run(run_inference(args))

    assert events == ["build_records", "validate_endpoints"]


def test_run_contract_refuses_changed_resume(tmp_path: Path, monkeypatch):
    args = parse_args(
        [
            "--model",
            "Qwen/Qwen3.6-27B",
            "--model-revision",
            "revision-a",
            "--endpoints",
            "http://localhost:8031/v1",
            "--datasets",
            "BLINK",
            "--lmudata",
            str(tmp_path),
            "--out",
            str(tmp_path / "results"),
            "--limit",
            "1",
        ]
    )
    _write_tsv(
        tmp_path / "BLINK.tsv",
        [
            {
                "index": 0,
                "image": "a" * 80,
                "question": "Where?",
                "A": "left",
                "B": "right",
                "answer": "A",
            }
        ],
    )

    class Models:
        async def list(self):
            return type("Result", (), {"data": [type("Model", (), {"id": args.model})()]})()

    class Client:
        models = Models()

    monkeypatch.setattr(
        "spatial_harness.run_track3_vllm.AsyncOpenAI", lambda **_kwargs: Client()
    )
    async def fake_infer(_client, _semaphore, item, *_args, **_kwargs):
        item["output"] = "A"

    monkeypatch.setattr("spatial_harness.run_track3_vllm._infer_one", fake_infer)

    __import__("asyncio").run(run_inference(args))
    config = json.loads((args.out / "run_config.json").read_text(encoding="utf-8"))
    assert config["model_revision"] == "revision-a"

    changed = parse_args(
        [
            "--model",
            args.model,
            "--model-revision",
            "revision-b",
            "--endpoints",
            "http://localhost:8031/v1",
            "--datasets",
            "BLINK",
            "--lmudata",
            str(tmp_path),
            "--out",
            str(args.out),
            "--limit",
            "1",
        ]
    )
    with pytest.raises(SystemExit, match="refusing to mix checkpoints"):
        __import__("asyncio").run(run_inference(changed))

    scaled = parse_args(
        [
            "--model",
            args.model,
            "--model-revision",
            "revision-a",
            "--endpoints",
            "http://localhost:8031/v1,http://localhost:8032/v1",
            "--datasets",
            "BLINK",
            "--lmudata",
            str(tmp_path),
            "--out",
            str(args.out),
            "--limit",
            "1",
            "--server-metadata",
            '{"replicas":2}',
        ]
    )
    asyncio.run(run_inference(scaled))
    migrated = json.loads((args.out / "run_config.json").read_text(encoding="utf-8"))
    assert migrated["endpoints"] == [
        "http://localhost:8031/v1",
        "http://localhost:8032/v1",
    ]
    assert migrated["execution_migrations"][-1]["reason"] == (
        "throughput-only-endpoint-scaling"
    )


def test_continuous_workers_do_not_wait_for_batch_straggler(tmp_path: Path, monkeypatch):
    _write_tsv(
        tmp_path / "BLINK.tsv",
        [
            {
                "index": index,
                "image": "a" * 80,
                "question": "Where?",
                "A": "left",
                "B": "right",
                "answer": "A",
            }
            for index in range(3)
        ],
    )
    args = parse_args(
        [
            "--model",
            "test/model",
            "--endpoints",
            "http://localhost:8031/v1",
            "--datasets",
            "BLINK",
            "--lmudata",
            str(tmp_path),
            "--out",
            str(tmp_path / "results"),
            "--modes",
            "main",
            "--prompt-modes",
            "noncot",
            "--concurrency",
            "2",
            "--checkpoint-every",
            "1",
        ]
    )

    class Models:
        async def list(self):
            return type("Result", (), {"data": [type("Model", (), {"id": args.model})()]})()

    class Client:
        models = Models()

    monkeypatch.setattr(
        "spatial_harness.run_track3_vllm.AsyncOpenAI", lambda **_kwargs: Client()
    )
    release_slow = asyncio.Event()
    third_completed = asyncio.Event()

    async def fake_infer(_client, _semaphore, item, *_args, **_kwargs):
        if item["index"] == "0":
            await release_slow.wait()
        item["output"] = "A"
        if item["index"] == "2":
            third_completed.set()

    monkeypatch.setattr("spatial_harness.run_track3_vllm._infer_one", fake_infer)

    async def exercise():
        task = asyncio.create_task(run_inference(args))
        await asyncio.wait_for(third_completed.wait(), timeout=1)
        checkpoint = _load_existing(
            args.out / "pred_main_noncot.checkpoint.jsonl"
        )
        assert ("BLINK", "1", "main", "noncot") in checkpoint
        assert ("BLINK", "2", "main", "noncot") in checkpoint
        assert not task.done()
        release_slow.set()
        await task

    asyncio.run(exercise())
