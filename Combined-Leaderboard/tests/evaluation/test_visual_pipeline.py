import base64
import json

from evaluation.common.visual_pipeline import (
    final_answer,
    image_for_openai,
    write_diagnostics,
)


def test_final_answer_rejects_accidental_choice_letters_and_unclosed_reasoning():
    assert final_answer("reasoning without a final choice", "mcq_letter") == ""
    assert final_answer("<think>There are 7 candidates", "integer") == ""


def test_openai_image_payload_preserves_original_bytes(tmp_path):
    original = b"\x89PNG\r\n\x1a\noriginal-benchmark-image-bytes"
    image = tmp_path / "sample.png"
    image.write_bytes(original)

    payload = image_for_openai({"question_id": "q1", "image": str(image)}, None)

    header, encoded = payload.split(",", 1)
    assert header == "data:image/png;base64"
    assert base64.b64decode(encoded) == original


def test_diagnostics_retain_finish_reason_and_completion_tokens(tmp_path):
    output = tmp_path / "diagnostics.jsonl"
    write_diagnostics(
        output,
        [
            {
                "question_id": "q1",
                "source_subset": "subset",
                "answer_type": "mcq_letter",
                "output": "<answer>A</answer>",
                "finish_reason": "stop",
                "completion_tokens": 8,
            }
        ],
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["finish_reason"] == "stop"
    assert row["completion_tokens"] == 8
