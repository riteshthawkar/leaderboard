import pytest

from evaluation.common.patch_vllm_phi4mm import (
    ORIGINAL_BLOCK,
    PATCHED_BLOCK,
    CompatibilityPatchError,
    patched_source,
)


def test_patched_source_transforms_exact_audited_block_once():
    source = f"before\n{ORIGINAL_BLOCK}after\n"

    corrected, changed = patched_source(source)

    assert changed is True
    assert corrected == f"before\n{PATCHED_BLOCK}after\n"
    assert ORIGINAL_BLOCK not in corrected


def test_patched_source_is_idempotent():
    source = f"before\n{PATCHED_BLOCK}after\n"

    corrected, changed = patched_source(source)

    assert changed is False
    assert corrected == source


def test_patched_source_rejects_unknown_layout():
    with pytest.raises(CompatibilityPatchError, match="Refusing to modify"):
        patched_source("unknown vLLM source\n")
