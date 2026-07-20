"""Apply the audited vLLM 0.25.1 Phi-4 mask-sum precision correction."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


PATCH_ID = "vllm-0.25.1-phi4mm-fp32-mask-sum-v1"
SUPPORTED_VLLM_VERSION = "0.25.1"
RELATIVE_TARGET = Path("vllm/model_executor/models/phi4mm.py")

ORIGINAL_BLOCK = """                temp_len = (
                    int(image_attention_mask[_bs, : B_ + 1, 0::2, 0::2].sum().item())
                    + (useful_height + 1)
                    + base_feat_height // base_feat_height_reduction
                )
"""

PATCHED_BLOCK = """                temp_len = (
                    int(
                        image_attention_mask[_bs, : B_ + 1, 0::2, 0::2]
                        .sum(dtype=torch.float32)
                        .item()
                    )
                    + (useful_height + 1)
                    + base_feat_height // base_feat_height_reduction
                )
"""


class CompatibilityPatchError(RuntimeError):
    """Raised when the installed dependency does not match the audited source."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def patched_source(source: str) -> tuple[str, bool]:
    """Return corrected source and whether a modification was required."""
    if PATCHED_BLOCK in source:
        return source, False
    occurrences = source.count(ORIGINAL_BLOCK)
    if occurrences != 1:
        raise CompatibilityPatchError(
            "Expected exactly one audited Phi-4 mask-sum block in vLLM "
            f"{SUPPORTED_VLLM_VERSION}, found {occurrences}. Refusing to modify an "
            "unknown dependency source."
        )
    return source.replace(ORIGINAL_BLOCK, PATCHED_BLOCK, 1), True


def installed_target() -> tuple[str, Path]:
    distribution = importlib.metadata.distribution("vllm")
    version = distribution.version.split("+", 1)[0]
    if version != SUPPORTED_VLLM_VERSION:
        raise CompatibilityPatchError(
            f"{PATCH_ID} supports vLLM {SUPPORTED_VLLM_VERSION}, found {version}."
        )
    return version, Path(distribution.locate_file(RELATIVE_TARGET)).resolve()


def main() -> int:
    version, target = installed_target()
    original_bytes = target.read_bytes()
    corrected, changed = patched_source(original_bytes.decode("utf-8"))

    if changed:
        backup = target.with_suffix(target.suffix + ".ms-vista-original")
        if not backup.exists():
            backup.write_bytes(original_bytes)
        temporary = target.with_suffix(target.suffix + ".ms-vista-tmp")
        temporary.write_text(corrected, encoding="utf-8")
        os.chmod(temporary, target.stat().st_mode)
        os.replace(temporary, target)

    corrected_bytes = target.read_bytes()
    if PATCHED_BLOCK.encode("utf-8") not in corrected_bytes:
        raise CompatibilityPatchError(f"{PATCH_ID} verification failed for {target}.")

    print(
        json.dumps(
            {
                "changed": changed,
                "patch_id": PATCH_ID,
                "sha256": sha256_bytes(corrected_bytes),
                "target": str(target),
                "vllm_version": version,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
