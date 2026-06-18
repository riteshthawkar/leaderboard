"""
Resolves a ``--model`` argument into a callable ``adapter(messages) -> str``.

Three kinds of model are supported, tried in this order:

1. ``example`` / ``example_adapter`` -- a deterministic offline stub (no model).
2. A VLMEvalKit model name from ``vlmeval.config.supported_VLM`` (e.g.
   ``Qwen2-VL-7B-Instruct``, ``GPT4o``). The toolkit instantiates and runs it.
3. An importable python module exposing ``model_generate(messages) -> str``
   (or ``generate``) -- for fully custom models.

``messages`` is VLMEvalKit's interleaved list, e.g.
``[{"type": "image", "value": "/path.png"}, {"type": "text", "value": "..."}]``.
"""

import importlib
import re

_OPTION_RE = re.compile(r"^([A-H])\.\s", re.MULTILINE)


def example_adapter(messages):
    """Offline stub: echo the first MCQ option letter, else a fixed answer."""
    for msg in messages:
        if msg.get("type") == "text":
            match = _OPTION_RE.search(msg.get("value", ""))
            return match.group(1) if match else "A"
    return "A"


def _wrap_vlmeval_model(model):
    def _call(messages):
        return str(model.generate(messages))
    return _call


def load_adapter(name):
    if name in ("example", "example_adapter"):
        return example_adapter

    # 2. a VLMEvalKit-supported model
    try:
        from vlmeval.config import supported_VLM
        if name in supported_VLM:
            print(f"  Loading VLMEvalKit model '{name}' ...")
            return _wrap_vlmeval_model(supported_VLM[name]())
    except Exception as exc:  # noqa: BLE001 - fall through to custom module
        print(f"  (vlmeval model lookup skipped: {exc})")

    # 3. a custom module exposing model_generate / generate
    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        raise SystemExit(
            f"Could not resolve --model '{name}': not 'example', not a "
            f"VLMEvalKit model, and not an importable module ({exc})."
        )
    fn = getattr(module, "model_generate", None) or getattr(module, "generate", None)
    if fn is None:
        raise SystemExit(
            f"Adapter module '{name}' must define model_generate(messages) -> str"
        )
    return fn
