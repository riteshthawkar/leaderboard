"""
Builds the four evaluation conditions of the paper for a single sample.

| Condition       | Image  | Prompt change                                            |
|-----------------|--------|----------------------------------------------------------|
| ``standard``    | real   | question + options + "answer with the option letter"     |
| ``cot``         | real   | + "let's think step by step"                             |
| ``no_image``    | gray   | identical text, image replaced by a gray placeholder     |
| ``no_image_plus``| gray  | + an extra "Cannot determine from the image" option,     |
|                 |        | which is the *correct* answer (hallucination probe)      |

Messages use VLMEvalKit's interleaved format:
``[{"type": "image", "value": path}, ..., {"type": "text", "value": prompt}]``
so they can be passed straight to ``model.generate(messages)``.
"""

from pathlib import Path

CONDITIONS = ["standard", "cot", "no_image", "no_image_plus"]

_COT_SUFFIX = "Let's think step by step, then end with the final answer."
_MCQ_INSTRUCTION = "Answer with the option letter only."
_FREEFORM_INSTRUCTION = "Answer concisely with only the final answer."

# in-process cache of generated gray placeholders, keyed by (w, h)
_GRAY_CACHE = {}


def _next_letter(options):
    return chr(ord("A") + len(options))


def format_options(options):
    return "\n".join(f"{k}. {options[k]}" for k in sorted(options))


def gray_image_for(path, cache_dir):
    """Return a path to a solid-gray image matching ``path``'s dimensions."""
    from PIL import Image

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(path) as im:
            size = im.size
    except Exception:  # noqa: BLE001 - fall back to a default canvas
        size = (448, 448)

    if size not in _GRAY_CACHE:
        out = cache_dir / f"gray_{size[0]}x{size[1]}.png"
        if not out.exists():
            Image.new("RGB", size, (128, 128, 128)).save(out)
        _GRAY_CACHE[size] = str(out)
    return _GRAY_CACHE[size]


def build_text(question, options, condition, no_image_option):
    """Construct the prompt text and the effective option set for a condition."""
    opts = dict(options)
    if condition == "no_image_plus" and opts:
        opts[_next_letter(opts)] = no_image_option

    parts = [question]
    if opts:
        parts.append(format_options(opts))
        parts.append(_MCQ_INSTRUCTION)
    else:
        if condition == "no_image_plus":
            parts.append(f'If the image is missing, answer: "{no_image_option}".')
        parts.append(_FREEFORM_INSTRUCTION)

    if condition == "cot":
        parts.append(_COT_SUFFIX)
    return "\n".join(parts), opts


def build_messages(sample, condition, no_image_option, gray_dir):
    """Return ``(messages, effective_options)`` for one sample + condition."""
    text, opts = build_text(
        sample.get("question", ""), sample.get("options") or {},
        condition, no_image_option,
    )

    images = sample.get("images") or []
    if condition in ("no_image", "no_image_plus"):
        images = [gray_image_for(p, gray_dir) for p in images]

    messages = [{"type": "image", "value": p} for p in images]
    messages.append({"type": "text", "value": text})
    return messages, opts
