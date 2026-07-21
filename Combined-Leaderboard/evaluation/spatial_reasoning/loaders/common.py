#!/usr/bin/env python3
"""
common.py — shared helpers for turning any spatial dataset into a VLMEvalKit MCQ TSV.

Every custom loader (prepare_custom.py, vsr_to_mcq.py) funnels its dataset-specific records
through these normalizers, so the OUTPUT is always the uniform VLMEvalKit schema that the rest
of the harness (make_manifest -> build_variants -> run) already understands:

    index | image(b64) | question | A | B | C | ... | answer(letter) | category

The only per-dataset work a loader must do is map raw fields -> the intermediate dict
{image, question, options, answer, category}. Everything below is dataset-agnostic and tested.
"""
import base64, io, os, string
import pandas as pd
from PIL import Image

OPTS = list(string.ascii_uppercase)


# ---------- images ----------
def pil_to_b64(img, fmt="PNG"):
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def to_b64(image):
    """Accept a PIL image, path, raw bytes, base64 string, an HF Image dict {'bytes'|'path'},
    or a LIST of any of these (multi-image rows -> JSON list of base64, VLMEvalKit's format)."""
    import json
    if isinstance(image, (list, tuple)):
        return json.dumps([to_b64(x) for x in image])
    if isinstance(image, dict):                          # HF datasets Image feature
        if image.get("bytes"):
            return pil_to_b64(Image.open(io.BytesIO(image["bytes"])))
        if image.get("path"):
            return pil_to_b64(Image.open(image["path"]))
    if isinstance(image, Image.Image):
        return pil_to_b64(image)
    if isinstance(image, (bytes, bytearray)):
        return pil_to_b64(Image.open(io.BytesIO(image)))
    s = str(image)
    if os.path.exists(s):
        return pil_to_b64(Image.open(s))
    return s  # assume it is already base64


# ---------- options / answers ----------
def normalize_options(options):
    """
    options -> (ordered dict {letter: text}, list of letters).
    Accepts:
      * list  ["dog", "cat", ...]               -> A,B,...
      * dict  {"A": "dog", "B": "cat"}           -> kept, labels stripped
      * dict  {"a": "...", ...} / trailing punct -> normalized to A,B,...
    Any leading inline label ("A. ", "B) ") in the text is stripped so we don't double-label.
    """
    if not options:
        return {}, []
    if isinstance(options, dict):
        items = [(str(k).strip().rstrip(").:").upper(), v) for k, v in options.items()]
    else:
        items = [(OPTS[i], v) for i, v in enumerate(options)]
    out = {}
    for letter, text in items:
        t = str(text).strip()
        if len(t) >= 2 and t[0].upper() in OPTS and t[1] in ").:":   # strip "A. " / "A)"
            t = t[2:].strip()
        out[letter] = t
    return out, list(out.keys())


def answer_to_letter(answer, opt_dict, index_base=0):
    """
    Map an answer given as a letter / integer index / option text -> option letter.
    `index_base` is EXPLICIT (0 or 1) — integer answers are ambiguous and each dataset
    differs, so the loader must declare it rather than us guessing (avoids a silent off-by-one).
    """
    if answer is None:
        return None
    a = str(answer).strip()
    letters = list(opt_dict)
    up = a.rstrip(").:").upper()
    if up in opt_dict:                                    # already a letter
        return up
    if a.lstrip("-").isdigit():                           # integer index (base declared by caller)
        i = int(a) - index_base
        return letters[i] if 0 <= i < len(letters) else None
    for L, t in opt_dict.items():                         # match option text
        if a.lower() == str(t).lower():
            return L
    return None


def yesno_to_mcq(answer):
    """'yes'/'no'/True/False/1/0 -> ({'A':'Yes','B':'No'}, letter)."""
    a = str(answer).strip().lower()
    yes_values = {"yes", "y", "true", "t", "1"}
    no_values = {"no", "n", "false", "f", "0"}
    if a in yes_values:
        letter = "A"
    elif a in no_values:
        letter = "B"
    else:
        letter = None
    return {"A": "Yes", "B": "No"}, letter


# ---------- output ----------
def rows_to_tsv(rows, path):
    """rows: list of dicts already in VLMEvalKit column form. Orders columns and writes TSV."""
    df = pd.DataFrame(rows)
    head = ["index", "image", "question"] + [c for c in OPTS if c in df.columns] + ["answer"]
    cols = [c for c in head if c in df.columns] + [c for c in df.columns if c not in head]
    df = df[cols]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    return len(df), path


def intermediate_to_row(idx, inter, index_base=0):
    """
    inter = {image, question, options, answer, category}
      image   : PIL | path | bytes | b64
      options : list | dict | None (None => keep as-is, e.g. already yes/no handled by caller)
      answer  : letter | index | text   (index_base declares 0- or 1-based for integer answers)
    Returns a VLMEvalKit row dict, or None if the answer can't be resolved (caller should skip+count).
    """
    opt_dict, letters = normalize_options(inter.get("options"))
    if len(opt_dict) < 2:                                 # need >=2 options to be MCQ (matches manifest eligibility)
        return None, "not_mcq"
    letter = answer_to_letter(inter.get("answer"), opt_dict, index_base)
    if letter is None:
        return None, "unresolved_answer"
    row = {"index": idx, "image": to_b64(inter["image"]),
           "question": str(inter.get("question", "")).strip(), "answer": letter}
    row.update(opt_dict)
    if inter.get("category") is not None:
        row["category"] = inter["category"]
    return row, None
