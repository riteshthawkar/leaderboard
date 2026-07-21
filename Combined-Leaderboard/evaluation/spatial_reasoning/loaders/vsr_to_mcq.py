#!/usr/bin/env python3
"""
vsr_to_mcq.py — recast VSR (Visual Spatial Reasoning) from Yes/No into MCQ.

VSR items are a true/false judgement: an image + a caption/statement, labelled true or false.
The paper standardizes ALL 13 benchmarks to MCQ, and No-Image++ can only inject a labelled
option into an MCQ. So we recast each VSR item to:

    question: <original statement/question>
    A. True (Yes)
    B. False (No)
    answer : A if the statement is true of the image else B

No-Image++ will later append "C. Cannot determine from the image" as the ground truth.

Input: a VLMEvalKit VSR Yes/No TSV (columns: index, image[b64], question, answer in {Yes,No,...}).
       If VSR is not in your VLMEvalKit install, export it to that TSV shape from the source
       dataset (cambridgeltl/vsr_*: image + caption + label 0/1) first.

Usage:
  python vsr_to_mcq.py --tsv ~/LMUData/VSR.tsv --out ~/LMUData/VSR_MCQ.tsv
  # then use --data VSR_MCQ everywhere (config.yaml, make_manifest, run_track3)
"""
import argparse, os, sys
import pandas as pd
from common import yesno_to_mcq, rows_to_tsv

INSTRUCTION = "Based on the image, is the following statement true?"


def recast(tsv_path, out_path, wrap_instruction=True):
    df = pd.read_csv(tsv_path, sep="\t")
    need = {"index", "image", "question", "answer"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"{tsv_path} missing columns {missing}; not a VSR Yes/No TSV")
    rows = []
    skipped = 0
    for _, r in df.iterrows():
        opts, letter = yesno_to_mcq(r["answer"])
        if letter is None:
            skipped += 1
            continue
        stmt = str(r["question"]).strip()
        q = f"{INSTRUCTION} {stmt}" if wrap_instruction else stmt
        row = {"index": int(r["index"]), "image": r["image"], "question": q, "answer": letter}
        row.update(opts)
        if "category" in df.columns:
            row["category"] = r["category"]
        rows.append(row)
    n, path = rows_to_tsv(rows, out_path)
    print(f"VSR -> MCQ: wrote {n} rows ({skipped} skipped) -> {path}  (use --data {os.path.basename(path)[:-4]})")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=os.path.expanduser("~/LMUData/VSR.tsv"))
    ap.add_argument("--out", default=os.path.expanduser("~/LMUData/VSR_MCQ.tsv"))
    ap.add_argument("--no-instruction", action="store_true", help="keep the raw statement as the question")
    a = ap.parse_args()
    recast(a.tsv, a.out, wrap_instruction=not a.no_instruction)
