# Track-3 (Spatial-CoT eval) — changes to apply: **v1 → v2**

**Read this whole file, then apply the changes to the Track-3 package you already have** (the
evaluation script `run_track3_vllm.py`, the judge `judge_track3.py`, and the data-loading scripts
under `loaders/`). Goal of the package: a submitter downloads the 13 spatial datasets, runs the
eval, and gets the final leaderboard numbers to submit.

Each change below gives the **exact final code** for the affected function(s). The safest way to
apply is: *replace the named function/block entirely with the version shown here.* The v2 code is
**backward-compatible** — it reads old TSVs (which have no `answer_type` column) as MCQ, so nothing
breaks if a step is applied out of order.

---

## TL;DR — what changed and why

| # | Change | Why | Effect on the numbers |
|---|--------|-----|-----------------------|
| 1 | **Removed the frozen No-Image++ sample** | The paper runs the No-Image++ ablation on the **full** valid sets (Table 3), not a 100/dataset frozen subset. The frozen file was a leaderboard-only add-on with no anti-gaming value for public benchmarks. | None (No-Image++ already ran on the full set). Just deletes 2 files + a doc section. |
| 2 | **v2: keep non-MCQ questions + add the paper's VQA judge** | The paper reports `main` accuracy on the FULL datasets and uses **two** judge prompts (Appx A.3): MCQ **and** VQA. v1 dropped every non-MCQ (numeric / free-form) item and used only the MCQ judge. | SpatialBench **152 → 174**, OmniSpatial **1304 → 1533**. |
| 3 | **SpatialBench parser fix** | The old option-parser silently dropped 2 "positional" items whose source has a mislabeled duplicate marker `(A)…(A)` and an instruction string jammed into the option text. | Included in the 174 above (recovers the last 2 items). |

> **Two-part rule of thumb:** *code* changes (below) make the harness *able* to keep + score non-MCQ
> items; a **data rebuild** (last section) is what actually *pulls those items in*. v2 code + old data
> = still MCQ-only. You need both.

---

## Change 1 — Remove the frozen No-Image++ set

1. **Delete** these files if present: `ablation_manifest.json`, `noimgpp_frozen_probe.json`,
   `freeze_probe.py`, `ablation/make_manifest.py`.
2. The runner must drive the eval with an **explicit 13-dataset `--datasets` list**, not `--manifest`.
   This is the exact list (it is the `DATASETS=` default in `run_eval.sh`; the names must match your
   TSV filenames):
   ```
   BLINK,CV-Bench-2D,CV-Bench-3D,MMVP,RealWorldQA,VStarBench,MMSIBench_wo_circular,3DSRBench,VSR_MCQ,SpatialBench,MindCube,OmniSpatial,SAT-Real
   ```
3. **Docs:** in the README, replace any "No-Image++ — fixed question set" section with a one-line note
   that No-Image++ runs on the **full valid set**; remove references to the two deleted JSON files.

No behavior change — the eval already ran the ablation on the full sets. This is cleanup only.

---

## Change 2 — v2: keep non-MCQ questions in `main`, and score them with the VQA judge

**How it works:** every question now carries an `answer_type` = `mcq` | `vqa`.
- `mcq` items behave exactly as before (options + MCQ judge → option letter).
- `vqa` items (numeric / free-form) carry the answer as **text**, appear only in `main × {CoT, non-CoT}`
  (they are **skipped in No-Image++**, which injects an option and therefore needs an MCQ), and are
  scored by the paper's **VQA judge** (returns `1` match / `0` no-match).

Apply the following to each file.

### 2a. `loaders/common.py`

**Replace `intermediate_to_row`** so that a non-MCQ item (fewer than 2 options) is *kept* as a VQA row
instead of being dropped, as long as it has a usable text answer:

```python
def intermediate_to_row(idx, inter, index_base=0):
    """
    inter = {image, question, options, answer, category}
      image   : PIL | path | bytes | b64
      options : list | dict | None (None => keep as-is, e.g. already yes/no handled by caller)
      answer  : letter | index | text   (index_base declares 0- or 1-based for integer answers)
    Returns a VLMEvalKit row dict, or None if the answer can't be resolved (caller should skip+count).
    """
    opt_dict, letters = normalize_options(inter.get("options"))
    if len(opt_dict) < 2:                                 # not MCQ -> keep as a VQA row if it has a usable answer
        ans = inter.get("answer")
        if ans is None or str(ans).strip() == "":
            return None, "no_answer"                      # genuinely unusable (e.g. visual-select w/ no text answer)
        row = {"index": idx, "image": to_b64(inter["image"]),
               "question": str(inter.get("question", "")).strip(),
               "answer": str(ans).strip(), "answer_type": "vqa"}
        if inter.get("category") is not None:
            row["category"] = inter["category"]
        return row, None
    # ... (the rest of the function — the MCQ path — is unchanged; it must set "answer_type": "mcq" on the row it returns)
```

> In the MCQ branch (the part after the block above), make sure the returned row dict includes
> `"answer_type": "mcq"`. That is the only change needed there.

**Replace `rows_to_tsv`** so the TSV header carries the new `answer_type` column:

```python
def rows_to_tsv(rows, path):
    """rows: list of dicts already in VLMEvalKit column form. Orders columns and writes TSV."""
    df = pd.DataFrame(rows)
    head = ["index", "image", "question"] + [c for c in OPTS if c in df.columns] + ["answer", "answer_type"]
    cols = [c for c in head if c in df.columns] + [c for c in df.columns if c not in head]
    df = df[cols]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    return len(df), path
```

### 2b. `run_track3_vllm.py`

**(i)** Near the top, alongside the other dataset constants, keep these two sets (they already exist in
v1 if you have the circular-eval fix; shown here for completeness):

```python
CIRCULAR_BASE_ONLY = {"3DSRBench"}          # one row per base question (drop the flip duplicates)
CIRCULAR = {"SpatialBench", "SAT-Real"}     # paper stars these: options rotated to every position
```

**(ii) Replace `build_question`** so it emits a VQA prompt when there are no options:

```python
def build_question(stem, optmap, dataset):
    lines = [f"Question:{stem}"]
    if optmap:                                            # MCQ
        lines.append("Options:")
        for L in sorted(optmap):
            lines.append(f"{L}.{optmap[L]}")
        lines.append("Please select the correct answer (letter and option text) from the options above.")
    else:                                                 # VQA (no options) — elicit a short direct answer
        lines.append("Answer the question directly with a short final answer.")
    q = "\n".join(lines)
    pre = DATASET_PROMPT.get(dataset)
    return (pre + "\n" + q) if pre else q
```

**(iii) Replace `build_records`** — this is the core of v2. It reads the `answer_type` column, routes
VQA items to a single text-answer record (skipped in No-Image++), and tags every record with
`answer_type` and a `group` id (the group id is what makes circular all-rotations-correct scoring work):

```python
def build_records(lmudir, dataset, mode, limit=0):
    df = pd.read_csv(os.path.join(lmudir, dataset + ".tsv"), sep="\t")
    # VLMEvalKit image dedup: some rows (e.g. 3DSRBench circular) store the image only once; other
    # rows put a short index-string in `image` that references the row holding the real base64.
    img_by_index = {str(r["index"]): str(r["image"]) for _, r in df.iterrows() if len(str(r["image"])) > 64}
    if dataset in CIRCULAR_BASE_ONLY and "qid" in df.columns:   # one row per base question (non-circular)
        def _base(q):
            q = str(q)
            for suf in ("-flip-1", "-flip", "-1"):
                if q.endswith(suf):
                    return q[:-len(suf)]
            return q
        df = df.assign(_b=df["qid"].map(_base)).drop_duplicates("_b").drop(columns="_b").reset_index(drop=True)
    if limit:
        df = df.head(limit)
    recs = []
    for _, row in df.iterrows():
        b = str(row["image"])
        if len(b) < 32 and b in img_by_index:          # reference -> resolve to the real image
            b = img_by_index[b]
        imgs = image_cells(b)
        stem = str(row["question"]).strip()
        base = str(row["index"])
        opts = present_opts(row)
        at = row.get("answer_type")                    # explicit column on custom TSVs; hosted TSVs are all MCQ
        atype = str(at).strip().lower() if (at is not None and pd.notna(at)) else ("mcq" if len(opts) >= 2 else "vqa")

        if atype == "vqa":                             # free-form / numeric item -> VQA judge, gt is the answer text
            if mode == "noimgpp":                      # No-Image++ injects an OPTION -> MCQ-only; skip VQA items
                continue
            recs.append(dict(dataset=dataset, index=base, group=base, answer_type="vqa",
                             question=build_question(stem, {}, dataset), options={},
                             imgs=imgs, gray=False, gt=str(row["answer"]).strip(), cannot_label=None))
            continue

        optmap = {L: str(row[L]).strip() for L in opts}
        gt = str(row["answer"]).strip().upper()
        gray = (mode == "noimgpp")
        # layouts: list of (record-index, optmap, correct-letter, cannot-label); every record shares `group`=base
        if mode == "noimgpp":                          # ablation: append "Cannot determine" = GT (single pass)
            nl = OPTS[len(opts)]
            om = dict(optmap); om[nl] = CANNOT
            layouts = [(base, om, nl, nl)]
        elif dataset in CIRCULAR and gt in optmap:      # paper stars these: rotate options to every position
            layouts = [(f"{base}_r{r}", rot, cor, None)
                       for r, (rot, cor) in enumerate(circular_rotations(optmap, gt))]
        else:
            layouts = [(base, dict(optmap), gt, None)]
        for idx, om, g, cannot in layouts:
            recs.append(dict(dataset=dataset, index=idx, group=base, answer_type="mcq",
                             question=build_question(stem, om, dataset),
                             options=om, imgs=imgs, gray=gray, gt=g, cannot_label=cannot))
    return recs
```

**Dependencies of `build_records`:** it uses core helpers that already exist in v1
(`present_opts`, `image_cells`, `data_url`, `DATASET_PROMPT`) plus these constants and the
circular-eval helper. If your v1 predates the circular fix, add them (near the top of the file);
if they already exist, leave them:

```python
import string
OPTS   = list(string.ascii_uppercase)
CANNOT = "Cannot determine from the image"

def circular_rotations(optmap, gt):
    """Circular eval: yield (rotated_optmap, correct_letter) for every cyclic shift of the option TEXTS.
    The letters stay A,B,C,...; only which text sits at each letter rotates, so the model must track the
    right answer across all positions. len(options) variants per question."""
    letters = sorted(optmap)
    texts = [optmap[L] for L in letters]
    n = len(letters)
    ai = letters.index(gt)
    for r in range(n):
        rot = {letters[i]: texts[(i - r) % n] for i in range(n)}
        yield rot, letters[(ai + r) % n]
```

**(iv)** In `main()`, the per-row **prediction writer** must persist `group` and `answer_type` so the
judge can route + aggregate. The written dict is:

```python
with open(os.path.join(a.out, f"pred_{tag}.jsonl"), "w") as f:
    for it in recs:
        f.write(json.dumps({k: it.get(k) for k in
                ("dataset", "index", "group", "answer_type", "options", "gt", "cannot_label")}
                | {"mode": mode, "pmode": pmode, "output": it.get("output")}) + "\n")
```

**(v) Token-budget defaults** (important for correct numbers — do **not** leave non-CoT at a small
value, or thinking models get truncated and score low):

```python
ap.add_argument("--max-tokens-noncot", type=int, default=16384)   # must equal the CoT budget
ap.add_argument("--max-tokens-cot",    type=int, default=16384)   # 16384 never truncates real CoT
```

### 2c. `judge_track3.py`

**(i)** Add the paper's **VQA-scoring** prompt and its two helpers next to the existing MCQ ones
(`JUDGE_SYS`, `judge_user`, `parse_letter`). This block is verbatim from the paper, Appx A.3:

```python
# ---- paper Appx A.3 "VQA Scoring" system prompt, verbatim (for non-MCQ / free-form / numeric answers) ----
JUDGE_VQA = (
    "You are a helpful assistant.\n\n Task: Given a short free-form \"Response\" and a gold-standard \"Gold\", "
    "decide if the Response expresses the SAME answer as Gold. Output \"1\" for match, \"0\" otherwise.\n\n "
    "Inputs:\n - Gold: the gold-standard answer which is either (i) a short phrase, (ii) an integer, or (iii) "
    "\"Yes\"/\"No\".\n - Response: a few words or a short phrase, possibly will include reasoning steps before "
    "the final answer.\n\n Output format:\n - STRICTLY OUTPUT EXACTLY ONE CHARACTER: \"1\" if matching, \"0\" "
    "if not.\n - Do not output any explanation, spaces, punctuation, or additional text.\n\n Rules:\n 1) Compare "
    "only the final answer in the Response to Gold. Ignore any reasoning steps or intermediate answers present "
    "in the Response.\n 2) If multiple conflicting answers or uncertainty like \"I don't know\" appear in the "
    "Response, output \"0\".\n 3) Do not use external knowledge; judge only based on the text in Gold and "
    "Response.\n 4) Punctuation, grammar, and minor spelling errors should be ignored.\n - uppercase/lowercase "
    "differences should be ignored.\n - hyphen and underscore are ignored. For ex, \"double-bus\" and \"double "
    "bus\" are considered the same.\n - synonyms of \"Yes\"/\"No\" like \"Y\"/\"N\", \"True\"/\"False\" must be "
    "considered the same.\n - word representations of numbers like \"one\"/\"two\"/\"three\" must be considered "
    "the same as \"1\"/\"2\"/\"3\".\n 5) Core concept and critical attributes must match. For example, \"New "
    "York City\" and \"New York State\" do not match. Other examples of non-matches are \"bus\" vs \"double "
    "bus\"; \"red\" vs \"light red\"; \"dog\" vs \"golden retriever\"; \"apple\" vs \"green apple\".\n 6) If the "
    "response says \"I don't know\", \"Cannot determine\", or similar, output \"0\".\n\n Examples:\n - Gold: "
    "Double Bus | Response: This is a bus -> 0\n - Gold: Double Bus | Response: I can see a double-bus -> 1\n - "
    "Gold: Yes | Response: Y -> 1\n - Gold: 10 | Response: ten -> 1\n - Gold: red | Response: light red -> 0\n - "
    "Gold: stop sign | Response: a stop sign on a pole -> 1\n - Gold: person | Response: man -> 0\n\n Now read "
    "the following Gold and Response and output exactly one character: \"1\" or \"0\".\n")


def judge_user_vqa(it):
    return f"Gold: {it.get('gt')}\nResponse: {it.get('output') or ''}"


def parse_bit(s):
    s = (s or "").strip()
    return s[0] if (s and s[0] in "01") else None
```

**(ii) Replace `judge_one`** so it routes by `answer_type` (VQA items → `JUDGE_VQA` + `judge_user_vqa`
+ `parse_bit`; everything else → the existing MCQ path):

```python
async def judge_one(client, sem, it, model):
    vqa = (it.get("answer_type") == "vqa")             # non-MCQ item -> paper's VQA scorer (match/no-match)
    async with sem:
        try:
            r = await client.chat.completions.create(
                model=model, temperature=0, max_tokens=4,
                messages=[{"role": "system", "content": JUDGE_VQA if vqa else JUDGE_SYS},
                          {"role": "user", "content": judge_user_vqa(it) if vqa else judge_user(it)}])
            content = r.choices[0].message.content
            it["judged"] = parse_bit(content) if vqa else parse_letter(content)
        except Exception as e:
            it["judged"], it["jerr"] = None, str(e)[:120]
    return it
```

**(iii) Replace `correct`** so a VQA item is correct when the judge returns `"1"`:

```python
def correct(it):
    j = it.get("judged")
    if j is None:
        return None
    if it.get("answer_type") == "vqa":                 # VQA judge outputs "1" (match) / "0"
        return j == "1"
    if it["mode"] == "noimgpp":
        return j == str(it.get("cannot_label") or "").upper()
    return j == str(it["gt"]).upper()
```

**(iv)** The aggregation groups by `(dataset, mode, pmode, group)` and marks a group correct only if
**all** its members are correct (this handles both single questions and circular all-rotations items).
If your v1 already has group-aware aggregation (from the circular fix), leave it. Otherwise it is:

```python
groups = collections.defaultdict(list)
for it in all_items:
    groups[(it["dataset"], it["mode"], it["pmode"], it.get("group", it["index"]))].append(correct(it))
# then: a group counts as one unit; correct iff all(members); drop the group if any member has judged=None
```

---

## Change 3 — SpatialBench parser fix (recovers the last 2 items → 174)

In `loaders/prepare_custom.py`, **replace `_split_paren_options`** with the version below. It (1) strips
an `"Answer with the option's letter…"` instruction that the source jams into the stem *and* into
option text, and (2) relabels a **duplicate/mislabeled** option marker (a 4th option tagged `(A)`
instead of `(D)`) to the next free letter — which is exactly what the 2 dropped positional items had:

```python
def _split_paren_options(q):
    """Split 'stem ... (A) x (B) y' -> (stem, {A:x, B:y}); (q, None) if <2 markers.
    Handles two SpatialBench-positional source quirks: (1) an 'Answer with the option's letter ...'
    instruction jammed into the stem AND into option text, and (2) a mislabeled/duplicate option
    marker (a 4th option tagged '(A)' instead of '(D)') -> relabel duplicates to the next letter."""
    marks = list(re.finditer(r'\(([A-H])\)\s*', q))
    if len(marks) < 2:
        return q.strip(), None
    instr = r"Answer with the option'?s letter[^.\n]*\.?"
    stem = re.sub(r"\s*" + instr + r"\s*$", "", q[:marks[0].start()].strip(), flags=re.I).strip()
    opts = {}
    for i, m in enumerate(marks):
        letter = m.group(1)
        if letter in opts:                                    # source mislabel/dup -> next free letter
            letter = "ABCDEFGH"[len(opts)]
        start, end = m.end(), (marks[i + 1].start() if i + 1 < len(marks) else len(q))
        text = re.sub(r"\s*" + instr + r"\s*", " ", q[start:end], flags=re.I).strip().rstrip(".")
        opts[letter] = text
    return stem, opts
```

---

## After applying the code changes: rebuild the data, then run

The code lets the harness *keep + score* non-MCQ items; the **data rebuild** is what pulls them in.
SpatialBench is **gated**, so a Hugging Face token is required.

```bash
# 1) force a rebuild of the two datasets whose counts change (delete only these, keep the rest cached)
rm -f LMUData/SpatialBench.tsv LMUData/OmniSpatial.tsv

# 2) rebuild (HF token needed; accept SpatialBench terms on its HF page first)
python prepare_data.py --lmudata ./LMUData --hf-token hf_xxx

# 3) run the eval (inference + judge) on the full 13-dataset list, e.g.
./run_eval.sh <model> <vl_endpoint> <judge_endpoint>
```

**Do users need to re-run their model from scratch?** No. The MCQ questions are byte-identical to v1,
so existing MCQ predictions are reused. Only the **new non-MCQ items** need inference (a small
incremental run), then VQA-judge them and merge. Only `main` accuracy for SpatialBench / OmniSpatial
moves; No-Image++ is untouched.

---

## Verify it worked

- `ablation_manifest.json` and `noimgpp_frozen_probe.json` are **gone**.
- `judge_track3.py` contains `JUDGE_VQA`; `run_track3_vllm.py` and `loaders/common.py` reference `answer_type`.
- After the rebuild, `SpatialBench.tsv` has **174** rows and an `answer_type` column containing **both**
  `mcq` and `vqa`; `OmniSpatial.tsv` has **1533** rows.
- A judged run produces `leaderboard.json` with `main_noncot`, `main_cot`, `main_delta`,
  `npp_noncot`, `npp_cot` per dataset.
