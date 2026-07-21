# Track-3 harness — known issues & gotchas (address in the public release)

Every item below was hit while building/running this harness. Symptom → cause → fix/status.
Ordered by pipeline stage. Items marked **[handled]** are already fixed in our scripts (keep the fix
in the public version); **[watch]** are edge cases the public script should guard.

> The harness is **hardware- and engine-agnostic**: inference and judging both talk to an
> OpenAI-compatible model endpoint (`--endpoints`). Serve the VL model and the judge with whatever
> stack you have (e.g. `vllm serve <model>`); the evaluation code does not care what's behind the URL.

## A. Data acquisition — building the TSVs

1. **VLMEvalKit won't `pip install` on Apple Silicon** — `decord>=0.6.0` has no arm64 wheel. **[handled]**
   Don't install the full package for data-prep; download the built-in TSVs directly from each dataset's
   `DATASET_URL` (grep `vlmeval/dataset/image_mcq.py`). Custom loaders only need `pandas/PIL/huggingface_hub`.
2. **Python venv corrupts on exFAT** (`UnicodeDecodeError` in `site.py` at interpreter start). **[handled]**
   Put the venv on a native filesystem; data can live on exFAT.
3. **opencompass mirror TLS cert expired** (`opencompass.openxlab.space`) — 7 of 8 built-ins are there.
   **[handled]** Download with cert-verify off **and md5-verify** (hashes are in the same source file).
4. **HF gated datasets** (SpatialBench): a *fine-grained* token needs the "Read access to public gated repos"
   permission; a *classic Read* token works out of the box. **[handled]** Read the raw token robustly
   (`grep -oE 'hf_[A-Za-z0-9_-]+'`) — the token file may contain a shell wrapper like `HF_TOKEN="hf_..."`.
5. **`huggingface-cli` may be missing from PATH** — use `python -c "from huggingface_hub import
   snapshot_download; ..."` instead. **[handled]**

## B. Dataset parsing — the 13 → uniform MCQ TSV

6. **Custom-loader field maps differ from upstream docs** — always `--inspect` first. **[handled]** Specific:
   - **SAT-Real**: `--split test` = the 150 real Q (`val` is 4001 synthetic); image is `image_bytes` (multi-frame ndarray).
   - **OmniSpatial**: records have *no* image field → image = `{task_type}/{id.split('_')[0]}.png`; 229 `Complex_Logic` rows have `options:[]` (visual-select) → skipped.
   - **MindCube**: options are embedded in the question **stem** (`A. .. B. ..`) — parse them out; multi-image, base = `.../data/`.
   - **SpatialBench**: three formats — counting (numeric)→skip, existence (Yes/No)→A/B, positional/reach/size (`(A).. (B)..`)→parse; gated source.
   - **VSR**: recast Yes/No → MCQ (A.Yes / B.No).
7. **String indices** — BLINK keys rows by strings (`val_Art_Style_1`), not ints. **[handled]** Never `int(index)`;
   keep the index as-is (this crashed `make_manifest` and would crash `build_variants`).
8. **VLMEvalKit image dedup** — in circular TSVs (**3DSRBench**), the image is stored once; other rows put a
   short **index-string** in the `image` field that references the row holding the base64 (e.g. `image='0'`).
   **[handled]** Resolve the reference before decoding — else `PIL: cannot identify image file`. ~52% of
   3DSRBench rows are references.
9. **Circular eval + dataset sizes vs the paper** — **[handled]** SpatialBench & SAT-Real (the two the paper
   stars, Table 5) are scored with **circular evaluation**: each question is re-asked with options rotated to
   every position, correct only if right in ALL rotations (`CIRCULAR` set in run_track3_vllm.py expands the
   rows and shares a `group` id; judge_track3.py groups by it and requires all-correct). A question whose gt
   option is empty falls back to single-pass (1/152 in SpatialBench). Residual documented gaps: SpatialBench
   152/174 (dropped 20 numeric-counting + 2 bad-answer items for text-MCQ uniformity); **3DSRBench** — the
   public TSV is now a circular/flip-expanded release (11,686 rows = 3,997 base Qs), scored **base-only
   non-circular** to match the paper's non-circular treatment (can't reproduce their exact ~5.2K); use
   **MMSIBench_wo_circular** (1K) to match the paper's MMSIBench.

## C. Inference

10. **`max_new_tokens` vs context length** — **[handled]** Too-small ceilings (1024/2048) truncate the reasoning
    trace *before* `<answer>` and the answer is lost. But `max_tokens` + prompt must FIT the served context or
    strict servers reject the request outright — setting `max_tokens = context = 32768` rejects EVERY CoT call
    (silent 100% loss). Default `--max-tokens-cot 16384` never truncates real traces (they hit EOS far sooner);
    to use the paper's full 32768 budget, serve a context >32768 and raise the flag to match.
11. **Thinking models reason even under the "non-CoT" bare prompt** (`enable_thinking=False` does not suppress
    it). **[handled]** The paper's scenario, with TWO consequences: (a) non-CoT output is *not* terse → regex
    is unreliable, use the judge (D-15); and (b) **the non-CoT answer budget must be as large as the CoT one** —
    a thinking model's non-CoT trace is just as long, so a small `--max-tokens-noncot` silently truncates it
    before the answer and *depresses the non-CoT score* (we measured up to **53% of MindCube non-CoT answers**
    hitting a 2048 cap; this biases the CoT-vs-non-CoT delta toward "CoT helps"). Both budgets now default to
    16384. Confirmed on Qwen3-VL-8B-Thinking (~3K-char traces on the bare prompt).
12. **vLLM `--limit-mm-per-prompt` defaults to 1 image** → 400s on multi-image sets (MindCube 4, MMSIBench 2,
    SAT 2, BLINK up to 3). **[handled]** Set `{"image": 8}` (and `--max-model-len` to 32768).
13. **Multi-image parsing + delivery** — **[handled]**, and the *nastiest* bug in this list because it
    throws **no error**: VLMEvalKit stores multi-image rows as a **Python-repr list with single quotes**
    (`['/9j/…', '/9j/…']`), **not JSON**. `json.loads` fails on the single quotes → if you fall back to
    treating the cell as one image, every multi-image question **silently collapses to its first image**,
    and comparison tasks score ~random with no warning (BLINK's compare-2-images sub-tasks went to **~0–4%**,
    a 20-pt hit on the dataset average). Parse with `ast.literal_eval` (try `json.loads` then `ast`). Made
    worse by item 14: a lenient base64 decode of the *whole* list-string "succeeds" by concatenating the
    images and opening only the first. Deliver the parsed images as multiple `image_url` parts in the content.
14. **Base64 must be re-encoded clean** — some datasets (BLINK, MMSIBench) store **MIME-wrapped** base64
    (newlines/whitespace embedded). vLLM decodes data-URL base64 with `validate=True` (strict) →
    `binascii.Error: Non-base64 digit found` → the whole request 400s → **100% of that dataset lost**.
    **[handled]** Re-encode every image to clean standard base64 before building the data URL (lenient
    `b64decode` → `b64encode`). **This bug hides locally**: Python's lenient `b64decode` silently *discards*
    whitespace, so images "decode fine" in a test and the failure only appears at the strict server.
    (Mime label can stay `image/png`; the server sniffs the real format.)

## D. Scoring

15. **Regex extraction is insufficient for Track 3** (free-form/CoT output) — you **must** use the LLM judge.
    **[handled]** (Unlike the terse direct-answer Track 1/2, where regex ≡ judge.) An earlier `<answer>`-tag
    prompt hack was a deviation and was reverted.
16. **Judge = Qwen3-30B-A3B-Instruct-2507** (text) with the **verbatim Appx A.3 MCQ prompt**. **[handled]**
    Not a smaller/different judge (paper validated κ>0.99 vs GPT-4o).
17. **No-Image++ conflicts with judge rule 8** — **[watch]** the A.3 prompt says output `"0"` for "Cannot
    determine" responses (rule 8), but in No-Image++ *"Cannot determine from the image"* is a real **option**
    whose letter is the GT. The judge must map it via rule 5 (option-text match). A prose abstention with no
    letter could be under-counted — worth a targeted check/override.

## E. Faithfulness checklist (must match the paper — verify before publishing)

- Judge model + Appx A.3 prompt, verbatim.
- `base_noncot` = bare prompt; `cot_default` = base + `<think>/<answer>`; **dataset prompts for OmniSpatial &
  MindCube** prepended (Appx A.2).
- Question format: `Question:<q>\nOptions:\nA.<optA>\n...\nPlease select the correct answer (letter and option
  text) from the options above.`
- Greedy (temperature 0), `max_tokens`/context 32768.
- No-Image++ = gray same-size image + appended "Cannot determine from the image" = GT (MCQ rows only).
- Paper runs 3 seeds; greedy makes runs deterministic (single seed OK).

## F. Serving & orchestration (engine-agnostic)

- The eval drives any **OpenAI-compatible endpoint** — keep the serving layer separate. Serve the VL model
  and the judge (`--endpoints`), one replica per GPU for throughput, round-robined by the client.
- **Verify your serving engine's version actually supports the model architecture** before a long run — newer
  model families sometimes need a newer engine, or a transformers-based runner as a fallback.
- The **30B-A3B judge needs ~2 GPUs** (`--tensor-parallel-size 2`) — it won't fit on a single mid-memory GPU.
- Raise the server's `--max-model-len` to 32768 and `--limit-mm-per-prompt` to cover the max images/row (C-12).
- If you orchestrate with a job scheduler that passes env vars as comma-joined lists, a config **value
  containing a comma** (e.g. a dataset list) can be mis-split — pass single values or a file.
- Point any framework cache dirs (kernel/compile caches) at a **writable** location.

## G. Reproducibility

- The No-Image++ held-aside set is pinned two ways: `ablation_manifest.json` (indices, seed 20260706) +
  `noimgpp_frozen_probe.json` (per-sample content sha1 + base-TSV md5) — ship both so re-runs are provably
  identical and dataset drift is detectable.
