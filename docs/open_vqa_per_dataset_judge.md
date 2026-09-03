# Open-VQA per-dataset label-set judge

Operational reference for the **live** MMBU open-ended answer-equivalence judge. This is
**answer correctness only** — not the `metadata_cot` metadata-field judge.

Historical design notes live in [`judge_upgrade_plan.md`](./judge_upgrade_plan.md) and
[`judge_prompt_and_parsing.md`](./judge_prompt_and_parsing.md). Those documents describe
the upgrade program; this file describes what is **currently wired** in `python -m mmbu judge`.

## Paradigm (two gates)

Each open-VQA row is graded with prompt variant **`open_per_dataset_v3`**
(`OPEN_VQA_PER_DATASET_PROMPT_V3` in [`src/mmbu/eval/judge_prompts_v2.py`](../src/mmbu/eval/judge_prompts_v2.py)).

The judge sees:

1. **Dataset label set** — all valid `class_label` values for that dataset, built from the 8-task
   open TSVs (`cls_open`, `det_grounding_open`, `seg_grounding_open`).
2. **Equivalence policy** — taxonomy-specific rules (see below).
3. **Question, true answer, model answer** — extracted answer text only; no images.

**Gate 1 — Grounding:** Does the model answer unambiguously assert **one** concept from the label
set (or an unambiguous synonym)? Empty, vague, hedged, or off-taxonomy answers → `[RESULT] 0`.

**Gate 2 — Equivalence:** If Gate 1 passes, is the grounded concept equivalent to the true answer
under the policy? Broader/narrower/parent-subtype/related-but-distinct → `[RESULT] 0`.

Output format (one line):

```text
Rationale: <brief rationale naming the grounded concept> [RESULT] <0 or 1>
```

## Taxonomy types

Built by [`src/mmbu/eval/dataset_label_taxonomy.py`](../src/mmbu/eval/dataset_label_taxonomy.py)
and cached at [`artifacts/dataset_label_taxonomy.json`](../artifacts/dataset_label_taxonomy.json).

| Type | When | Policy gist |
|------|------|-------------|
| `binary_tumor` | pos/neg tumor sides both present | Do not collapse tumor vs not-tumor sides |
| `receptor_infection` | receptor / infection / DR staging labels | Positive/negative are biomarker-specific, not generic malignancy |
| `multiclass` | ≥3 distinct classes | Do not collapse distinct diagnoses (e.g. benign keratosis ≠ normal) |
| `other` | fallback | Same specificity or unambiguous synonym only |

Regenerate taxonomy after TSV label changes:

```bash
python -c "from mmbu.eval.dataset_label_taxonomy import write_taxonomy_artifact; write_taxonomy_artifact()"
```

Use the **full 8-task** TSVs — not v4-only release TSVs — so label sets stay stable.

## Judge models

| Judge | Backend | System prompt | Notes |
|-------|---------|---------------|-------|
| `Qwen/Qwen2.5-32B-Instruct-AWQ` | vLLM GPU | (in user prompt) | Production default for bulk cache |
| `claude-sonnet-5` | Anthropic API / Batch | `GOLD_ADJUDICATION_SYSTEM_PROMPT` | Recommended API adjudicator |
| `claude-opus-5` | Anthropic API / Batch | `OPUS_JUDGE_SYSTEM_PROMPT` | Stricter format contract; higher `max_new_tokens` |

Anthropic caches are **not** mixed with Qwen rows (`is_anthropic_model` in
[`llm_judge_cache_compat.py`](../src/mmbu/eval/llm_judge_cache_compat.py)).

## Cache layout

```text
<MMBU_JUDGE_CACHE>/open_vqa/<sanitized_judge_model>/<task>/<answer_model>.csv
```

Example:

```text
…/llm_judge_cache/open_vqa/claude-sonnet-5/classification_open_VQA_cot/gpt-5.6-sol.csv
```

Cache keys include `prompt_hash` for `open_per_dataset_v3`. Resume is safe: re-running skips
rows with `llm_judge_status=ok`.

Paths: `from mmbu.paths import judge_cache_open`.

## CLI (full 8-task merge)

Default prompt variant is already `open_per_dataset_v3`:

```bash
export PYTHONPATH=src
python -m mmbu judge \
  --kind open \
  --judge-model claude-sonnet-5 \
  --prompt-variant open_per_dataset_v3 \
  --model-filter gpt-5.6-sol \
  --cache-dir "$(python -c 'from mmbu.paths import judge_cache_open; print(judge_cache_open())')"
```

Inspect progress:

```bash
python -m mmbu judge --kind open --judge-model claude-sonnet-5 --status-json
```

## v4 public/private subset

For the frozen v4 eval split, use the dedicated wrapper (does **not** judge the full benchmark):

```bash
python scripts/run_sonnet_v4_open_rejudge.py --status-json
python scripts/run_sonnet_v4_open_rejudge.py   # fill missing Sonnet rows only
python scripts/retry_sonnet_unparsed_judge.py  # retry unparsed Anthropic rows
```

PDF report after judging:

```bash
python -m mmbu.v4_sonnet_report --rejoin
```

Or the full Pasteur orchestration:

```bash
bash scripts/run_v4_sonnet_rejudge_pasteur.sh
```

## Not in scope

- **metadata_cot** — separate Sonnet batch for modality/submodality fields (`metadata_judge_model`).
- **Closed VQA** — uses `CLOSED_VQA_PROMPT_V2` (letter-first); no per-dataset label set.
- **Segmentation multilabel** — closed/seg tasks use set-equivalence rubric, not label-set grounding.
- **det_guess_bbox_open** — IoU direct score, no LLM judge.

## Related artifacts

- Judge agreement pack: [`artifacts/llm_judge/README.md`](../artifacts/llm_judge/README.md)
- Experiment arm registry: `open_per_dataset_v3` in `EXPERIMENT_ARMS` (`judge_prompts_v2.py`)
