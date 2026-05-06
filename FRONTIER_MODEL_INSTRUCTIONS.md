# Running Frontier API Models

This repo can run the same VLM evaluation pipeline with OpenAI or Gemini API models through the `openai` and `gemini` model adapters.

The workflow and output structure are the same as the local Hugging Face model runs:

- The runner is still `src/run_vlm_eval.py`.
- The dataset/task config is still passed with `--config`.
- The model backend is still selected with `--type`.
- The model name is still selected with `--name`.
- Results are still written as JSONL under the configured `runtime.output_dir`.

The API example configs list all 10 evaluation tasks:

- `configs/openai_api_example.yaml`
- `configs/gemini_api_example.yaml`

## OpenAI

Set your API key:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

Run with the OpenAI adapter:

```bash
python src/run_vlm_eval.py \
  --config configs/openai_api_example.yaml \
  --type openai \
  --name gpt-4o
```

You can swap `gpt-4o` for another OpenAI vision-capable model if needed.

To run a 10-example smoke test before the full evaluation, add `--test`:

```bash
python src/run_vlm_eval.py \
  --config configs/openai_api_example.yaml \
  --type openai \
  --name gpt-4o \
  --test
```

## Gemini

Set your API key:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

Run with the Gemini adapter:

```bash
python src/run_vlm_eval.py \
  --config configs/gemini_api_example.yaml \
  --type gemini \
  --name gemini-1.5-pro
```

You can swap `gemini-1.5-pro` for another Gemini multimodal model if needed.

To run a 10-example smoke test before the full evaluation, add `--test`:

```bash
python src/run_vlm_eval.py \
  --config configs/gemini_api_example.yaml \
  --type gemini \
  --name gemini-1.5-pro \
  --test
```

## Recommended API Settings

API-backed models are called one sample at a time inside the adapter. For early testing, use `batch_size: 1` to avoid rate-limit surprises:

```yaml
runtime:
  batch_size: 1
  max_new_tokens: 1024
  save_every: 50
  log_first_batch: true
  output_dir: "/pasteur/u/rdcunha/code/mmbu/results"
```

The `--test` flag is only supported for the frontier API adapters (`openai` and `gemini`). It limits the run to at most 10 examples total across the configured tasks.

Because the API example configs include all 10 tasks, use `--test` first to validate the full all-task configuration cheaply before removing `--test` for the complete run.

## Output

The output format is unchanged. Each JSONL row keeps the same fields used by local model runs, including `index`, `question`, `image_path`, `modality`, `answer`, and any available task metadata.

Output files are written under:

```text
<runtime.output_dir>/<model_name>/<model_name>_<task_name>.jsonl
```

For example:

```text
/pasteur/u/rdcunha/code/mmbu/results/gpt-4o/gpt-4o_detection_grounding_open_VQA.jsonl
```

## Notes

- Do not commit API keys. Set them through environment variables only.
- Existing local model adapters are unchanged.
- Existing task configs can still be used. The API example configs only set `batch_size: 1` for safer testing.
