# mmbu-onboard-model reference

## Adapter → venv → Slurm profile

| `--type` | Venv (`configs/cluster.yaml`) | Profile |
|----------|----------------------------------|---------|
| `qwen2_5vl`, `qwen3vl`, `intern`, `medvlm`, `lingshu`, `octomed` | `.venv/bin/activate` | `gpu_l40s`, `gpu_a6000`, `gpu_a100` |
| `qwen3_6` | `qwen36/bin/activate` | `qwen36_l40s`, `qwen36_moe_l40s` |
| `qwen3_6_vllm` | `qwen36-vllm/bin/activate` | `qwen36_vllm_l40s`, `qwen36_moe_vllm_l40s` |
| `octomed_vllm` | `qwen36-vllm/bin/activate` | `octomed_vllm_l40s`, `octomed_vllm_a6000` |
| `llavamed` | `llava/bin/activate` | `gpu_l40s` |
| `gemma3` | `.venv` (verify) | `gpu_l40s` |
| judge stage | `judge/bin/activate` | `judge_l40s`, `judge_h200` |
| `anthropic`, `openai`, `gemini` | `.venv` | local or batch API (no GPU profile) |

## MODEL_ALIASES

Display name → HF id (`src/models/__init__.py`):

| Alias | HF repo |
|-------|---------|
| `Octomed-2-8B` | `OctoMed/OctoMed-8B` |

Artifacts use the alias as directory stem when that is the configured `model.name`.

## Run config skeleton

```yaml
model:
  type: "qwen3_6_vllm"
  name: "Qwen/Qwen3.6-27B-FP8"

run:
  tasks: all
  stages: [infer]

execution:
  mode: slurm
  inference_profile: qwen36_vllm_l40s
  isolate_tasks: true
  job_prefix: my_model_v1

runtime:
  output_dir: "/pasteur/u/rdcunha/code/mmbu/results_cot_v3"
  batch_size: 4
  max_new_tokens: 2048
  save_every: 50
```

## Submit commands

```bash
# Render sbatch only
python -m mmbu submit --config configs/runs/X.yaml --stages infer --isolate-tasks

# Render + sbatch
python -m mmbu submit --config configs/runs/X.yaml --stages infer --isolate-tasks --launch

# Full pipeline (infer → score → judge → aggregate), local
python -m mmbu run --config configs/runs/X.yaml --tasks all

# Inference-only legacy mode (single sbatch, all tasks one job)
python -m mmbu submit \
  --profile gpu_l40s \
  --config configs/all_tasks.yaml \
  --type qwen2_5vl \
  --name Qwen/Qwen2.5-VL-7B-Instruct \
  --job-name qwen25_7b_all
# → eval/generated/qwen25_7b_all.sh
```

Generated scripts land in `eval/generated/`.

## API model ids (frontier)

| Project name | Anthropic API id |
|--------------|------------------|
| Haiku 4.5 | `claude-haiku-4-5-20251001` |
| Sonnet 5 | `claude-sonnet-5` |
| Opus 5 | `claude-opus-5` |
| Fable 5 | `claude-fable-5` |

Example configs: `configs/anthropic_sonnet5.yaml`, `configs/openai_api_example.yaml`.

## Pasteur cache snippet (manual runs)

```bash
export XDG_CACHE_HOME="/pasteur/u/rdcunha/.cache"
export VLLM_CACHE_ROOT="${XDG_CACHE_HOME}/vllm"
export HF_HOME="/pasteur/u/rdcunha/models"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export UV_CACHE_DIR="/pasteur/u/rdcunha/uv_cache"
```

## Qwen3.6 helpers

| Script | Purpose |
|--------|---------|
| `scripts/submit_qwen36_vllm_infer.sh` | smoke / gate / remaining |
| `scripts/qwen36_vllm_cutover.py` | manifest, cancel, resubmit |
| `eval/smoke_qwen36_vllm_l40s.sh` | compat smoke sbatch |
| `docs/qwen36_vllm_cutover.md` | cutover runbook |

## Gemma legacy exception

If unified pipeline submit fails for Gemma, legacy path:

```bash
python src/run_vlm_eval_gemma.py --config ... --type gemma3 --name google/gemma-3-...
```

Prefer migrating to `python -m mmbu infer` once validated.
