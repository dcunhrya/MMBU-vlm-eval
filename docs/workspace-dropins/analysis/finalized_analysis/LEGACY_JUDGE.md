# Legacy judge runners — retired

`run_closed_vqa_llm_judge.py`, `run_open_vqa_llm_judge.py`, and the matching `.sh` / sbatch wrappers are **legacy**.

Use the inference CLI instead:

```bash
cd /pasteur/u/rdcunha/code/mmbu/inference
source judge/bin/activate
export PYTHONPATH=src
python -m mmbu judge --kind open --results-dir "$MMBU_RESULTS_DIR" --model-filter <stem>
```

`monitor_anthropic_llm_judge.py` (inference) should submit `python -m mmbu judge`, not sbatch these scripts.

Smoke tests belong under `inference/eval/` or `python -m mmbu judge --max-rows N`.
