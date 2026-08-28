# Retire the analysis eval fork

Canonical eval: `inference/src/mmbu/eval/` (+ `mmbu.paths`).

Analysis `src/` is frozen (`docs/workspace-dropins/analysis/src/FROZEN.md`). After figures import inference:

1. `export PYTHONPATH=/pasteur/u/rdcunha/code/mmbu/inference/src`
2. Replace `from src.constants import *` TSV paths with `mmbu.paths.data_root()` / `configs/tasks.yaml`. Keep color maps locally if needed.
3. Call `mmbu.eval` parsers/metrics instead of `src.utils_*`.
4. Delete or archive analysis `src/utils_*.py` and `src/llm_judge_*.py`.
5. Stop using `finalized_analysis/run_*_llm_judge.{py,sh}` (`LEGACY_JUDGE.md`).
6. Inference `monitor_anthropic_llm_judge.py`: submit `python -m mmbu judge` only; drop `analysis_python` and analysis smoke sbatch.

Do not vendor `prepare_final_dataset` into inference.

Install freeze notices on Pasteur:

```bash
bash scripts/install_workspace_index.sh
```
