# FROZEN — do not extend

This directory is a **diverged fork** of `/pasteur/u/rdcunha/code/mmbu/inference/src/mmbu/eval/`.

Canonical parsers, metrics, and LLM judge code live in the inference repo.

- Do not add new judge gates, parsers, or metric functions here.
- New analysis should `import mmbu.eval` / `mmbu.paths` with `PYTHONPATH=…/inference/src`.
- Keep files only until figures are switched; then delete this tree.

See `../AGENTS.md` and inference `CATALOG.md`.
