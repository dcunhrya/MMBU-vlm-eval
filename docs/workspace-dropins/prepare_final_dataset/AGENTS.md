# AGENTS.md (prepare_final_dataset)

HF ImageFolder staging, validation, and CXR image QA for `ryandcunha/mmbu` and `ryandcunha/mmbu-context`.

Workspace index: `/pasteur/u/rdcunha/code/mmbu/CATALOG.md`.

## Job

- `python -m prepare_final_dataset.prepare_mmbu_hf_dataset`
- `python -m prepare_final_dataset.prepare_mmbu_context_hf_dataset`
- `python -m prepare_final_dataset.validate_mmbu_hf_dataset`
- `python -m prepare_final_dataset.diagnose_image_quality` / `fix_dark_images`

Run from `/pasteur/u/rdcunha/code/mmbu/src` with `uv run --no-sync`. Pasteur cache env (`HF_HOME`, `UV_CACHE_DIR`, `XDG_CACHE_HOME`).

## Do not

- Copy this package into the inference repo.
- Upload to Hub without explicit confirmation (`hf upload` / `upload-large-folder`).
- Change Hub visibility/gating/license from the CLI.
- Stage from symlink trees for full-image reupload — use `--copy-images` into `mmbu_hf_upload/` / `mmbu_context_hf_upload/`.

Sanitized metadata must not include `answer`, `class_label`, or `mask_path`.
