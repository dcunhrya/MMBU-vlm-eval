# artifacts/

Discoverability only. Large JSONL and judge CSVs stay where they already live; this directory holds **symlinks**, not copies.

On Pasteur run:

```bash
bash scripts/install_workspace_index.sh
```

That creates:

| Link | Target |
|------|--------|
| `artifacts/llm_judge_cache` | `../src/analysis/finalized_analysis/llm_judge_cache` |
| `artifacts/results_cot_v3` | `../results_cot_v3` |
| `/pasteur/u/rdcunha/code/mmbu/artifacts/llm_judge_cache` | same judge cache (workspace-level index) |

Do not `cp` the 8GB cache. New code reads `mmbu.paths.judge_cache()` / `$MMBU_JUDGE_CACHE`.
