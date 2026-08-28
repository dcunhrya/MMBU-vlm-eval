# Canonical eval package (parsers, metrics, LLM judge) lives in this directory
# on the Pasteur working tree.
#
# Import cache roots from mmbu.paths, not hardcoded pasteur strings:
#   from mmbu.paths import judge_cache_open, judge_cache_closed, results_dir
#
# Judge modules should set:
#   DEFAULT_JUDGE_CACHE_DIR = str(judge_cache_open())   # or judge_cache_closed()
#
# Analysis tree copy at ../src/analysis/src is a diverged fork — do not add
# features there. See docs/workspace-dropins/analysis/ and CATALOG.md.
