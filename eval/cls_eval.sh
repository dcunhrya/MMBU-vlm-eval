#!/bin/bash
#SBATCH -p pasteur
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --account=pasteur
#SBATCH -J eval_mg
#SBATCH -N 1
#SBATCH --output=slurm_logs/base-%j-out.txt
#SBATCH --error=slurm_logs/base-%j-err.txt

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

mkdir -p slurm_logs

source /pasteur/u/rdcunha/code/mmbu/inference/.venv/bin/activate
cd /pasteur/u/rdcunha/code/mmbu/inference

###############################################
export HF_TOKEN="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"
export PYTHONPATH="$(pwd):$PYTHONPATH"
###############################################

echo "Running run_vlm_eval.py"

CONFIG_PATH="configs/cls_config.yaml"
MODEL_TYPE="llava"
MODEL_NAME="llava-hf/llava-1.5-7b-hf"

python src/run_vlm_eval.py --config ${CONFIG_PATH} --type ${MODEL_TYPE} --name ${MODEL_NAME}