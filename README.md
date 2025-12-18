# MMBU Inference Evaluation Pipeline (Autoregressive)

# Setup

llava will contain environment for LLaVA-Med model

.venv will contain environment for all models

- Can run `bash scripts/setup.sh`

OR (if issue)

- Install `requirements-default.txt` with `uv pip install -r requirements-default.txt` to create .venv
- Clone LLaVA-Med repo in src directory `git clone https://github.com/microsoft/LLaVA-Med.git`
- Install `requirements-llava.txt` in a new uv environment called llava

# Running Code

To run code you need to edit a .yaml config file in configs/ and a .sh eval file in eval/ and then run `sbatch scripts/{eval_file}.sh`

## Configs

In a .yaml file:

- In "model", no need to change anything (this will be set in .sh scripts file for eval)
- In "tasks", set the "name" and "data_path" for all eval tasks (can do multiple at once)
    - An example is shown in `configs/all_tasks.yaml`
    - All task are defined in the 'Tasks' section
- In "runtime", no need to change anything unless you want to edit model params or output dir

## Eval (Scripts)

In a .sh file (examples of cls, det, seg are shown):

- Change paths to current directory and environment directory
- EDIT HUGGINGFACE TOKEN on line 19 (it will not work as the token you put on github expires)
- CONFIG_PATH is path to .yaml file
- MODEL_TYPE is listed in `src/models/__init__.py` and will be in the 'Models' section
- MODEL_NAME is hf path to model and will be in the 'Models' section

**NOTE**: All models can be run with `run_vlm_eval.py` EXCEPT GEMMA MODELS NEED `run_vlm_eval_gemma.py`

**NOTE** Edit `run_vlm_eval.py`if want to change "output_dir" (line 50) or "cache_dir" (line 57) or "base_path" (line 73)

## Tasks

This is a list of all tasks and their corresponding file name (shown in 'configs/all_tasks.yaml')

- name: "classification_closed_VQA"
    - data_path: "final_cls/final_subsampled_cls_closed_12_15_25.tsv"

- name: "classification_open_VQA"
    - data_path: "final_cls/final_subsampled_cls_open_12_15_25.tsv"
    
- name: "detection_guess_bbox_closed_VQA"
    - data_path: "final_det/final_subsampled_det_guess_bbox_closed_12_15.tsv"

- name: "detection_guess_bbox_open_VQA"
    - data_path: "final_det/final_subsampled_det_guess_bbox_open_12_15.tsv"

- name: "detection_grounding_closed_VQA"
    - data_path: "final_det/final_subsampled_det_grounding_closed_12_15.tsv"

- name: "detection_grounding_open_VQA"
    - data_path: "final_det/final_subsampled_det_grounding_open_12_15.tsv"

- name: "segmentation_grounding_closed_VQA"
    - data_path: "final_seg/final_subsampled_seg_grounding_closed_12_15.tsv"

- name: "segmentation_grounding_open_VQA"
    - data_path: "final_seg/final_subsampled_seg_grounding_open_12_15.tsv"

- name: "segmentation_guess_bbox_open_VQA"
    - data_path: "final_seg/final_subsampled_seg_guess_mask_open_12_15.tsv"

## Models

List of all models that can be implement along with MODEL_TYPE and MODEL_NAME

- Qwen2-VL-2B-Instruct
    - MODEL_TYPE="qwen2vl"
    - MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
- Qwen2.5-VL-3B-Instruct
    - MODEL_TYPE="qwen2_5vl"
    - MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
- Qwen2.5-VL-7B-Instruct
    - MODEL_TYPE="qwen2_5vl"
    - MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
- Qwen2.5-VL-32B-Instruct
    - MODEL_TYPE="qwen2_5vl"
    - MODEL_NAME="Qwen/Qwen2.5-VL-32B-Instruct"
- Qwen3-VL-4B-Instruct
    - MODEL_TYPE="qwen3vl"
    - MODEL_NAME="Qwen/Qwen3-VL-4B-Instruct"
- Qwen3-VL-4B-Thinking
    - MODEL_TYPE="qwen3vl"
    - MODEL_NAME="Qwen/Qwen3-VL-4B-Thinking"
- Qwen3-VL-8B-Instruct
    - MODEL_TYPE="qwen3vl"
    - MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
- Qwen3-VL-8B-Thinking
    - MODEL_TYPE="qwen3vl"
    - MODEL_NAME="Qwen/Qwen3-VL-8B-Thinking"
- Qwen3-VL-32B-Instruct
    - MODEL_TYPE="qwen3vl"
    - MODEL_NAME="Qwen/Qwen3-VL-32B-Instruct"
- Qwen3-VL-32B-Thinking
    - MODEL_TYPE="qwen3vl"
    - MODEL_NAME="Qwen/Qwen3-VL-32B-Thinking"
- InternVL3_5-8B
    - MODEL_TYPE="intern"
    - MODEL_NAME="OpenGVLab/InternVL3_5-8B"	   
- gemma-3-4b-it
    - MODEL_TYPE="gemma3"
    - MODEL_NAME="google/gemma-3-4b-it"		   
- medgemma-4b-it
    - MODEL_TYPE="gemma3"
    - MODEL_NAME="google/medgemma-4b-it"	   
- Lingshu-7B
    - MODEL_TYPE="lingshu"
    - MODEL_NAME="lingshu-medical-mllm/Lingshu-7B"	   
- Lingshu-32B
    - MODEL_TYPE="lingshu"
    - MODEL_NAME="lingshu-medical-mllm/Lingshu-32B"
- llava-1.5-7b-hf
    - MODEL_TYPE="llava"
    - MODEL_NAME="llava-hf/llava-1.5-7b-hf"	   
- llava-med-v1.5-mistral-7b
    - MODEL_TYPE="llavamed"
    - MODEL_NAME="microsoft/llava-med-v1.5-mistral-7b"
- MedVLM-R1
    - MODEL_TYPE="medvlm"
    - MODEL_NAME="JZPeterPan/MedVLM-R1"
