#!/usr/bin/env python3
import os
import argparse
import json
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from vqa_dataset import PromptDataset, prompt_collate, create_template
from models import load_model_adapter


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_envs(model_dir):
    os.environ["HF_HOME"] = model_dir
    os.environ["TRANSFORMERS_CACHE"] = model_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = model_dir
    os.environ["VLLM_CACHE_ROOT"] = model_dir
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # editable by config if needed


def log_first_batch(batch_outputs, out_dir):
    path = os.path.join(out_dir, "first_batch_log.txt")
    with open(path, "w") as f:
        for idx, text in enumerate(batch_outputs):
            f.write(f"--- Sample {idx} ---\n{text}\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML config file.")
    parser.add_argument("--type", type=str, required=True,
                        help="type of model used from models/__init__.py")
    parser.add_argument("--name", type=str, required=True,
                        help="huggingface path to model")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    tasks_cfg = cfg["tasks"]
    run_cfg  = cfg["runtime"]
    output_dir = '/pasteur/u/rdcunha/code/mmbu/results'
    
    # model_type = model_cfg["type"]
    # model_name = model_cfg["name"]
    model_type = args.type
    model_name = args.name
    device     = model_cfg.get("device", "auto")
    cache_dir  = "/pasteur/u/rdcunha/models"

    os.makedirs(output_dir, exist_ok=True)
    file_model_name = model_name.split('/')[-1]
    model_path = file_model_name.replace('/', '_')
    output_dir = os.path.join(output_dir, model_path)
    os.makedirs(output_dir, exist_ok=True)
    
    set_envs(cache_dir)

    adapter = load_model_adapter(model_type, model_name, device, cache_dir)
    model, processor = adapter.load()

    # ----------------------------------
    # Dataset setup
    # ----------------------------------
    base_path = '/pasteur/u/rdcunha/data_cache/mmbu/final_data/subsampled_mmbu_data'
    
    for task_cfg in tasks_cfg:
        print(f"Running task: {task_cfg['name']}")
        out_file = os.path.join(output_dir, f"{file_model_name.replace('/', '_')}_{task_cfg['name']}.jsonl")
        tsv_path = os.path.join(base_path, task_cfg["data_path"])
        df = pd.read_csv(tsv_path, sep='\t')
        
        add_options = ("open" not in task_cfg["name"])
        dataset = PromptDataset(df=df, add_options=add_options)
        loader = DataLoader(
            dataset,
            batch_size=run_cfg["batch_size"],
            shuffle=False,
            collate_fn=prompt_collate,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4
        )
    
        existing = set()
        if os.path.exists(out_file):
            with open(out_file, "r") as f:
                for line in f:
                    try:
                        j = json.loads(line)
                        existing.add(j["index"])
                    except:
                        pass
                        
        counter = 0
        saved = []
        first_batch_logged = False
        
        with open(out_file, "a") as f:
            for batch in tqdm(loader, desc="Inference"):
        
                new_batch = [x for x in batch if x["index"] not in existing]
                if not new_batch:
                    continue
        
                # inference
                try:
                    all_inputs = []
                    for item in new_batch:
                        single_msg = adapter.create_template(item)
                        single_inp = adapter.prepare_inputs([single_msg], processor, model)
                        all_inputs.append(single_inp)
                    
                    batched_inputs = adapter.stack_inputs(all_inputs, model)
                    
                    outputs = adapter.infer(model, processor, batched_inputs, run_cfg["max_new_tokens"])
                except: 
                    print(f"could not generate for {batch}")
                    continue
        
                # log first batch only
                if run_cfg["log_first_batch"] and not first_batch_logged:
                    log_first_batch(outputs, output_dir)
                    first_batch_logged = True
        
                # save results
                for it, out_text in zip(new_batch, outputs):
                    obj = {
                        "index": it["index"],
                        "question": it["question"],
                        "image_path": it["image_path"],
                        "dataset": it["dataset"],
                        "modality": it["modality"],
                        "class_label": it["class_label"],
                        "answer": out_text
                    }
                    if "options" in it and it["options"] is not None:
                        obj["options"] = it["options"]
                
                    saved.append(obj)
                    existing.add(it["index"])
                    counter += 1
        
                    if counter % 50 == 0:
                        for s in saved:
                            f.write(json.dumps(s) + "\n")
                        f.flush()
                        saved = []
        
            # Save remainder
            for s in saved:
                f.write(json.dumps(s) + "\n")
    
    print('Completed')


if __name__ == "__main__":
    main()
