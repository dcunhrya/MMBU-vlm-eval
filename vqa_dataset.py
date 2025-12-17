import os
import json
import ast
import hashlib
import random
import re
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class PromptDataset(Dataset):
    def __init__(self, df, add_options=True):
        self.df = df.reset_index(drop=True)
        self.add_options = add_options

        self.images = []

        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            path = row["image_path"]

            with Image.open(path) as img:
                img.load()
                final_img = img.copy()
            final_img = pad_to_512(final_img)

            self.images.append(final_img)

        print("✅ All images loaded")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.images[idx]

        base = {
            "index": int(row["index"]),
            "question": row["question"],
            "image_path": row["image_path"],
            "dataset": row["dataset"],
            "class_label": row["class_label"],
            "image": img,
            "modality": row["modality"],
        }

        if self.add_options:
            base["question"] = row["question"] + f" Options: {row['options']}"
            base["options"] = row["options"]

        return base

def pad_to_512(img):
    """
    Pads image to 512x512 WITHOUT resizing.
    Image stays top-left aligned.
    Padding is added ONLY to the right and bottom.
    """
    w, h = img.size
    size = 512

    if w >= size and h >= size:
        return img

    pad_w = max(0, size - w)
    pad_h = max(0, size - h)

    # (left, top, right, bottom)
    return ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=0)

def prompt_collate(batch):
    # Keep as list so VLM inference (e.g., vLLM multimodal) works correctly.
    return batch


def create_template(item):
    conversation = {
        "prompt": f"USER: <image>\n{item['question']}\nASSISTANT:",
        "multi_modal_data": {"image": item['image']},
    }
    return conversation