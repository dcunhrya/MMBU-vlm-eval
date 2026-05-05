import unittest
import pandas as pd
import os
import ast
import numpy as np
from PIL import Image
import logging
import re

# --- CONFIGURATION ---
BASE_DIR = '/pasteur/u/rdcunha/data_cache/mmbu/final_data/subsampled_mmbu_data'
# Update this if your image paths in the TSV are relative. 
# If they are absolute paths, this variable is ignored by the logic below.
IMAGE_ROOT_DIR = '/pasteur/u/rdcunha/data_cache/mmbu/final_data/subsampled_mmbu_data' 
RESULTS_DIR = '/pasteur/u/rdcunha/code/mmbu/inference/src/dataset_check'

TSV_FILES = [
    'final_cls/final_subsampled_cls_open_1_13_v2.tsv',
    'final_cls/final_subsampled_cls_closed_1_13_v2.tsv',
    'final_det/final_subsampled_det_grounding_closed_1_13_v2.tsv',
    'final_det/final_subsampled_det_grounding_open_1_13_v2.tsv't,
    # 'final_det/final_subsampled_det_guess_bbox_closed_1_12_v2.tsv',
    # 'final_det/final_subsampled_det_guess_bbox_open_1_12_v2.tsv',
    'final_seg/final_subsampled_seg_grounding_closed_1_13_v2.tsv',
    'final_seg/final_subsampled_seg_grounding_open_1_13_v2.tsv',
]

LOG_FILE = os.path.join(RESULTS_DIR, 'dataset_validation_results.log')

# --- LOGGING SETUP ---
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

class TestTSVIntegrity(unittest.TestCase):

    def setUp(self):
        """Prepare the list of files to test."""
        self.files_to_process = []
        for rel_path in TSV_FILES:
            full_path = os.path.join(BASE_DIR, rel_path)
            if os.path.exists(full_path):
                self.files_to_process.append((rel_path, full_path))
            else:
                logging.error(f"CRITICAL: File not found at {full_path}")

    def test_closed_options_integrity(self):
        """
        Logic for 'closed' files:
        1. 'None of the above' must be in options.
        2. All keys in 'answer' (e.g., 'A', 'B') must exist in options.
        3. If single answer, verify option text matches 'class_label'.
        4. Option counts must be consistent within a dataset.
        """
        logging.info("=== STARTING CLOSED OPTION INTEGRITY TESTS ===")
        
        for rel_path, full_path in self.files_to_process:
            if 'closed' not in rel_path:
                continue
            
            logging.info(f"Processing: {rel_path}")
            try:
                df = pd.read_csv(full_path, sep='\t')
            except Exception as e:
                logging.error(f"Failed to read {rel_path}: {e}")
                continue

            # Iterate by dataset to check consistency
            for dataset_name, group in df.groupby('dataset'):
                option_counts = []
                
                for idx, row in group.iterrows():
                    # Parse Options List
                    try:
                        raw_options = row['options']
                        if pd.isna(raw_options):
                            logging.error(f"[{rel_path}][Row {idx}] Options are NaN.")
                            continue
                        options_list = ast.literal_eval(raw_options)
                    except (ValueError, SyntaxError):
                        logging.error(f"[{rel_path}][Row {idx}] Malformed options string: {row['options']}")
                        continue

                    # TEST 1: "None of the above" presence
                    has_none = any("none of the above" in opt.lower() for opt in options_list)
                    if not has_none:
                        logging.error(f"[{rel_path}][Dataset: {dataset_name}][Row {idx}] Missing 'None of the above' in options.")

                    # Build Map: { 'A': 'Cat', 'B': 'Dog' }
                    opt_map = {}
                    for opt in options_list:
                        # Assumes format "A) text" or "A. text"
                        if ')' in opt:
                            parts = opt.split(')', 1)
                        elif '.' in opt:
                            parts = opt.split('.', 1)
                        else:
                            parts = [opt]
                        
                        if len(parts) > 1:
                            key = parts[0].strip().upper()
                            val = parts[1].strip()
                            opt_map[key] = val

                    # TEST 2: Answer Key Validity (Handling Multiple Letters)
                    raw_answer = str(row['answer']).strip().upper()
                    
                    # Robustly extract all capital letters found in the answer string
                    # This handles "A", "A,B", "A, B", "['A', 'C']" etc.
                    answer_keys = re.findall(r'[A-Z]', raw_answer)
                    
                    if not answer_keys:
                         logging.warning(f"[{rel_path}][Row {idx}] No valid answer keys found in '{raw_answer}'.")

                    for key in answer_keys:
                        if key not in opt_map:
                            logging.error(f"[{rel_path}][Row {idx}] Answer key '{key}' (from '{raw_answer}') NOT FOUND in options: {list(opt_map.keys())}")
                        else:
                            # TEST 3: Text Match Check
                            # We only strictly check text match if it is a SINGLE answer.
                            # (Matching multi-label text lists against multi-answer keys is ambiguous/complex)
                            if len(answer_keys) == 1:
                                chosen_text = opt_map[key]
                                class_label = str(row['class_label']).strip()
                                if class_label.lower() not in chosen_text.lower():
                                     logging.warning(f"[{rel_path}][Row {idx}] Label Mismatch. Key '{key}' -> '{chosen_text}', but class_label is '{class_label}'.")

                    option_counts.append(len(options_list))

                # TEST 4: Consistent Option Counts per Dataset
                if option_counts:
                    unique_counts = set(option_counts)
                    if len(unique_counts) > 1:
                        logging.error(f"[{rel_path}][Dataset: {dataset_name}] Inconsistent option counts found: {unique_counts}")

    def test_image_validity(self):
        """
        For ALL files:
        1. Verify image can be loaded.
        2. Verify image is not white/blank.
        """
        logging.info("=== STARTING IMAGE VALIDITY TESTS ===")
        
        for rel_path, full_path in self.files_to_process:
            logging.info(f"Checking images for: {rel_path}")
            df = pd.read_csv(full_path, sep='\t')
            
            for idx, row in df.iterrows():
                img_path = row.get('image_path')
                
                if pd.isna(img_path):
                    logging.error(f"[{rel_path}][Row {idx}] Image path is NaN.")
                    continue

                if os.path.isabs(img_path):
                    target_path = img_path
                else:
                    target_path = os.path.join(IMAGE_ROOT_DIR, img_path)

                try:
                    with Image.open(target_path) as img:
                        img.load() 
                        
                        # Check for White/Blank Image
                        img_gray = img.convert('L')
                        img_data = np.array(img_gray)
                        mean_intensity = np.mean(img_data)
                        std_dev = np.std(img_data)
                        
                        # High mean (white) + Low std_dev (flat)
                        if mean_intensity > 250 and std_dev < 5:
                            logging.warning(f"[{rel_path}][Row {idx}] Image appears BLANK/WHITE. Mean: {mean_intensity:.2f}. Path: {target_path}")

                except FileNotFoundError:
                    logging.error(f"[{rel_path}][Row {idx}] Image NOT FOUND at: {target_path}")
                except Exception as e:
                    logging.error(f"[{rel_path}][Row {idx}] Corrupt Image. Error: {e}. Path: {target_path}")

if __name__ == '__main__':
    unittest.main()