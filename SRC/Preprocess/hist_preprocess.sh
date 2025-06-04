#!/bin/bash

ROOT_DIR="../packages/CLAM"
USER_DIR="/nfs/home/users/echauhan/sharedscratch"

python "$ROOT_DIR/create_patches_fp.py" \
  --source "$USER_DIR/TCGA_BRCA/data" \
  --seg --patch --stitch \
  --save_dir "$USER_DIR/TCGA_BRCA/patching" \
  --preset tcga.csv

python "$ROOT_DIR/packages/CLAM/extract_features_fp.py" \
  --data_h5_dir "$USER_DIR/TCGA_BRCA/patching" \
  --data_slide_dir "$USER_DIR/TCGA_BRCA/data" \
  --csv_path "$ROOT_DIR/dataset_csv/brca.csv" \
  --feat_dir "$USER_DIR/TCGA_BRCA/uni_v2_features" \
  --model_name "uni_v2" \
  --batch_size 1024 \
  --no_auto_skip \
  --target_patch_size 224