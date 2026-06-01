
import os
import re
import random
import shutil
from itertools import product
import torch
 
 
DATA_FOLDER  = "data"
SOURCE_DIR   = "dataset_small"          # flat folder containing all raw images
DATASET_NAME = "dataset_small"
 
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10
 
SEED = 42

old_root = SOURCE_DIR
new_root = os.path.join(DATA_FOLDER, DATASET_NAME)
sets     = ['train', 'val', 'test']

FILENAME_RE = re.compile(
    r"^(?P<attr1>[^_]+)_(?P<attr2>[^_]+)_(?P<obj>[^_]+)_(?P<index>\d+)\.json_\d+_?\.png$"
)
 
raw_files = sorted(f for f in os.listdir(old_root) if f.endswith(".png"))
 
data = []
for fname in raw_files:
    m = FILENAME_RE.match(fname)
    if m:
        data.append({
            'filename': fname,
            'attr1':    m.group('attr1'),
            'attr2':    m.group('attr2'),
            'obj':      m.group('obj'),
        })
    else:
        print(f"WARNING: skipping unrecognised filename: {fname}")

attrs1  = sorted(set(d['attr1'] for d in data))
attrs2  = sorted(set(d['attr2'] for d in data))
objs    = sorted(set(d['obj']   for d in data))
triplets = sorted(set((d['attr1'], d['attr2'], d['obj']) for d in data))
 
print(f"attr1 values : {attrs1}")
print(f"attr2 values : {attrs2}")
print(f"obj values   : {objs}")
print(f"Unique triplets: {len(triplets)}")

random.seed(SEED)
 
set_assignment = {}   # filename -> 'train' | 'val' | 'test'
 
from collections import defaultdict
shuffled_triplets = list(triplets)
random.shuffle(shuffled_triplets)
 
n_total = len(shuffled_triplets)
n_train = max(1, round(n_total * TRAIN_RATIO))
n_val   = max(1, round(n_total * VAL_RATIO))
# test gets the remainder
 
triplet_split = {}   # (attr1, attr2, obj) -> 'train' | 'val' | 'test'
for i, triplet in enumerate(shuffled_triplets):
    if   i < n_train:
        triplet_split[triplet] = 'train'
    elif i < n_train + n_val:
        triplet_split[triplet] = 'val'
    else:
        triplet_split[triplet] = 'test'
 
# Every image in a triplet inherits the triplet's split assignment
set_assignment = {}   # filename -> 'train' | 'val' | 'test'
for d in data:
    set_assignment[d['filename']] = triplet_split[(d['attr1'], d['attr2'], d['obj'])]
 
#  Create directory structure 
 
os.makedirs(new_root, exist_ok=True)
for triplet in triplets:
    os.makedirs(os.path.join(new_root, 'images', '_'.join(triplet)), exist_ok=True)
 
#  Move images and build metadata 
 
new_data = []
for d in data:
    filename = d['filename']
    attr1, attr2, obj = d['attr1'], d['attr2'], d['obj']
 
    file_path = os.path.join(old_root, filename)
    new_dir   = os.path.join(new_root, 'images', f'{attr1}_{attr2}_{obj}')
    shutil.move(file_path, new_dir)
 
    new_data.append({
        'image': f"{attr1}_{attr2}_{obj}/{filename}",
        'attr1': attr1,
        'attr2': attr2,
        'obj':   obj,
        '_image': filename,
        'set':   set_assignment[filename],
    })
 
#  Save metadata 
 
split_name = 'compositional-split-natural'
torch.save(new_data, os.path.join(new_root, f'metadata_{split_name}.t7'))
 
# ── Write split pair files ───────────────────────────────────────────────────────
# Each file lists only the triplets exclusively assigned to that split.
 
split_triplets = defaultdict(list)
for triplet, s in triplet_split.items():
    split_triplets[s].append(triplet)
 
os.makedirs(os.path.join(new_root, split_name), exist_ok=True)
for s in sets:
    pairs = sorted(split_triplets[s])
    with open(os.path.join(new_root, split_name, f'{s}_triplets.txt'), 'w+') as file:
        file.writelines([f"{a1} {a2} {o}\n" for a1, a2, o in pairs])
 