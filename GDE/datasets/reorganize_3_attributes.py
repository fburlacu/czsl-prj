import os
import re
import random
import shutil
from collections import defaultdict

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATA_FOLDER  = r"D:\Univer\Master\2B - 2026\fomo repo\czsl-prj\GDE\data"
SOURCE_DIR   = r"D:\Univer\Master\2B - 2026\fomo repo\czsl-prj\GDE\data\images" # root containing per-object subfolders
DATASET_NAME = "3_attributes"  # name of the new dataset folder to create

# Attribute slot mapping. Filename order is <color>_<material>_<object>_<copy>.
# attr1 = color, attr2 = material, obj = object.
# (composition_dataset.py expects keys attr1, attr2, obj.)

# Triplet-level split ratios (applied to UNIQUE triplets, not images).
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20   

# If True, build the test set so that roughly half of its IMAGES come from
# triplets also present in training (seen) and half from triplets held out
# of training (unseen) — a balanced synthetic benchmark in the paper's
# seen/unseen sense. If False, do a plain triplet-level ratio split.
BALANCE_TEST_SEEN_UNSEEN = True

# Fraction of a "seen-test" triplet's copies that go to the test pool
# (the rest go to train). Only used when BALANCE_TEST_SEEN_UNSEEN is True.
SEEN_TEST_HOLDOUT = 0.5

SEED = 42

SPLIT_NAME = "compositional-split-natural"

# ─────────────────────────────────────────────────────────────────────────────
# Filename parsing
# ─────────────────────────────────────────────────────────────────────────────
# Matches: red_glass_apple_1.png   (color_material_object_copynumber.png)
# Object names may themselves be unknown, but the COPY number is always the
# final _<digits> before .png, so we anchor on that and treat everything
# between the 2nd underscore and the copy index as the object — this keeps
# multi-word safety even though the current set is single-word.

FILENAME_RE = re.compile(
    r"^(?P<color>[^_]+)_(?P<material>[^_]+)_(?P<obj>.+)_(?P<copy>\d+)\.png$"
)


def parse_all_files(source_root):
    """Walk per-object subfolders and parse every png into a record."""
    records = []
    for dirpath, _dirnames, filenames in os.walk(source_root):
        for fname in sorted(filenames):
            if not fname.endswith(".png"):
                continue
            m = FILENAME_RE.match(fname)
            if not m:
                print(f"WARNING: skipping unrecognised filename: {fname}")
                continue
            records.append({
                "filename": fname,
                "src_path": os.path.join(dirpath, fname),
                "color":    m.group("color"),
                "material": m.group("material"),
                "obj":      m.group("obj"),
            })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Split logic
# ─────────────────────────────────────────────────────────────────────────────

def ensure_primitive_coverage(train_triplets, all_colors, all_materials, all_objs,
                              triplet_pool, rng):
    """
    Guarantee every primitive (color, material, object) appears in at least one
    training triplet. If a primitive is missing, pull a triplet containing it
    from the pool into train. Returns the (possibly enlarged) train set.
    """
    train = set(train_triplets)

    def covered(prims_idx):
        seen = set()
        for t in train:
            seen.add(t[prims_idx])
        return seen

    for idx, universe in [(0, all_colors), (1, all_materials), (2, all_objs)]:
        missing = set(universe) - covered(idx)
        for prim in sorted(missing):
            candidates = [t for t in triplet_pool if t[idx] == prim]
            if not candidates:
                raise RuntimeError(
                    f"Primitive {prim!r} (slot {idx}) appears in NO triplet at all."
                )
            chosen = rng.choice(candidates)
            train.add(chosen)
    return train


def main():
    rng = random.Random(SEED)

    old_root = SOURCE_DIR
    new_root = os.path.join(DATA_FOLDER, DATASET_NAME)

    # 1) Parse every image -------------------------------------------------------
    data = parse_all_files(old_root)
    if not data:
        raise RuntimeError(f"No parsable .png files found under {old_root!r}.")

    colors    = sorted(set(d["color"]    for d in data))
    materials  = sorted(set(d["material"] for d in data))
    objs      = sorted(set(d["obj"]      for d in data))
    triplets  = sorted(set((d["color"], d["material"], d["obj"]) for d in data))

    print(f"colors    : {colors}")
    print(f"materials : {materials}")
    print(f"objects   : {objs}")
    print(f"Unique triplets: {len(triplets)}")
    print(f"Total images   : {len(data)}")

    # 2) Triplet-level split -----------------------------------------------------
    shuffled = list(triplets)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = max(1, round(n_total * TRAIN_RATIO))
    n_val   = max(1, round(n_total * VAL_RATIO))

    train_triplets = set(shuffled[:n_train])
    val_triplets   = set(shuffled[n_train:n_train + n_val])
    test_triplets  = set(shuffled[n_train + n_val:])

    # 3) Guarantee every primitive is represented in TRAIN -----------------------
    train_triplets = ensure_primitive_coverage(
        train_triplets, colors, materials, objs, shuffled, rng
    )
    # remove any newly-promoted triplet from val/test to avoid overlap
    val_triplets  -= train_triplets
    test_triplets -= train_triplets

    # 4) Decide per-image set assignment ----------------------------------------
    # Group image records by triplet
    by_triplet = defaultdict(list)
    for d in data:
        by_triplet[(d["color"], d["material"], d["obj"])].append(d)

    set_assignment = {}   # filename -> split

    # Helper to mark every image of a triplet to a single split
    def assign_all(triplet, split):
        for d in by_triplet[triplet]:
            set_assignment[d["filename"]] = split

    # train / val are whole-triplet
    for t in train_triplets:
        assign_all(t, "train")
    for t in val_triplets:
        assign_all(t, "val")

    if not BALANCE_TEST_SEEN_UNSEEN:
        # Plain triplet-level: all test triplets are purely held-out (unseen)
        for t in test_triplets:
            assign_all(t, "test")
    else:
        # Balanced benchmark: make ~half the test IMAGES "seen" by sourcing them
        # from triplets that are ALSO in train, and ~half "unseen" from held-out
        # triplets. We split the test triplets into two groups.
        test_list = sorted(test_triplets)
        rng.shuffle(test_list)
        half = len(test_list) // 2
        unseen_test = set(test_list[:half])     # held out of train -> unseen
        seen_test   = set(test_list[half:])     # also seeded into train -> seen

        # Unseen test triplets: every copy goes to test
        for t in unseen_test:
            assign_all(t, "test")

        # Seen test triplets: copies are split between train and test
        for t in seen_test:
            copies = list(by_triplet[t])
            rng.shuffle(copies)
            n_test = max(1, round(len(copies) * SEEN_TEST_HOLDOUT))
            test_copies  = copies[:n_test]
            train_copies = copies[n_test:]
            for d in test_copies:
                set_assignment[d["filename"]] = "test"
            for d in train_copies:
                set_assignment[d["filename"]] = "train"
            # This triplet now appears in the TRAIN triplet set as well
            train_triplets.add(t)

    # 5) Create directory structure + move images --------------------------------
    os.makedirs(new_root, exist_ok=True)
    for t in triplets:
        os.makedirs(os.path.join(new_root, "images", "_".join(t)), exist_ok=True)

    new_data = []
    for d in data:
        color, material, obj = d["color"], d["material"], d["obj"]
        folder = f"{color}_{material}_{obj}"
        dst_dir = os.path.join(new_root, "images", folder)
        dst_path = os.path.join(dst_dir, d["filename"])

        if os.path.abspath(d["src_path"]) != os.path.abspath(dst_path):
            shutil.copy2(d["src_path"], dst_path)   # copy, not move — keeps raws safe

        new_data.append({
            "image":  f"{folder}/{d['filename']}",
            "attr1":  color,        # color
            "attr2":  material,     # material
            "obj":    obj,          # object
            "_image": d["filename"],
            "set":    set_assignment[d["filename"]],
        })

    # 6) Save metadata -----------------------------------------------------------
    torch.save(new_data, os.path.join(new_root, f"metadata_{SPLIT_NAME}.t7"))

    # 7) Write split triplet files ----------------------------------------------
    # Each file lists the triplets assigned to that split. A triplet that was
    # seeded into both train and test (balanced mode) is listed in BOTH train
    # and test, which is exactly how seen-pair evaluation expects it.
    os.makedirs(os.path.join(new_root, SPLIT_NAME), exist_ok=True)

    final_train = set(train_triplets)
    final_val   = set(val_triplets)
    # test triplet file = every triplet that has at least one image in test
    final_test = set()
    for d in new_data:
        if d["set"] == "test":
            final_test.add((d["attr1"], d["attr2"], d["obj"]))

    split_to_triplets = {
        "train": sorted(final_train),
        "val":   sorted(final_val),
        "test":  sorted(final_test),
    }

    for split, trips in split_to_triplets.items():
        path = os.path.join(new_root, SPLIT_NAME, f"{split}_triplets.txt")
        with open(path, "w+") as f:
            f.writelines(f"{c} {m} {o}\n" for c, m, o in trips)

    # 8) Report ------------------------------------------------------------------
    n_imgs = defaultdict(int)
    for d in new_data:
        n_imgs[d["set"]] += 1

    seen_triplets = set(final_train)
    n_seen_test_imgs = sum(
        1 for d in new_data
        if d["set"] == "test" and (d["attr1"], d["attr2"], d["obj"]) in seen_triplets
    )
    n_test_imgs = n_imgs["test"]

    print("\n── split summary ─────────────────────────────")
    print(f" train triplets : {len(final_train):>4}   images: {n_imgs['train']}")
    print(f" val   triplets : {len(final_val):>4}   images: {n_imgs['val']}")
    print(f" test  triplets : {len(final_test):>4}   images: {n_imgs['test']}")
    if n_test_imgs:
        frac = n_seen_test_imgs / n_test_imgs
        print(f" test seen images : {n_seen_test_imgs}/{n_test_imgs} "
              f"({frac:.0%} seen, {1-frac:.0%} unseen)")
    print("──────────────────────────────────────────────")
    print(f"Wrote metadata + triplet files under {new_root}/")


if __name__ == "__main__":
    main()