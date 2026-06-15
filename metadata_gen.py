import os
import torch

DATASET_DIR = r"GDE/data/3_attribute_dataset"

def build_missing_t7():
    images_dir = os.path.join(DATASET_DIR, "images")
    split_dir = os.path.join(DATASET_DIR, "compositional-split-natural")
    
    def load_split(filename):
        path = os.path.join(split_dir, filename)
        with open(path, 'r') as f:
            return set(tuple(line.strip().split()) for line in f)

    print("Reading text splits...")
    train_triplets = load_split("train_triplets.txt")
    val_triplets = load_split("val_triplets.txt")
    test_triplets = load_split("test_triplets.txt")

    metadata = []
    
    print("Scanning images and building metadata dictionary...")
    for root, _, files in os.walk(images_dir):
        for filename in files:
            if not filename.endswith(".png"): 
                continue
            
            parts = filename.replace(".png", "").split("_")
            if len(parts) >= 4:
                attr1, attr2, obj = parts[0], parts[1], parts[2]
                triplet = (attr1, attr2, obj)
                
                if triplet in train_triplets: 
                    set_type = 'train'
                elif triplet in val_triplets: 
                    set_type = 'val'
                elif triplet in test_triplets: 
                    set_type = 'test'
                else: 
                    continue
                
                rel_path = os.path.relpath(os.path.join(root, filename), images_dir)
                
                metadata.append({
                    'image': rel_path.replace('\\', '/'),
                    'attr1': attr1,
                    'attr2': attr2,
                    'obj': obj,
                    'set': set_type
                })

    t7_path = os.path.join(DATASET_DIR, "metadata_compositional-split-natural.t7")
    torch.save(metadata, t7_path)
    print(f"Complete. Processed {len(metadata)} images and saved to {t7_path}")

if __name__ == "__main__":
    build_missing_t7()