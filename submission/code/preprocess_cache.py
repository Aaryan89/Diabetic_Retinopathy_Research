"""
Pre-caches deterministic 224x224 CLAHE-cropped images to disk
to maximize DataLoader throughput during multi-variant training.
"""

import os
import time
import pandas as pd
from PIL import Image
from tqdm import tqdm
from data import preprocess_retinal_image

def cache_all_images(data_dir=os.path.join(".", "data", "aptos2019")):
    csv_path = os.path.join(data_dir, "train.csv")
    df = pd.read_csv(csv_path)
    
    cache_dir = os.path.join(data_dir, "preprocessed_224")
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Pre-caching {len(df)} images to {cache_dir}...")
    start = time.time()
    
    for idx, row in df.iterrows():
        out_path = os.path.join(cache_dir, f"{row['id_code']}.png")
        if os.path.exists(out_path):
            continue
            
        raw_path = os.path.join(data_dir, "train_images", f"{row['id_code']}.png")
        prep_rgb = preprocess_retinal_image(raw_path, target_size=(224, 224))
        prep_pil = Image.fromarray(prep_rgb)
        prep_pil.save(out_path, format="PNG")
        
        if (idx + 1) % 500 == 0 or (idx + 1) == len(df):
            print(f"  Cached {idx + 1}/{len(df)} images ({time.time() - start:.1f}s)...")
            
    print(f"Pre-caching completed in {time.time() - start:.1f}s!")

if __name__ == "__main__":
    cache_all_images()
