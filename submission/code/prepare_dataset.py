"""
Dataset Preparation Script for APTOS 2019 Blindness Detection
Extracts raw images and builds train.csv matching official Kaggle competition format.
"""
import os
import io
import pandas as pd
from PIL import Image

def prepare_aptos_dataset():
    data_dir = os.path.join(".", "data", "aptos2019")
    img_dir = os.path.join(data_dir, "train_images")
    os.makedirs(img_dir, exist_ok=True)
    
    # Save Kaggle CLI command instructions
    instructions_file = os.path.join(data_dir, "kaggle_command.txt")
    with open(instructions_file, "w", encoding="utf-8") as f:
        f.write("# Official Kaggle CLI download command:\n")
        f.write("kaggle competitions download -c aptos2019-blindness-detection -p ./data/aptos2019\n")
        f.write("unzip ./data/aptos2019/aptos2019-blindness-detection.zip -d ./data/aptos2019/\n")
    print(f"Wrote Kaggle command instructions to {instructions_file}")
    
    # URL to public mirror of APTOS 2019
    parquet_url = "https://huggingface.co/datasets/bumbledeep/aptos/resolve/main/data/train-00000-of-00001.parquet"
    print(f"Loading APTOS dataset from {parquet_url}...")
    df = pd.read_parquet(parquet_url)
    print(f"Loaded parquet dataframe with {len(df)} rows.")
    
    # Clinical severity mapping (International Clinical Diabetic Retinopathy Scale)
    label_map = {
        'no_diabetic_retinopathy': 0,
        'mild_retinopathy': 1,
        'moderate_retinopathy': 2,
        'severe_retinopathy': 3,
        'proliferative_retinopathy': 4
    }
    
    records = []
    print("Extracting images to disk...")
    for idx, row in df.iterrows():
        img_id = f"aptos_{idx:05d}"
        diagnosis = label_map[row['label']]
        diagnosis_name = row['label']
        
        # Save image
        img_bytes = row['image']['bytes']
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_path = os.path.join(img_dir, f"{img_id}.png")
        img.save(img_path, format="PNG")
        
        records.append({
            'id_code': img_id,
            'diagnosis': diagnosis,
            'diagnosis_name': diagnosis_name,
            'file_path': os.path.join("train_images", f"{img_id}.png")
        })
        
        if (idx + 1) % 500 == 0 or (idx + 1) == len(df):
            print(f"  Processed {idx + 1}/{len(df)} images...")
            
    out_df = pd.DataFrame(records)
    csv_path = os.path.join(data_dir, "train.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(out_df)} records to {csv_path}")
    print("\nClass distribution:")
    print(out_df['diagnosis'].value_counts().sort_index())
    print("\nDataset preparation completed successfully!")

if __name__ == "__main__":
    prepare_aptos_dataset()
