"""
Exploratory Data Analysis (EDA) Script for APTOS 2019 Blindness Detection
Generates:
1. Class distribution bar chart -> ./submission/results/class_distribution.png
2. Original vs Preprocessed sample images per class -> ./submission/results/sample_images_per_class.png
3. Dataset distribution and split summary tables -> ./submission/results/dataset_split_summary.csv
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Import preprocessing from data.py
from data import preprocess_retinal_image, get_stratified_splits, GLOBAL_SEED

def run_eda(data_dir=os.path.join(".", "data", "aptos2019"),
            results_dir=os.path.join(".", "submission", "results")):
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "train.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    total_samples = len(df)
    print(f"Loaded {total_samples} samples from {csv_path}")

    # Standard ICDR DR severity naming
    class_names = [
        "0 - No DR",
        "1 - Mild NPDR",
        "2 - Moderate NPDR",
        "3 - Severe NPDR",
        "4 - Proliferative DR"
    ]
    
    counts = df['diagnosis'].value_counts().sort_index()
    percentages = (counts / total_samples) * 100

    print("\n--- APTOS 2019 Class Distribution ---")
    for grade, count in counts.items():
        print(f"  Class {grade} ({class_names[grade]}): {count} images ({percentages[grade]:.2f}%)")

    # 1. Plot Class Distribution Bar Chart
    plt.figure(figsize=(10, 6))
    colors = ['#2b5c8f', '#3690c0', '#67a9cf', '#ef8a62', '#b2182b']
    bars = plt.bar(range(5), counts.values, color=colors, edgecolor='black', linewidth=1.2, width=0.6)
    
    plt.title("APTOS 2019 Diabetic Retinopathy Class Distribution", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("International Clinical Diabetic Retinopathy (ICDR) Severity Grade", fontsize=12, labelpad=10)
    plt.ylabel("Number of Fundus Images", fontsize=12, labelpad=10)
    plt.xticks(range(5), class_names, fontsize=10, rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.ylim(0, max(counts.values) * 1.15)

    # Add count and percentage labels above each bar
    for bar, count, pct in zip(bars, counts.values, percentages.values):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 35,
                 f"{count:,}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    chart_path = os.path.join(results_dir, "class_distribution.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\nSaved class distribution chart to {chart_path}")

    # 2. Stratified Splits Summary
    train_df, val_df, test_df = get_stratified_splits(csv_path, seed=GLOBAL_SEED)
    
    split_summary = pd.DataFrame({
        'Grade': range(5),
        'Severity Name': class_names,
        'Full Dataset': counts.values,
        'Train (70%)': train_df['diagnosis'].value_counts().sort_index().values,
        'Val (15%)': val_df['diagnosis'].value_counts().sort_index().values,
        'Test (15%)': test_df['diagnosis'].value_counts().sort_index().values,
    })
    split_summary_path = os.path.join(results_dir, "dataset_split_summary.csv")
    split_summary.to_csv(split_summary_path, index=False)
    print(f"Saved stratified split summary to {split_summary_path}")
    print("\nSplit Summary Table:\n", split_summary.to_string(index=False))

    # 3. Plot Sample Images Per Class (Original vs Preprocessed with CLAHE & Crop)
    fig, axes = plt.subplots(5, 4, figsize=(14, 16))
    fig.suptitle("APTOS 2019 Fundus Images: Raw Input vs. Preprocessed (Circular Crop + CLAHE)",
                 fontsize=15, fontweight='bold', y=0.99)

    for grade in range(5):
        grade_samples = df[df['diagnosis'] == grade].head(2)
        for sample_idx, (_, row) in enumerate(grade_samples.iterrows()):
            img_rel_path = row['file_path'] if 'file_path' in row else os.path.join("train_images", f"{row['id_code']}.png")
            img_full_path = os.path.join(data_dir, img_rel_path)
            
            raw_rgb = np.array(Image.open(img_full_path).convert("RGB"))
            prep_rgb = preprocess_retinal_image(raw_rgb, target_size=(224, 224))

            col_raw = sample_idx * 2
            col_prep = sample_idx * 2 + 1

            # Raw image plot
            axes[grade, col_raw].imshow(raw_rgb)
            axes[grade, col_raw].set_title(f"{class_names[grade]}\n[Raw #{sample_idx+1}]", fontsize=9)
            axes[grade, col_raw].axis('off')

            # Preprocessed image plot
            axes[grade, col_prep].imshow(prep_rgb)
            axes[grade, col_prep].set_title(f"{class_names[grade]}\n[Crop + CLAHE]", fontsize=9, color='#08519c', fontweight='bold')
            axes[grade, col_prep].axis('off')

    plt.tight_layout()
    sample_img_path = os.path.join(results_dir, "sample_images_per_class.png")
    plt.savefig(sample_img_path, dpi=300)
    plt.close()
    print(f"Saved sample comparison image to {sample_img_path}")

if __name__ == "__main__":
    run_eda()
