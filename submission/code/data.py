"""
Data Preprocessing, Augmentation, and Dataset/DataLoader Module for APTOS 2019
Diabetic Retinopathy Ordinal Regression Study.

Pipeline:
1. Circular Crop to eliminate non-retinal black borders
2. Resize to 224x224
3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
4. ImageNet Normalization
5. Stratified 70/15/15 Train/Val/Test Split with Fixed Seed (42)
6. Data Augmentation (Train only): Flip, Rotation, Color/Brightness Jitter
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    import skimage.exposure as exposure

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

GLOBAL_SEED = 42

def crop_image_from_gray(img_rgb, tol=7):
    if img_rgb.ndim == 2:
        mask = img_rgb > tol
        return img_rgb[np.ix_(mask.any(1), mask.any(0))]
    elif img_rgb.ndim == 3:
        gray = np.mean(img_rgb, axis=2)
        mask = gray > tol
        if not mask.any():
            return img_rgb
        row_mask = mask.any(1)
        col_mask = mask.any(0)
        return img_rgb[np.ix_(row_mask, col_mask)]
    return img_rgb

def apply_clahe(img_rgb, clip_limit=2.0, tile_grid_size=(8, 8)):
    if HAS_CV2:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_float = img_rgb.astype(np.float32) / 255.0
        enhanced_float = exposure.equalize_adapthist(img_float, clip_limit=clip_limit / 100.0)
        return (enhanced_float * 255.0).astype(np.uint8)

def preprocess_retinal_image(image_input, target_size=(224, 224)):
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("RGB")
        img_rgb = np.array(pil_img)
    elif isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        img_rgb = image_input if image_input.shape[-1] == 3 else np.stack([image_input]*3, axis=-1)
    else:
        raise TypeError("Unsupported image input type")

    cropped_rgb = crop_image_from_gray(img_rgb)
    if cropped_rgb.size == 0 or cropped_rgb.shape[0] < 10 or cropped_rgb.shape[1] < 10:
        cropped_rgb = img_rgb

    if HAS_CV2:
        resized_rgb = cv2.resize(cropped_rgb, target_size, interpolation=cv2.INTER_AREA)
    else:
        pil_crop = Image.fromarray(cropped_rgb)
        resized_rgb = np.array(pil_crop.resize(target_size, Image.Resampling.BILINEAR))

    enhanced_rgb = apply_clahe(resized_rgb, clip_limit=2.0, tile_grid_size=(8, 8))
    return enhanced_rgb

class APTOSDataset(Dataset):
    """
    PyTorch Dataset for APTOS 2019 Diabetic Retinopathy Fundus Images.
    Supports fast loading from disk cache with on-the-fly train augmentation.
    """
    def __init__(self, df, data_dir, is_train=False, target_size=(224, 224)):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.is_train = is_train
        self.target_size = target_size
        self.cache_dir = os.path.join(data_dir, "preprocessed_224")

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if self.is_train:
            self.aug = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=25),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.aug = transforms.Compose([
                transforms.ToTensor(),
                self.normalize
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['id_code']
        cached_path = os.path.join(self.cache_dir, f"{img_id}.png")

        if os.path.exists(cached_path):
            img = Image.open(cached_path).convert("RGB")
        else:
            raw_path = os.path.join(self.data_dir, "train_images", f"{img_id}.png")
            prep_rgb = preprocess_retinal_image(raw_path, target_size=self.target_size)
            img = Image.fromarray(prep_rgb)

        tensor_img = self.aug(img)
        label = int(row['diagnosis'])
        return tensor_img, label, img_id

def get_stratified_splits(csv_path, seed=GLOBAL_SEED):
    df = pd.read_csv(csv_path)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df['diagnosis']
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df['diagnosis']
    )
    train_df = train_df.copy().reset_index(drop=True)
    val_df = val_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    return train_df, val_df, test_df

def get_dataloaders(data_dir=os.path.join(".", "data", "aptos2019"),
                    batch_size=32,
                    num_workers=0,
                    seed=GLOBAL_SEED):
    csv_path = os.path.join(data_dir, "train.csv")
    train_df, val_df, test_df = get_stratified_splits(csv_path, seed=seed)

    train_dataset = APTOSDataset(train_df, data_dir, is_train=True)
    val_dataset   = APTOSDataset(val_df, data_dir, is_train=False)
    test_dataset  = APTOSDataset(test_df, data_dir, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return (train_loader, val_loader, test_loader), (train_df, val_df, test_df)
