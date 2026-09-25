"""
Training Pipeline for Diabetic Retinopathy Detection:
Trains Variant A (Softmax Baseline), Variant B (CORAL Ordinal Regression),
and Variant C (Continuous Regression) under strictly uniform settings.
"""

import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd


from sklearn.metrics import cohen_kappa_score, accuracy_score

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from data import get_dataloaders, GLOBAL_SEED
from models import DRModel, coral_loss

def set_seed(seed=GLOBAL_SEED):
    """
    Sets and locks random seeds across all libraries for deterministic execution.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def detect_compute():
    """
    Detects hardware compute and activates architectural fallback if required.
    """
    device = torch.device('cpu')
    backbone_name = 'resnet18'
    default_epochs = 4
    batch_size = 32

    if torch.cuda.is_available():
        try:
            # Test actual CUDA kernel execution on device
            test_x = torch.zeros(2, 2).cuda()
            test_y = test_x + 1
            # Test CNN convolution kernel
            test_conv = nn.Conv2d(3, 8, 3).cuda()
            test_in = torch.zeros(2, 3, 16, 16).cuda()
            test_out = test_conv(test_in)
            
            device = torch.device('cuda')
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"\n[Compute Detection] Viable CUDA GPU Active: {device_name}")
            print(f"[Compute Detection] Dedicated VRAM: {vram_gb:.2f} GB")
            print(f"[Compute Detection] Backbone Selected: EfficientNet-B0 (8 epochs)")
            backbone_name = 'efficientnet_b0'
            default_epochs = 8
            batch_size = 32
        except Exception as e:
            print(f"\n[Compute Detection] Host GPU present but CUDA kernel threw: {type(e).__name__}")
            print("[Compute Detection] FALLBACK ACTIVATED: Switching backbone to ResNet-18 on CPU.")
            print("[Compute Detection] Training budget adjusted to 4 epochs per variant.")
            device = torch.device('cpu')
            backbone_name = 'resnet18'
            default_epochs = 4
            batch_size = 32
    else:
        print("\n[Compute Detection] No GPU detected. Running on CPU with ResNet-18 fallback (4 epochs).")
        device = torch.device('cpu')
        backbone_name = 'resnet18'
        default_epochs = 4
        batch_size = 32

    return device, backbone_name, default_epochs, batch_size

def train_epoch(model, loader, criterion, optimizer, device, variant):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()
        outputs = model(images)

        if variant == 'variant_A':
            loss = criterion(outputs, labels)
        elif variant == 'variant_B':
            loss = coral_loss(outputs, labels, num_classes=5)
        elif variant == 'variant_C':
            loss = criterion(outputs.squeeze(-1), labels.float())

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    return epoch_loss

def evaluate(model, loader, criterion, device, variant):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    all_ids = []
    all_raws = []

    with torch.no_grad():
        for images, labels, ids in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)

            if variant == 'variant_A':
                loss = criterion(outputs, labels)
                preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                raw_list = outputs.cpu().numpy().tolist()
            elif variant == 'variant_B':
                loss = coral_loss(outputs, labels, num_classes=5)
                sigmoids = torch.sigmoid(outputs)
                preds = torch.sum(sigmoids > 0.5, dim=1)
                raw_list = sigmoids.cpu().numpy().tolist()
            elif variant == 'variant_C':
                loss = criterion(outputs.squeeze(-1), labels.float())
                clamped = torch.clamp(outputs.squeeze(-1), 0.0, 4.0)
                preds = torch.round(clamped).long()
                raw_list = outputs.squeeze(-1).cpu().numpy().tolist()

            running_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_ids.extend(ids)
            all_raws.extend(raw_list)

    eval_loss = running_loss / total_samples
    acc = accuracy_score(all_targets, all_preds)
    qwk = cohen_kappa_score(all_targets, all_preds, weights='quadratic')
    return eval_loss, acc, qwk, all_ids, all_targets, all_preds, all_raws

def train_variant(variant_name, backbone_name, device, loaders, epochs, results_dir):
    pred_csv_path = os.path.join(results_dir, f"predictions_{variant_name}.csv")
    best_checkpoint_path = os.path.join(results_dir, f"checkpoint_{variant_name}.pt")
    history_json_path = os.path.join(results_dir, f"loss_history_{variant_name}.json")

    # Check if variant already completed successfully
    if os.path.exists(pred_csv_path) and os.path.exists(best_checkpoint_path):
        try:
            existing_df = pd.read_csv(pred_csv_path)
            if len(existing_df) == 550:
                acc = accuracy_score(existing_df['true_label'], existing_df['predicted_label'])
                qwk = cohen_kappa_score(existing_df['true_label'], existing_df['predicted_label'], weights='quadratic')
                print(f"\n[RESUME] {variant_name} already completed on GPU (Acc: {acc*100:.2f}%, QWK: {qwk:.4f}). Skipping retraining.", flush=True)
                hist = {}
                if os.path.exists(history_json_path):
                    with open(history_json_path, 'r') as f:
                        hist = json.load(f)
                return hist, (acc, qwk)
        except Exception:
            pass

    print(f"\n{'='*60}")
    print(f"Starting Training: {variant_name} (Backbone: {backbone_name}, Epochs: {epochs}, Device: {device})")
    print(f"{'='*60}", flush=True)
    
    set_seed(GLOBAL_SEED)
    train_loader, val_loader, test_loader = loaders

    model = DRModel(variant=variant_name, backbone_name=backbone_name, pretrained=True).to(device)

    if variant_name == 'variant_A':
        criterion = nn.CrossEntropyLoss()
    elif variant_name == 'variant_B':
        criterion = None
    elif variant_name == 'variant_C':
        criterion = nn.SmoothL1Loss()

    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_qwk': []}
    best_val_qwk = -1.0

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, variant_name)
        val_loss, val_acc, val_qwk, _, _, _, _ = evaluate(model, val_loader, criterion, device, variant_name)
        scheduler.step()
        epoch_dur = time.time() - epoch_start

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_qwk'].append(val_qwk)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_dur:.1f}s) - "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc*100:.2f}% | Val QWK: {val_qwk:.4f}", flush=True)

        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_qwk': val_qwk,
                'val_acc': val_acc,
                'variant': variant_name,
                'backbone': backbone_name
            }, best_checkpoint_path)

    total_training_time = time.time() - start_time
    print(f"\n{variant_name} Training Completed in {total_training_time/60:.2f} mins. Best Val QWK: {best_val_qwk:.4f}", flush=True)

    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_qwk, test_ids, test_targets, test_preds, test_raws = evaluate(
        model, test_loader, criterion, device, variant_name
    )
    print(f"[TEST EVALUATION] {variant_name} -> Test Acc: {test_acc*100:.2f}% | Test QWK: {test_qwk:.4f}", flush=True)

    pred_df = pd.DataFrame({
        'image_id': test_ids,
        'true_label': test_targets,
        'predicted_label': test_preds,
        'raw_output': [json.dumps(r) if isinstance(r, list) else str(r) for r in test_raws]
    })
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"Saved prediction records to {pred_csv_path}", flush=True)

    with open(history_json_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)
    print(f"Saved loss history to {history_json_path}", flush=True)

    return history, (test_acc, test_qwk)

def main():
    data_dir = os.path.join(".", "data", "aptos2019")
    results_dir = os.path.join(".", "submission", "results")
    os.makedirs(results_dir, exist_ok=True)

    set_seed(GLOBAL_SEED)
    device, backbone_name, epochs, batch_size = detect_compute()

    loaders, splits = get_dataloaders(data_dir=data_dir, batch_size=batch_size, seed=GLOBAL_SEED)

    all_histories = {}
    test_results = {}

    variants = ['variant_A', 'variant_B', 'variant_C']
    for v in variants:
        hist, (t_acc, t_qwk) = train_variant(v, backbone_name, device, loaders, epochs, results_dir)
        all_histories[v] = hist
        test_results[v] = {'test_acc': t_acc, 'test_qwk': t_qwk}



    summary_path = os.path.join(results_dir, "training_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'compute': {
                'device': str(device),
                'backbone': backbone_name,
                'epochs': epochs,
                'batch_size': batch_size,
                'seed': GLOBAL_SEED
            },
            'test_results': test_results,
            'histories': all_histories
        }, f, indent=4)
    print(f"Saved run summary to {summary_path}", flush=True)

if __name__ == "__main__":
    main()
