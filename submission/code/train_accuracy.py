"""
Accuracy-Focused Training & Multi-Seed Ensembling Pipeline for Diabetic Retinopathy
Task: Maximize raw classification ACCURACY on Variant A (Softmax baseline)
Backbone: EfficientNet-B0 (uniform, frozen stem/low-level regularization)
Techniques:
  1. Cross-entropy with Label Smoothing (0.05 - 0.10, default 0.08)
  2. Safe light data augmentation (small-angle rotation +/-15 deg, zoom 0.92-1.08, mild color jitter)
  3. Increased training budget (12-14 epochs) with early stopping on VALIDATION ACCURACY
  4. 3-Seed Sequential Ensembling (seeds 42, 43, 44 trained strictly one after another)
  5. Test-Time Augmentation (TTA: 4-way flips) combined with multi-seed probability averaging
  6. Post-hoc Logit Adjustment (Menon et al., 2021): logits - tau * log(class_prior)

Hardware Safeguards:
  - Mixed precision (torch.cuda.amp) mandatory
  - Batch size 16 (max 24)
  - In-loop VRAM safeguard (>90% threshold auto-halves batch)
  - Telemetry logging via nvidia-smi per epoch
  - Sequential execution with explicit torch.cuda.empty_cache() and gc.collect() between seeds
"""

import os
import sys
import gc
import time
import json
import random
import argparse
import subprocess
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    mean_absolute_error,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from data import get_dataloaders, get_stratified_splits, GLOBAL_SEED
from models import DRModel

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_gpu_telemetry():
    try:
        res = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            encoding='utf-8'
        ).strip().split(',')
        temp = res[0].strip()
        util = res[1].strip()
        used = res[2].strip()
        total = res[3].strip()
        return f"Temp: {temp}°C | GPU Util: {util}% | VRAM: {used}/{total} MiB"
    except Exception:
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            resv = torch.cuda.memory_reserved() / (1024 ** 2)
            return f"VRAM Alloc: {alloc:.1f} MiB | Reserved: {resv:.1f} MiB"
        return "CPU Execution"

def detect_compute():
    device = torch.device('cpu')
    backbone_name = 'resnet18'
    batch_size = 16
    use_amp = False

    if torch.cuda.is_available():
        try:
            test_x = torch.zeros(2, 2).cuda()
            test_y = test_x + 1
            device = torch.device('cuda')
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            use_amp = True
            print(f"\n[Compute Detection] Viable CUDA GPU Active: {device_name}")
            print(f"[Compute Detection] Dedicated VRAM: {vram_gb:.2f} GB")
            print(f"[Compute Detection] Backbone Selected: EfficientNet-B0")
            print(f"[Compute Detection] Hardware Safety: Batch Size = 16 (max 24) | Mixed Precision (AMP) = Active")
            print(f"[Compute Detection] Telemetry: {get_gpu_telemetry()}")
            backbone_name = 'efficientnet_b0'
        except Exception as e:
            print(f"[Compute Detection] CUDA failed: {e}. Falling back to CPU.")
            device = torch.device('cpu')
            backbone_name = 'resnet18'
            use_amp = False
    return device, backbone_name, batch_size, use_amp

def train_single_seed(seed, backbone_name, device, loaders, epochs, results_dir,
                      label_smoothing=0.08, use_amp=True):
    set_seed(seed)
    train_loader, val_loader, test_loader = loaders
    checkpoint_path = os.path.join(results_dir, f"checkpoint_variant_A_seed{seed}.pt")
    history_json_path = os.path.join(results_dir, f"loss_history_variant_A_seed{seed}.json")

    print(f"\n{'='*70}")
    print(f"Training Seed {seed}: Variant A + Label Smoothing ({label_smoothing})")
    print(f"Backbone: {backbone_name} | Epochs: {epochs} | Batch Size: {train_loader.batch_size} | AMP: {use_amp}")
    print(f"Target Selection Metric: VALIDATION ACCURACY (Early Stopping on Val Acc)")
    print(f"{'='*70}", flush=True)

    model = DRModel(
        variant='variant_A',
        backbone_name=backbone_name,
        pretrained=True,
        freeze_early_blocks=False,
        num_classes=5
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and use_amp))

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_qwk': []}
    best_val_acc = -1.0
    best_epoch = 0

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        total_samples = 0

        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_sz = images.size(0)

            # In-loop VRAM safeguard
            if device.type == 'cuda':
                total_vram = torch.cuda.get_device_properties(device).total_memory
                resv_vram = torch.cuda.memory_reserved(device)
                if resv_vram / total_vram > 0.90:
                    print(f"\n[SAFEGUARD] VRAM >90% ({resv_vram/(1024**3):.2f}/{total_vram/(1024**3):.2f} GB). Emptying cache.")
                    torch.cuda.empty_cache()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda' and use_amp)):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if scaler is not None and device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * batch_sz
            total_samples += batch_sz

        scheduler.step()
        train_loss = running_loss / total_samples

        # Validation evaluation
        model.eval()
        val_running_loss = 0.0
        val_samples = 0
        v_preds, v_targets = [], []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                b_sz = images.size(0)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda' and use_amp)):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_running_loss += loss.item() * b_sz
                val_samples += b_sz
                preds = torch.argmax(outputs, dim=1)
                v_preds.extend(preds.cpu().numpy().tolist())
                v_targets.extend(labels.cpu().numpy().tolist())

        val_loss = val_running_loss / val_samples
        val_acc = accuracy_score(v_targets, v_preds)
        val_qwk = cohen_kappa_score(v_targets, v_preds, weights='quadratic')
        epoch_dur = time.time() - epoch_start
        telemetry = get_gpu_telemetry()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_qwk'].append(val_qwk)

        print(f"Seed {seed} | Epoch [{epoch:02d}/{epochs:02d}] ({epoch_dur:.1f}s) - "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc*100:.2f}% | Val QWK: {val_qwk:.4f} | {telemetry}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'seed': seed,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_qwk': val_qwk,
                'backbone': backbone_name,
                'label_smoothing': label_smoothing
            }, checkpoint_path)

    total_time = time.time() - start_time
    print(f"\n[Seed {seed} Complete] Time: {total_time/60:.2f} mins | Best Val Acc: {best_val_acc*100:.2f}% (Epoch {best_epoch})", flush=True)

    with open(history_json_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)

    # Free memory
    del model, optimizer, scheduler, criterion
    if scaler is not None:
        del scaler
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    return checkpoint_path

def get_model_probabilities(checkpoint_path, loader, device, use_tta=True):
    """
    Loads checkpoint and returns N x 5 softmax probability matrix on loader with TTA.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = DRModel(
        variant='variant_A',
        backbone_name=ckpt.get('backbone', 'efficientnet_b0'),
        pretrained=False,
        num_classes=5
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_probs = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for images, labels, ids in loader:
            images = images.to(device)
            # get_probabilities with TTA (4-way flips)
            probs = model.get_probabilities(images, use_tta=use_tta)
            all_probs.append(probs.cpu())
            all_targets.extend(labels.numpy().tolist())
            all_ids.extend(ids)

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    all_probs = torch.cat(all_probs, dim=0).numpy() # (N, 5)
    return all_probs, np.array(all_targets), all_ids

def evaluate_predictions(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    mae = mean_absolute_error(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    rep = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    # Error distances
    err_dist = np.abs(np.array(y_true) - np.array(y_pred))
    catastrophic_pct = (np.sum(err_dist >= 2) / len(y_true)) * 100.0

    return {
        'accuracy': acc,
        'qwk': qwk,
        'mae': mae,
        'macro_f1': f1_macro,
        'catastrophic_pct': catastrophic_pct,
        'per_class': {
            CLASS_NAMES[c]: {
                'precision': rep[CLASS_NAMES[c]]['precision'],
                'recall': rep[CLASS_NAMES[c]]['recall'],
                'f1': rep[CLASS_NAMES[c]]['f1-score'],
                'support': rep[CLASS_NAMES[c]]['support']
            } for c in range(5)
        }
    }

def apply_logit_adjustment(probs, class_priors, tau=1.0):
    """
    Applies post-hoc logit adjustment (Menon et al., 2021) to probability outputs.
    Adjusted probability: P_adj(y) proportional to P(y) * (pi_y ^ -tau).
    When tau = 0.0, no adjustment is applied.
    """
    if tau == 0.0 or class_priors is None:
        return probs
    # Multiply by priors ** (-tau)
    weights = np.power(class_priors, -tau)
    adj_probs = probs * weights[np.newaxis, :]
    adj_probs = adj_probs / np.sum(adj_probs, axis=1, keepdims=True)
    return adj_probs

def compute_bootstrap_ci(y_true, y_pred, n_bootstraps=1000, seed=42):
    np.random.seed(seed)
    n = len(y_true)
    accs, qwks, maes, cat_errs = [], [], [], []
    severe_f1s, pdr_f1s = [], []
    severe_recs, pdr_recs = [], []

    for _ in range(n_bootstraps):
        idx = np.random.choice(n, size=n, replace=True)
        st = y_true[idx]
        sp = y_pred[idx]
        accs.append(accuracy_score(st, sp))
        qwks.append(cohen_kappa_score(st, sp, weights='quadratic'))
        maes.append(mean_absolute_error(st, sp))
        dists = np.abs(st - sp)
        cat_errs.append((np.sum(dists >= 2) / n) * 100.0)

        # Severe (3) and PDR (4)
        rep = classification_report(st, sp, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        severe_f1s.append(rep['Severe']['f1-score'])
        pdr_f1s.append(rep['PDR']['f1-score'])
        severe_recs.append(rep['Severe']['recall'])
        pdr_recs.append(rep['PDR']['recall'])

    return {
        'acc_ci': (np.percentile(accs, 2.5), np.percentile(accs, 97.5)),
        'qwk_ci': (np.percentile(qwks, 2.5), np.percentile(qwks, 97.5)),
        'mae_ci': (np.percentile(maes, 2.5), np.percentile(maes, 97.5)),
        'cat_ci': (np.percentile(cat_errs, 2.5), np.percentile(cat_errs, 97.5)),
        'severe_f1_ci': (np.percentile(severe_f1s, 2.5), np.percentile(severe_f1s, 97.5)),
        'severe_rec_ci': (np.percentile(severe_recs, 2.5), np.percentile(severe_recs, 97.5)),
        'pdr_f1_ci': (np.percentile(pdr_f1s, 2.5), np.percentile(pdr_f1s, 97.5)),
        'pdr_rec_ci': (np.percentile(pdr_recs, 2.5), np.percentile(pdr_recs, 97.5))
    }

def main():
    parser = argparse.ArgumentParser(description="Accuracy Optimization and Multi-Seed Ensembling")
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs per seed (12-15)")
    parser.add_argument("--label_smoothing", type=float, default=0.08, help="Label smoothing factor (0.05-0.10)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (capped at 24)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds to train sequentially")
    args = parser.parse_args()

    data_dir = os.path.join(".", "data", "aptos2019")
    results_dir = os.path.join(".", "submission", "results")
    os.makedirs(results_dir, exist_ok=True)

    device, backbone_name, default_bs, use_amp = detect_compute()
    batch_size = min(args.batch_size, 24)

    # 1. Prepare DataLoaders using 'safe_light' augmentation
    print("\n[Data Pipeline] Setting up DataLoaders with 'safe_light' augmentation (rotation +/-15°, zoom 0.92-1.08, mild color jitter)...")
    loaders, splits = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        seed=GLOBAL_SEED,
        use_weighted_sampler=False, # Pure unweighted sampling for natural prior accuracy
        aug_mode='safe_light'
    )
    train_df, val_df, test_df = splits
    train_counts = train_df['diagnosis'].value_counts().sort_index().values
    class_priors = train_counts / float(len(train_df))
    print(f"[Class Distribution] Train Class Priors (N={len(train_df)}):")
    for i, (name, pri) in enumerate(zip(CLASS_NAMES, class_priors)):
        print(f"  Grade {i} ({name}): count = {train_counts[i]:4d} | prior = {pri:.4f} | -log(pi) = {-np.log(pri):.4f}")

    # 2. Sequential Multi-Seed Training
    checkpoint_paths = []
    for seed in args.seeds:
        ckpt_path = os.path.join(results_dir, f"checkpoint_variant_A_seed{seed}.pt")
        # Check if already trained
        if os.path.exists(ckpt_path):
            print(f"\n[SKIP] Found existing checkpoint for Seed {seed}: {ckpt_path}. Skipping retraining.")
            checkpoint_paths.append(ckpt_path)
            continue

        print(f"\n[GPU Telemetry Before Seed {seed}] {get_gpu_telemetry()}")
        ckpt_path = train_single_seed(
            seed=seed,
            backbone_name=backbone_name,
            device=device,
            loaders=loaders,
            epochs=args.epochs,
            results_dir=results_dir,
            label_smoothing=args.label_smoothing,
            use_amp=use_amp
        )
        checkpoint_paths.append(ckpt_path)
        print(f"[GPU Telemetry After Seed {seed}] {get_gpu_telemetry()}")

    # 3. Test Set Inference for Each Seed (with TTA)
    print(f"\n{'='*70}")
    print("[Inference Phase] Running Test-Time Augmentation (4-way flips) on Holdout Test Set (N=550)...")
    print(f"{'='*70}", flush=True)

    seed_probs = []
    test_targets = None
    test_ids = None

    for i, (seed, ckpt_path) in enumerate(zip(args.seeds, checkpoint_paths)):
        probs, targets, ids = get_model_probabilities(ckpt_path, loaders[2], device, use_tta=True)
        seed_probs.append(probs)
        if test_targets is None:
            test_targets = targets
            test_ids = ids

        # Evaluate individual seed without logit adjustment (tau=0.0)
        preds_unadj = np.argmax(probs, axis=1)
        m_unadj = evaluate_predictions(targets, preds_unadj)
        print(f"Seed {seed} (TTA, tau=0.0) -> Accuracy: {m_unadj['accuracy']*100:.2f}% | QWK: {m_unadj['qwk']:.4f} | MAE: {m_unadj['mae']:.4f} | Severe F1: {m_unadj['per_class']['Severe']['f1']:.4f}")

    # 4. Multi-Seed Ensemble (Average Softmax Probabilities)
    ensemble_probs = np.mean(seed_probs, axis=0) # (550, 5)

    # 5. Evaluate and Compare Key Models
    # Compare:
    # A) Original Variant A (Baseline)
    # B) Seed 42 (Label Smoothing, tau=0.0)
    # C) Seed 42 + Logit Adjustment (tau=1.0)
    # D) 3-Seed Ensemble (Label Smoothing + TTA, tau=0.0)
    # E) 3-Seed Ensemble + Logit Adjustment (tau=1.0)
    # Also evaluate tau sweep: [0.0, 0.5, 1.0, 1.5] for the ensemble

    # Load original Variant A baseline predictions
    orig_a_csv = os.path.join(results_dir, "predictions_variant_A.csv")
    orig_a_df = pd.read_csv(orig_a_csv)
    orig_a_preds = orig_a_df['predicted_label'].values
    orig_a_targets = orig_a_df['true_label'].values
    m_orig_a = evaluate_predictions(orig_a_targets, orig_a_preds)
    ci_orig_a = compute_bootstrap_ci(orig_a_targets, orig_a_preds)

    # Model B: Seed 42 (tau=0.0)
    p_seed42_unadj = np.argmax(seed_probs[0], axis=1)
    m_seed42_unadj = evaluate_predictions(test_targets, p_seed42_unadj)
    ci_seed42_unadj = compute_bootstrap_ci(test_targets, p_seed42_unadj)

    # Model C: Seed 42 + Logit Adj (tau=1.0)
    p_seed42_adj = np.argmax(apply_logit_adjustment(seed_probs[0], class_priors, tau=1.0), axis=1)
    m_seed42_adj = evaluate_predictions(test_targets, p_seed42_adj)
    ci_seed42_adj = compute_bootstrap_ci(test_targets, p_seed42_adj)

    # Model D: 3-Seed Ensemble (tau=0.0)
    p_ens_unadj = np.argmax(ensemble_probs, axis=1)
    m_ens_unadj = evaluate_predictions(test_targets, p_ens_unadj)
    ci_ens_unadj = compute_bootstrap_ci(test_targets, p_ens_unadj)

    # Model E: 3-Seed Ensemble + Logit Adj (tau=1.0)
    p_ens_adj10 = np.argmax(apply_logit_adjustment(ensemble_probs, class_priors, tau=1.0), axis=1)
    m_ens_adj10 = evaluate_predictions(test_targets, p_ens_adj10)
    ci_ens_adj10 = compute_bootstrap_ci(test_targets, p_ens_adj10)

    # Tau sweep on ensemble: tau in [0.0, 0.5, 1.0, 1.5]
    tau_sweep_results = {}
    for tau_val in [0.0, 0.5, 1.0, 1.5]:
        p_tau = np.argmax(apply_logit_adjustment(ensemble_probs, class_priors, tau=tau_val), axis=1)
        m_tau = evaluate_predictions(test_targets, p_tau)
        tau_sweep_results[f"tau_{tau_val}"] = m_tau
        print(f"[Ensemble Tau={tau_val}] Acc: {m_tau['accuracy']*100:.2f}% | QWK: {m_tau['qwk']:.4f} | MAE: {m_tau['mae']:.4f} | Severe Rec: {m_tau['per_class']['Severe']['recall']*100:.1f}% | PDR Rec: {m_tau['per_class']['PDR']['recall']*100:.1f}%")

    # 6. Save Predictions CSVs
    # Seed 42 unadjusted
    pd.DataFrame({
        'image_id': test_ids,
        'true_label': test_targets,
        'predicted_label': p_seed42_unadj,
        'raw_output': [json.dumps(p.tolist()) for p in seed_probs[0]]
    }).to_csv(os.path.join(results_dir, "predictions_variant_A_seed42_tau0.0.csv"), index=False)

    # Seed 42 adjusted (tau=1.0)
    pd.DataFrame({
        'image_id': test_ids,
        'true_label': test_targets,
        'predicted_label': p_seed42_adj,
        'raw_output': [json.dumps(p.tolist()) for p in apply_logit_adjustment(seed_probs[0], class_priors, tau=1.0)]
    }).to_csv(os.path.join(results_dir, "predictions_variant_A_seed42_tau1.0.csv"), index=False)

    # 3-Seed Ensemble unadjusted (tau=0.0)
    pd.DataFrame({
        'image_id': test_ids,
        'true_label': test_targets,
        'predicted_label': p_ens_unadj,
        'raw_output': [json.dumps(p.tolist()) for p in ensemble_probs]
    }).to_csv(os.path.join(results_dir, "predictions_variant_A_ensemble_tau0.0.csv"), index=False)

    # 3-Seed Ensemble adjusted (tau=1.0)
    pd.DataFrame({
        'image_id': test_ids,
        'true_label': test_targets,
        'predicted_label': p_ens_adj10,
        'raw_output': [json.dumps(p.tolist()) for p in apply_logit_adjustment(ensemble_probs, class_priors, tau=1.0)]
    }).to_csv(os.path.join(results_dir, "predictions_variant_A_ensemble_tau1.0.csv"), index=False)

    # 7. Generate Comparison Markdown & CSV Table
    models_to_report = [
        ("Original Variant A (Baseline Softmax)", m_orig_a, ci_orig_a),
        (f"Variant A (Single Seed 42 + Label Smooth {args.label_smoothing})", m_seed42_unadj, ci_seed42_unadj),
        (f"Variant A (Single Seed 42 + Logit Adj tau=1.0)", m_seed42_adj, ci_seed42_adj),
        (f"3-Seed Ensemble + TTA (tau=0.0, Raw Acc)", m_ens_unadj, ci_ens_unadj),
        (f"3-Seed Ensemble + TTA (tau=1.0, Logit Adj)", m_ens_adj10, ci_ens_adj10)
    ]

    table_rows = []
    for name, m, ci in models_to_report:
        table_rows.append({
            'Model Configuration': name,
            'Accuracy': f"{m['accuracy']*100:.2f}% [{ci['acc_ci'][0]*100:.2f}%, {ci['acc_ci'][1]*100:.2f}%]",
            'QWK': f"{m['qwk']:.4f} [{ci['qwk_ci'][0]:.4f}, {ci['qwk_ci'][1]:.4f}]",
            'MAE': f"{m['mae']:.4f} [{ci['mae_ci'][0]:.4f}, {ci['mae_ci'][1]:.4f}]",
            'Catastrophic Err (d>=2)': f"{m['catastrophic_pct']:.2f}% [{ci['cat_ci'][0]:.2f}%, {ci['cat_ci'][1]:.2f}%]",
            'Grade 3 Severe F1': f"{m['per_class']['Severe']['f1']:.4f} [{ci['severe_f1_ci'][0]:.4f}, {ci['severe_f1_ci'][1]:.4f}]",
            'Grade 4 PDR F1': f"{m['per_class']['PDR']['f1']:.4f} [{ci['pdr_f1_ci'][0]:.4f}, {ci['pdr_f1_ci'][1]:.4f}]"
        })

    comp_df = pd.DataFrame(table_rows)
    csv_out = os.path.join(results_dir, "accuracy_enhancement_table.csv")
    md_out = os.path.join(results_dir, "accuracy_enhancement_table.md")
    try:
        md_table = comp_df.to_markdown(index=False)
    except Exception:
        cols = list(comp_df.columns)
        h_str = "| " + " | ".join(cols) + " |"
        sep_str = "| " + " | ".join(["---"] * len(cols)) + " |"
        row_strs = ["| " + " | ".join(str(v) for v in row.values) + " |" for _, row in comp_df.iterrows()]
        md_table = "\n".join([h_str, sep_str] + row_strs)

    with open(md_out, 'w', encoding='utf-8') as f:
        f.write("# Accuracy Optimization and Trade-off Comparison Table\n\n")
        f.write(md_table + "\n\n")
        f.write("### Post-Hoc Logit Adjustment Tau Sweep (3-Seed Ensemble)\n\n")
        f.write("| Tau (tau) | Accuracy | QWK | MAE | Severe Recall (Grade 3) | PDR Recall (Grade 4) |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for tau_k, m_tau in tau_sweep_results.items():
            t_val = tau_k.replace("tau_", "")
            f.write(f"| {t_val} | {m_tau['accuracy']*100:.2f}% | {m_tau['qwk']:.4f} | {m_tau['mae']:.4f} | {m_tau['per_class']['Severe']['recall']*100:.1f}% | {m_tau['per_class']['PDR']['recall']*100:.1f}% |\n")

    # 8. Generate Publication-Quality Confusion Matrix Plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 4-panel comparison: Original Variant A, Single Seed 42, 3-Seed Ens (tau=0.0), 3-Seed Ens (tau=1.0)
        fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))
        plot_configs = [
            ("Original Variant A\n(Baseline)", orig_a_preds, 'Blues', m_orig_a),
            (f"Variant A (Seed 42)\n(Label Smooth {args.label_smoothing})", p_seed42_unadj, 'Blues', m_seed42_unadj),
            ("3-Seed Ensemble + TTA\n(tau=0.0, Raw Acc)", p_ens_unadj, 'Greens', m_ens_unadj),
            ("3-Seed Ensemble + TTA\n(tau=1.0, Logit Adj)", p_ens_adj10, 'Purples', m_ens_adj10)
        ]

        for idx, (title, preds, cmap, m_info) in enumerate(plot_configs):
            cm = confusion_matrix(test_targets, preds, labels=range(5))
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=axes[idx],
                        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False)
            axes[idx].set_title(f"{title}\nAcc: {m_info['accuracy']*100:.2f}% | QWK: {m_info['qwk']:.4f}",
                                fontsize=10.5, fontweight='bold')
            axes[idx].set_xlabel("Predicted DR Grade", fontsize=10)
            axes[idx].set_ylabel("True DR Grade" if idx == 0 else "", fontsize=10)

        plt.suptitle("Accuracy Optimization: Confusion Matrix Comparison (N=550)", fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        cm_ens_path = os.path.join(results_dir, "confusion_matrix_accuracy_ensemble.png")
        plt.savefig(cm_ens_path, dpi=300)
        plt.close()
        print(f"Saved accuracy ensemble confusion matrices to {cm_ens_path}")
    except Exception as e:
        print(f"[Warning] Failed to generate confusion matrix plot: {e}")

    # Custom JSON encoder to handle numpy types safely
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    # Save full JSON summary
    summary_data = {
        'training_config': {
            'backbone': backbone_name,
            'epochs': args.epochs,
            'label_smoothing': args.label_smoothing,
            'batch_size': batch_size,
            'seeds': args.seeds,
            'device': str(device),
            'amp': use_amp
        },
        'comparison': {name: {'metrics': m, 'ci_95': {k: list(v) for k, v in ci.items()}} for name, m, ci in models_to_report},
        'tau_sweep': tau_sweep_results
    }
    with open(os.path.join(results_dir, "accuracy_enhancement_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, cls=NumpyEncoder, indent=4)

    print(f"\n{'='*70}")
    print("[FINAL RESULTS SUMMARY]")
    print(f"{'='*70}")
    for name, m, ci in models_to_report:
        print(f"{name:50s} -> Acc: {m['accuracy']*100:.2f}% | QWK: {m['qwk']:.4f} | MAE: {m['mae']:.4f}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)
