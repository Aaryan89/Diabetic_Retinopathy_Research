"""
Plot and summarize accuracy enhancement results from saved predictions.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    mean_absolute_error,
    classification_report,
    confusion_matrix,
    f1_score
)

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
results_dir = os.path.join(".", "submission", "results")

# Load prediction files
orig_df = pd.read_csv(os.path.join(results_dir, "predictions_variant_A.csv"))
seed42_df = pd.read_csv(os.path.join(results_dir, "predictions_variant_A_seed42_tau0.0.csv"))
ens_unadj_df = pd.read_csv(os.path.join(results_dir, "predictions_variant_A_ensemble_tau0.0.csv"))
ens_adj_df = pd.read_csv(os.path.join(results_dir, "predictions_variant_A_ensemble_tau1.0.csv"))

test_targets = orig_df['true_label'].values

def eval_df(df):
    preds = df['predicted_label'].values
    acc = accuracy_score(test_targets, preds)
    qwk = cohen_kappa_score(test_targets, preds, weights='quadratic')
    mae = mean_absolute_error(test_targets, preds)
    return preds, acc, qwk, mae

orig_preds, a_acc, a_qwk, a_mae = eval_df(orig_df)
s42_preds, s_acc, s_qwk, s_mae = eval_df(seed42_df)
ens_preds, e_acc, e_qwk, e_mae = eval_df(ens_unadj_df)
adj_preds, adj_acc, adj_qwk, adj_mae = eval_df(ens_adj_df)

print(f"Original Variant A: Acc = {a_acc*100:.2f}%, QWK = {a_qwk:.4f}, MAE = {a_mae:.4f}")
print(f"Seed 42 (tau=0.0):  Acc = {s_acc*100:.2f}%, QWK = {s_qwk:.4f}, MAE = {s_mae:.4f}")
print(f"3-Seed Ens (tau=0): Acc = {e_acc*100:.2f}%, QWK = {e_qwk:.4f}, MAE = {e_mae:.4f}")
print(f"3-Seed Ens (tau=1): Acc = {adj_acc*100:.2f}%, QWK = {adj_qwk:.4f}, MAE = {adj_mae:.4f}")

# 4-panel confusion matrix
fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))
plot_configs = [
    ("Original Variant A\n(Baseline)", orig_preds, 'Blues', a_acc, a_qwk),
    ("Variant A (Seed 42)\n(Label Smooth 0.08)", s42_preds, 'Blues', s_acc, s_qwk),
    ("3-Seed Ensemble + TTA\n(tau=0.0, Raw Acc Focus)", ens_preds, 'Greens', e_acc, e_qwk),
    ("3-Seed Ensemble + TTA\n(tau=1.0, Logit Adj Focus)", adj_preds, 'Purples', adj_acc, adj_qwk)
]

for idx, (title, preds, cmap, acc_val, qwk_val) in enumerate(plot_configs):
    cm = confusion_matrix(test_targets, preds, labels=range(5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=axes[idx],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False)
    axes[idx].set_title(f"{title}\nAcc: {acc_val*100:.2f}% | QWK: {qwk_val:.4f}",
                        fontsize=10.5, fontweight='bold')
    axes[idx].set_xlabel("Predicted DR Grade", fontsize=10)
    axes[idx].set_ylabel("True DR Grade" if idx == 0 else "", fontsize=10)

plt.suptitle("Accuracy Optimization: Confusion Matrix Comparison on Test Split (N=550)", fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
cm_path = os.path.join(results_dir, "confusion_matrix_accuracy_ensemble.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"Saved confusion matrix plot to: {cm_path}")
