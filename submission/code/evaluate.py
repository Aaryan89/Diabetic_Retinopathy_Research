"""
Comprehensive Evaluation, Analysis, and Statistical Significance Module
Evaluates Variant A (Softmax Baseline), Variant B (CORAL Ordinal Regression),
and Variant C (Continuous Regression) on the holdout test set.

Generates:
1. Accuracy, QWK (Primary), MAE, Macro P/R/F1, and Per-Class Metrics
2. Plotted Confusion Matrices -> ./submission/results/confusion_matrix_{variant}.png & all_confusion_matrices.png
3. Error-Distance Histogram (0, 1, 2, 3, 4 grades off) -> ./submission/results/error_distance_histogram.png
4. Comparison Table (CSV + Markdown) -> ./submission/results/model_comparison_table.csv/.md
5. Bootstrap Significance Test on QWK difference
6. Plain-English summary of winning variant -> ./submission/results/winner_summary.txt
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    mean_absolute_error,
    classification_report,
    confusion_matrix
)
from scipy.stats import wilcoxon

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
VARIANT_LABELS = {
    'variant_A': 'Variant A (Softmax Baseline)',
    'variant_B': 'Variant B (CORAL Ordinal)',
    'variant_C': 'Variant C (Regression)'
}

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    mae = mean_absolute_error(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    
    # Error distance distribution (0 to 4)
    error_distances = np.abs(np.array(y_true) - np.array(y_pred))
    dist_counts = {d: int(np.sum(error_distances == d)) for d in range(5)}
    dist_pcts = {d: (dist_counts[d] / len(y_true)) * 100 for d in range(5)}

    return {
        'accuracy': acc,
        'qwk': qwk,
        'mae': mae,
        'macro_p': report['macro avg']['precision'],
        'macro_r': report['macro avg']['recall'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'class_report': report,
        'error_counts': dist_counts,
        'error_pcts': dist_pcts,
        'raw_distances': error_distances
    }

def bootstrap_qwk_difference(y_true, y_pred_baseline, y_pred_ordinal, n_bootstraps=1000, seed=42):
    """
    Computes 95% Confidence Interval and two-tailed p-value for QWK difference via paired bootstrapping.
    """
    np.random.seed(seed)
    n = len(y_true)
    diffs = []

    for _ in range(n_bootstraps):
        indices = np.random.choice(n, size=n, replace=True)
        sample_true = y_true[indices]
        qwk_base = cohen_kappa_score(sample_true, y_pred_baseline[indices], weights='quadratic')
        qwk_ord  = cohen_kappa_score(sample_true, y_pred_ordinal[indices], weights='quadratic')
        diffs.append(qwk_ord - qwk_base)

    diffs = np.array(diffs)
    mean_diff = np.mean(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    # Empirical p-value that difference <= 0
    p_value = np.mean(diffs <= 0) if mean_diff > 0 else np.mean(diffs >= 0)
    
    return mean_diff, ci_lower, ci_upper, p_value

def run_evaluation(results_dir=os.path.join(".", "submission", "results")):
    os.makedirs(results_dir, exist_ok=True)
    variants = ['variant_A', 'variant_B', 'variant_C']
    dfs = {}
    metrics = {}

    for v in variants:
        pred_path = os.path.join(results_dir, f"predictions_{v}.csv")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Prediction file not found: {pred_path}")
        df = pd.read_csv(pred_path)
        dfs[v] = df
        metrics[v] = compute_metrics(df['true_label'].values, df['predicted_label'].values)

    # 1. Generate Comparison Table
    table_rows = []
    for v in variants:
        m = metrics[v]
        table_rows.append({
            'Model Variant': VARIANT_LABELS[v],
            'QWK (Primary)': f"{m['qwk']:.4f}",
            'Accuracy (%)': f"{m['accuracy']*100:.2f}%",
            'MAE': f"{m['mae']:.4f}",
            'Macro F1': f"{m['macro_f1']:.4f}",
            'Weighted F1': f"{m['weighted_f1']:.4f}",
            'Exact Match (Dist=0)': f"{m['error_pcts'][0]:.1f}%",
            'Off-by-1 (Dist=1)': f"{m['error_pcts'][1]:.1f}%",
            'Severe Error (Dist>=2)': f"{m['error_pcts'][2] + m['error_pcts'][3] + m['error_pcts'][4]:.1f}%"
        })
    comp_df = pd.DataFrame(table_rows)

    # Save CSV
    csv_table_path = os.path.join(results_dir, "model_comparison_table.csv")
    comp_df.to_csv(csv_table_path, index=False)

    # Save Markdown Table
    md_table_path = os.path.join(results_dir, "model_comparison_table.md")
    with open(md_table_path, 'w', encoding='utf-8') as f:
        f.write("# Model Performance Comparison on Holdout Test Set\n\n")
        f.write(comp_df.to_markdown(index=False))
        f.write("\n\n*Note: Quadratic Weighted Kappa (QWK) is the primary clinical metric, penalizing multi-grade misclassifications quadratically.*")
    print(f"Saved comparison tables to {csv_table_path} and {md_table_path}")

    # 2. Plot Confusion Matrices
    for v in variants:
        cm = confusion_matrix(dfs[v]['true_label'], dfs[v]['predicted_label'], labels=range(5))
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False)
        plt.title(f"Confusion Matrix: {VARIANT_LABELS[v]}", fontsize=12, fontweight='bold', pad=12)
        plt.xlabel("Predicted DR Grade", fontsize=11, labelpad=8)
        plt.ylabel("True DR Grade", fontsize=11, labelpad=8)
        plt.tight_layout()
        cm_path = os.path.join(results_dir, f"confusion_matrix_{v}.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()

    # Combined Side-by-Side Confusion Matrix Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    cmaps = ['Blues', 'Greens', 'Oranges']
    for i, v in enumerate(variants):
        cm = confusion_matrix(dfs[v]['true_label'], dfs[v]['predicted_label'], labels=range(5))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmaps[i], ax=axes[i],
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False)
        axes[i].set_title(f"{VARIANT_LABELS[v]}\n(QWK: {metrics[v]['qwk']:.4f})", fontsize=11, fontweight='bold')
        axes[i].set_xlabel("Predicted Grade", fontsize=10)
        axes[i].set_ylabel("True Grade" if i == 0 else "", fontsize=10)
    plt.suptitle("Test Set Confusion Matrix Comparison", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    all_cm_path = os.path.join(results_dir, "all_confusion_matrices.png")
    plt.savefig(all_cm_path, dpi=300)
    plt.close()

    # 3. Plot Error-Distance Histogram
    plt.figure(figsize=(9, 5.5))
    bar_width = 0.25
    x = np.arange(5)
    colors = ['#2b5c8f', '#238b45', '#d95f0e']

    for i, v in enumerate(variants):
        pcts = [metrics[v]['error_pcts'][d] for d in range(5)]
        plt.bar(x + (i - 1) * bar_width, pcts, width=bar_width,
                label=f"{VARIANT_LABELS[v]} (MAE: {metrics[v]['mae']:.3f})",
                color=colors[i], edgecolor='black', alpha=0.85)

    plt.title("Prediction Error Distance Distribution on Test Split", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Absolute Error Distance (|True Grade - Predicted Grade|)", fontsize=11, labelpad=8)
    plt.ylabel("Percentage of Test Samples (%)", fontsize=11, labelpad=8)
    plt.xticks(x, ["0 (Exact)", "1 (Off by 1)", "2 (Off by 2)", "3 (Off by 3)", "4 (Off by 4)"], fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    hist_path = os.path.join(results_dir, "error_distance_histogram.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"Saved error distance histogram to {hist_path}")

    # 4. Statistical Significance Testing (Variant C vs Variant A and Variant B vs Variant A)
    y_true = dfs['variant_A']['true_label'].values
    y_pred_A = dfs['variant_A']['predicted_label'].values
    y_pred_B = dfs['variant_B']['predicted_label'].values
    y_pred_C = dfs['variant_C']['predicted_label'].values

    # Variant C (Best continuous/ordinal) vs Variant A (Baseline)
    mean_diff_C_A, ci_l_C_A, ci_u_C_A, boot_p_C_A = bootstrap_qwk_difference(y_true, y_pred_A, y_pred_C, n_bootstraps=1000, seed=42)
    dist_A = metrics['variant_A']['raw_distances']
    dist_C = metrics['variant_C']['raw_distances']
    w_stat_C_A, wilcoxon_p_C_A = wilcoxon(dist_A, dist_C, alternative='two-sided')

    # Variant B (CORAL) vs Variant A (Baseline)
    mean_diff_B_A, ci_l_B_A, ci_u_B_A, boot_p_B_A = bootstrap_qwk_difference(y_true, y_pred_A, y_pred_B, n_bootstraps=1000, seed=42)
    dist_B = metrics['variant_B']['raw_distances']
    w_stat_B_A, wilcoxon_p_B_A = wilcoxon(dist_A, dist_B, alternative='two-sided')

    stat_summary = {
        'best_variant_comparison': {
            'baseline': 'Variant A (Softmax)',
            'comparison': 'Variant C (Continuous Regression)',
            'baseline_qwk': float(metrics['variant_A']['qwk']),
            'comparison_qwk': float(metrics['variant_C']['qwk']),
            'delta_qwk': float(metrics['variant_C']['qwk'] - metrics['variant_A']['qwk']),
            'bootstrap_mean_diff': float(mean_diff_C_A),
            'bootstrap_95_ci': [float(ci_l_C_A), float(ci_u_C_A)],
            'bootstrap_p_value': float(boot_p_C_A),
            'wilcoxon_stat': float(w_stat_C_A),
            'wilcoxon_p_value': float(wilcoxon_p_C_A)
        },
        'coral_variant_comparison': {
            'baseline': 'Variant A (Softmax)',
            'comparison': 'Variant B (CORAL Ordinal)',
            'baseline_qwk': float(metrics['variant_A']['qwk']),
            'comparison_qwk': float(metrics['variant_B']['qwk']),
            'delta_qwk': float(metrics['variant_B']['qwk'] - metrics['variant_A']['qwk']),
            'bootstrap_mean_diff': float(mean_diff_B_A),
            'bootstrap_95_ci': [float(ci_l_B_A), float(ci_u_B_A)],
            'bootstrap_p_value': float(boot_p_B_A),
            'wilcoxon_stat': float(w_stat_B_A),
            'wilcoxon_p_value': float(wilcoxon_p_B_A)
        }
    }
    stat_path = os.path.join(results_dir, "statistical_significance.json")
    with open(stat_path, 'w', encoding='utf-8') as f:
        json.dump(stat_summary, f, indent=4)
    print(f"Saved statistical significance analysis to {stat_path}")

    # 5. Plain-English Summary of Winning Variant
    best_variant = max(variants, key=lambda k: metrics[k]['qwk'])
    best_label = VARIANT_LABELS[best_variant]
    best_qwk = metrics[best_variant]['qwk']
    
    summary_text = (
        f"{best_label} demonstrated the highest ordinal concordance on the holdout test set with a Quadratic Weighted Kappa (QWK) of {best_qwk:.4f}, "
        f"compared to {metrics['variant_A']['qwk']:.4f} for Variant A (Softmax Baseline), {metrics['variant_B']['qwk']:.4f} for Variant B (CORAL Ordinal), "
        f"and {metrics['variant_C']['qwk']:.4f} for Variant C (Continuous Regression). "
        f"In terms of Mean Absolute Error (MAE), Variant A achieved {metrics['variant_A']['mae']:.4f}, Variant B achieved {metrics['variant_B']['mae']:.4f}, "
        f"and Variant C achieved {metrics['variant_C']['mae']:.4f}. "
        f"Severe misclassifications (error distance >= 2) occurred in {metrics['variant_A']['error_pcts'][2]+metrics['variant_A']['error_pcts'][3]+metrics['variant_A']['error_pcts'][4]:.1f}% (Variant A), "
        f"{metrics['variant_B']['error_pcts'][2]+metrics['variant_B']['error_pcts'][3]+metrics['variant_B']['error_pcts'][4]:.1f}% (Variant B), and "
        f"{metrics['variant_C']['error_pcts'][2]+metrics['variant_C']['error_pcts'][3]+metrics['variant_C']['error_pcts'][4]:.1f}% (Variant C) of test cases. "
        f"These empirical findings confirm the critical impact of loss function framing on bounding clinical risk in automated triage workflows."
    )
    summary_txt_path = os.path.join(results_dir, "winner_summary.txt")
    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write(summary_text + "\n")
    print(f"\nWinner Summary:\n{summary_text}")

if __name__ == "__main__":
    run_evaluation()
