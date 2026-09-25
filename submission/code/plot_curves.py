"""
Standalone Loss Curves Plotting Script (Runs via Anaconda Python)
Reads training histories from results directory and generates high-resolution loss curve PNGs.
"""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_loss_plots(results_dir="./submission/results"):
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "training_summary.json")

    histories = {}
    variants = ['variant_A', 'variant_B', 'variant_C']
    labels = {
        'variant_A': 'Variant A (Softmax Baseline)',
        'variant_B': 'Variant B (CORAL Ordinal Regression)',
        'variant_C': 'Variant C (Continuous Regression)'
    }
    colors = {
        'variant_A': '#2b5c8f',
        'variant_B': '#238b45',
        'variant_C': '#d95f0e'
    }

    # Load histories from individual json files if available
    for v in variants:
        v_path = os.path.join(results_dir, f"loss_history_{v}.json")
        if os.path.exists(v_path):
            with open(v_path, 'r', encoding='utf-8') as f:
                histories[v] = json.load(f)

    # Fallback to summary json
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
            if 'histories' in summary:
                for k, v in summary['histories'].items():
                    if k not in histories:
                        histories[k] = v

    # 1. Plot individual loss curves
    for v, hist in histories.items():
        if not hist.get('train_loss'):
            continue
        epochs = len(hist['train_loss'])
        x = range(1, epochs + 1)
        
        plt.figure(figsize=(8, 5))
        plt.plot(x, hist['train_loss'], label='Training Loss', color=colors.get(v, '#2b5c8f'), lw=2.2)
        plt.plot(x, hist['val_loss'], label='Validation Loss', color='#b2182b', lw=2.2, linestyle='--')
        plt.title(f"Training & Validation Loss: {labels.get(v, v)}", fontsize=13, fontweight='bold', pad=12)
        plt.xlabel("Epoch", fontsize=11, labelpad=8)
        plt.ylabel("Loss", fontsize=11, labelpad=8)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        out_path = os.path.join(results_dir, f"loss_curves_{v}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Generated individual loss plot: {out_path}")

    # 2. Plot combined validation loss comparison
    if len(histories) >= 2:
        plt.figure(figsize=(9, 5.5))
        for v in variants:
            if v in histories and histories[v].get('val_loss'):
                epochs = len(histories[v]['val_loss'])
                x = range(1, epochs + 1)
                plt.plot(x, histories[v]['val_loss'], label=f"{labels[v]} (Val Loss)",
                         color=colors[v], lw=2.2)

        plt.title("Validation Loss Comparison Across All 3 Variants (EfficientNet-B0)", fontsize=13, fontweight='bold', pad=12)
        plt.xlabel("Epoch", fontsize=11, labelpad=8)
        plt.ylabel("Validation Loss", fontsize=11, labelpad=8)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        all_curves_path = os.path.join(results_dir, "all_loss_curves.png")
        plt.savefig(all_curves_path, dpi=300)
        plt.close()
        print(f"Generated combined loss plot: {all_curves_path}")

if __name__ == "__main__":
    generate_loss_plots()
