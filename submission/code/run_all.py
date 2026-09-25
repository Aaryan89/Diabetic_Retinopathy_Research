"""
Master Execution Script for the Diabetic Retinopathy Ordinal Regression Study.
Orchestrates:
1. Compute Detection & Environment Verification
2. Data Verification & Preprocessing Checks
3. Exploratory Data Analysis (EDA)
4. Multi-Variant Training (Variant A, B, C)
5. Comprehensive Evaluation & Significance Testing
"""

import os
import sys
import time

def main():
    print("=" * 80)
    print("DIABETIC RETINOPATHY ORDINAL REGRESSION STUDY: MASTER PIPELINE")
    print("=" * 80)
    start_total = time.time()

    # Step 1: Run EDA and generate distribution plots
    print("\n>>> STEP 1: Running Exploratory Data Analysis (EDA)...")
    from eda import run_eda
    run_eda()
    print(">>> STEP 1 COMPLETE: EDA charts and summaries saved to ./submission/results/")

    # Step 2: Run Training across all 3 variants
    print("\n>>> STEP 2: Launching Model Training for Variants A, B, and C...")
    from train import main as run_train
    run_train()
    print(">>> STEP 2 COMPLETE: All models trained, checkpoints and predictions saved.")

    # Step 3: Run Evaluation, Analysis, and Statistical Significance Testing
    print("\n>>> STEP 3: Executing Evaluation & Statistical Significance Testing...")
    from evaluate import run_evaluation
    run_evaluation()
    print(">>> STEP 3 COMPLETE: Comparison tables, confusion matrices, and significance results saved.")

    total_time = time.time() - start_total
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time/60:.2f} MINUTES!")
    print("=" * 80)

if __name__ == "__main__":
    main()
