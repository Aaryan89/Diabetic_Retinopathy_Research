# Ordinal Neural Networks for Diabetic Retinopathy Staging

> **Enforcing Monotonic Rank Consistency for Safety-Critical Clinical Decision Support**

Diabetic Retinopathy (DR) grading is a safety-critical classification task where standard neural network loss functions (like categorical cross-entropy) fail to capture the ordinal severity of the disease. In a clinical setting, confusing vision-threatening *Proliferative DR (Grade 4)* with *Severe NPDR (Grade 3)* is a minor triage error, but confusing it with *No DR (Grade 0)* is a catastrophic failure that could result in irreversible blindness.

This project implements and evaluates distance-aware, rank-consistent ordinal formulations on the **APTOS 2019 Blindness Detection** dataset to strictly penalize catastrophic multi-grade misclassifications. 

---

## 👥 Project Team & Academic Details

**Institution:** Department of Computer Science and Engineering, MIT World Peace University, Pune  
**Course:** Artificial Intelligence & Expert Systems Laboratory (CSE30070)  
**Academic Year:** 2026–2027  
**Project Mentor:** Prof. Pramod Mali  

| Sr. No. | PRN | Student Name |
| :---: | :---: | :--- |
| 1 | **1262241524** | Nayna Sharma |
| 2 | **1262241534** | Harshad Pardhi |
| 3 | **1262241729** | Aaryan Kumbhare |
| 4 | **1262241775** | Anurag Harapanahalli |

---

## 🎯 Key Breakthroughs

1. **Resolution of Minority-Stage Collapse:** By replacing standard categorical Softmax with **Conditional Ordinal Regression (Variant D - CORN)**, Severe NPDR (Grade 3) sensitivity surged to **65.7%** (compared to 0.0% in unweighted CORAL), resolving minority-stage collapse and slashing catastrophic multi-grade triage errors ($|y - \hat{y}| \ge 2$) to **5.10%**.
2. **Featured Multi-Seed Ensemble (Variant E):** Combining 3-seed ensembling with 4-way flip TTA, label smoothing ($\alpha = 0.08$), and post-hoc logit adjustment achieved the highest classification accuracy of **83.82%** (vs. 83.45% for the best single seed) and stabilized stochastic training variance (single-seed mean 82.73% $\rightarrow$ 83.82%), while reaching a peak **QWK of 0.8929** at $\tau = 0.5$ in sensitivity testing.
3. **Clinical Referral Triage Insights:** Comprehensive triage analysis demonstrated clear clinical trade-offs: Variant E minimized unnecessary secondary clinic over-referrals to just **2.52%** (12/477), whereas Variant D (CORN) minimized dangerous under-triage of sight-threatening disease to **15.07%** (11/73).
4. **Hardware Efficiency:** The entire pipeline operates on a lightweight **EfficientNet-B0 backbone (5.3M parameters)**, training smoothly under 1 GB VRAM limits, proving clinical-grade performance is possible on portable hardware without massive parameter bloat.

---

## ⚙️ System Architecture

Our clinical pipeline is divided into three synchronized partitions: Primary Screening Clinic, Automated AI Diagnostic Engine, and Clinical Triage & Specialist Referral.

![System Architecture](submission/results/system_architecture.png)
*Fig: End-to-End System Architecture Diagram: Preprocessing, EfficientNet-B0 Backbone, Ordinal Inference Pathways, and Clinical Decision Support Triage Rules.*

**Backbone Optimization Strategy:** The backbone is an ImageNet-pretrained EfficientNet-B0 (5.3M parameters). While Variants A, B, C, and E are fine-tuned end-to-end with all layers unfrozen, Variant D (CORN) freezes the stem and first three MBConv stages (`features[:4]`) to retain general low-level edge filters and prevent feature drift under minority-class reweighting.

---

## 🔬 Mathematical Formulations

To ensure robust readability, key equations optimizing clinical safety are highlighted below:

### 1. The Metric Blindness of Softmax (Baseline)
Standard cross-entropy optimizes independent nominal probabilities without any concept of rank. The loss gradient is symmetric with respect to all non-target classes, providing **zero mathematical incentive** to prefer a harmless off-by-one error over a life-threatening off-by-four error.
<p align="center">
  <b><i>L<sub>CE</sub></i> = − ∑ <i>y<sub>k</sub></i> log(<i>p̂<sub>k</sub></i>)</b>
</p>

### 2. Conditional Ordinal Regression for Neural Networks (CORN)
Instead of predicting unconditioned probabilities, CORN models the step-by-step conditional probability of exceeding rank $k$, given that the previous rank was already reached. Crucially, a healthy patient updates only the first task, preventing overwhelming majority-class gradients from crushing the decision intervals for severe minority classes.
<p align="center">
  <b><i>L<sub>CORN</sub>(x, y)</i> = [ − 1 / min(y + 1, K − 1) ] ∑ BCE( <i>g<sub>k</sub>(x)</i>, 𝕀(<i>y > k</i>) )</b>
</p>

### 3. Differentiable Soft-QWK Loss
To align training directly with our clinical target metric (Quadratic Weighted Kappa), we construct soft observed and chance agreement matrices over each mini-batch:
<p align="center">
  <b><i>L<sub>SoftQWK</sub></i> = ( ∑ <i>w<sub>i,j</sub> O<sub>i,j</sub></i> ) / ( ∑ <i>w<sub>i,j</sub> E<sub>i,j</sub></i> + <i>ε</i> )</b>
</p>
<p align="center"><i>Where weights are quadratic penalties based on distance:</i> <b><i>w<sub>i,j</sub></i> = (<i>i − j</i>)² / (<i>K − 1</i>)²</b></p>

The total optimization objective dynamically balances rank consistency and kappa maximization: **Total Loss = <i>L<sub>CORN</sub></i> + <i>λ</i> · <i>L<sub>SoftQWK</sub></i>**

---

## 📊 Evaluation & Results

All models were evaluated on a locked holdout test set of 550 images.

### Prediction Error Distance 
The histogram below reveals how Variant D (CORN) and Continuous Regression tighten the error distribution, pulling misclassifications toward the true diagonal (Exact Match or Off-by-1) and eliminating off-by-3 or off-by-4 errors entirely.

![Error Distance Histogram](submission/results/error_distance_histogram.png)

### Side-by-Side Confusion Matrices
![All Confusion Matrices](submission/results/all_confusion_matrices.png)

### Main Clinical Performance Comparison (Holdout Test Set, N = 550)

| Model Variant / Configuration | QWK | Accuracy (%) | MAE | Catastrophic Error ($d \ge 2$) | Severe F1 (Grade 3) | PDR F1 (Grade 4) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant A: Softmax Baseline** | 0.8724 | 80.18% | 0.2673 | 5.64% | 0.3810 | 0.5169 |
| **Variant B: CORAL Ordinal** | 0.7273 | 59.64% | 0.6964 | 26.40% | 0.0000 | 0.3692 |
| **Variant C: Continuous Regression** | 0.8788 | 75.64% | 0.2873 | **3.80%** | 0.2528 | 0.3589 |
| **Variant D (CORN): Conditional Ordinal** | 0.8482 | 69.09% | 0.3764 | 5.10% | 0.2724 | 0.4584 |
| **Variant E: 3-Seed Ensemble + TTA** | **0.8859** | **83.82%** | **0.2200** | 4.36% | **0.4561** | **0.5570** |

*Note: Best value in each column is shown in bold. Variant E achieves the highest accuracy (83.82%) and stabilizes performance across stochastic training runs (single-seed mean 82.73% $\rightarrow$ 83.82%), with QWK (0.8859) and MAE (0.2200) comparable to the best individual run (Seed 44: QWK 0.8901, MAE 0.2200, Catastrophic Error 4.18%). Variant D delivers the highest Severe NPDR recall (65.7%), and Variant C yields the lowest catastrophic error rate (3.80%).*

### Clinical Referral Triage Performance (N = 550)

| Model Variant | Under-Triage Count ($y \ge 3 \to \hat{y} \le 2$) | Under-Triage Rate (%) | Over-Referral Count ($y \le 2 \to \hat{y} \ge 3$) | Over-Referral Rate (%) | Clinical Trade-off Profile |
| :--- | :---: | :---: | :---: | :---: | :---|
| **Variant A: Softmax Baseline** | 21 / 73 | 28.77% | 27 / 477 | 5.66% | Unweighted multiclass baseline |
| **Variant B: CORAL Ordinal** | 8 / 73 | 10.96% | 96 / 477 | 20.13% | High false referrals from class collapse |
| **Variant C: Continuous Regression** | 23 / 73 | 31.51% | 30 / 477 | 6.29% | Conservative; bounds multi-grade drift |
| **Variant D (CORN): Conditional Ordinal** | 11 / 73 | **15.07%** | 82 / 477 | 17.19% | High-sensitivity screening specialist |
| **Variant E: 3-Seed Ens + TTA ($\tau = 0.0$)** | 22 / 73 | 30.14% | 12 / 477 | **2.52%** | Minimum false referrals; confirmatory focus |
| **Variant E: 3-Seed Ens + TTA ($\tau = 1.0$)** | 14 / 73 | 19.18% | 44 / 477 | 9.22% | Balanced trade-off via logit adjustment |

*Note: Sight-threatening cases requiring urgent referral (Grades 3–4): N = 73; non-referral cases for routine monitoring (Grades 0–2): N = 477.*

---

## 📐 System Design (UML)

### Activity Workflow
![UML Activity Diagram](submission/results/uml_activity_diagram.png)

### Class Architecture
![UML Class Diagram](submission/results/uml_class_diagram.png)

---

## 📂 Repository Layout

```text
aiesl_main/
│
├── submission/
│   ├── code/                      # Conversion scripts and utilities
│   ├── results/                   # Evaluation plots, matrices, and UML diagrams
│   ├── summaries/                 # Distilled research paper summaries
│   ├── final_paper.md             # Core markdown manuscript (Source of truth)
│   ├── final_paper.docx           # Typeset IEEE-format Word document with OMML math
│   ├── final_paper.pdf            # Final compiled PDF manuscript
│   └── talking_points.md          # Key insights for presentation/defense
│
├── .gitignore
└── README.md                      # Project overview and team details (This file)
```
