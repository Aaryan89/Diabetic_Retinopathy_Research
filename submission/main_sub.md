# Ordinal Neural Networks for Diabetic Retinopathy Staging: Enforcing Monotonic Rank Consistency for Safety-Critical Clinical Decision Support

**Nayna Sharma**  
*Dept. of Computer Science and Engineering*  
*MIT World Peace University*  
Pune, India  
PRN: 1262241524  
nayna.sharma@mitwpu.edu.in  

**Harshad Pardhi**  
*Dept. of Computer Science and Engineering*  
*MIT World Peace University*  
Pune, India  
PRN: 1262241534  
harshad.pardhi@mitwpu.edu.in  

**Aaryan Kumbhare**  
*Dept. of Computer Science and Engineering*  
*MIT World Peace University*  
Pune, India  
PRN: 1262241729  
aaryan.kumbhare@mitwpu.edu.in  

**Anurag Harapanahalli**  
*Dept. of Computer Science and Engineering*  
*MIT World Peace University*  
Pune, India  
PRN: 1262241775  
anurag.harapanahalli@mitwpu.edu.in  

> **Course:** Artificial Intelligence & Expert Systems Laboratory (CSE30070)  
> **Project Mentor:** Prof. Pramod Mali  
> **Academic Year:** 2026–2027  

---

## 1. ABSTRACT

Automated grading of Diabetic Retinopathy (DR) across the 5-point International Clinical Diabetic Retinopathy (ICDR) scale (Grade 0: No DR, Grade 1: Mild Non-Proliferative DR [NPDR], Grade 2: Moderate NPDR, Grade 3: Severe NPDR, Grade 4: Proliferative DR [PDR]) is essential for scalable screening but vulnerable to triage failure when trained with standard nominal cross-entropy. Categorical cross-entropy treats all diagnostic errors symmetrically: confusing vision-threatening PDR (Grade 4) with healthy tissue (Grade 0) incurs the same loss penalty as confusing it with Severe NPDR (Grade 3), despite the drastically different clinical consequences.

In this work, we evaluate rank-consistent ordinal formulations and regularized ensembling on the APTOS 2019 benchmark (3,662 fundus photographs) using a uniform, lightweight EfficientNet-B0 backbone (5.3M parameters). We compare five paradigms under locked stratified data splits: (A) Nominal Softmax Baseline, (B) Consistent Rank Logits (CORAL), (C) Continuous Regression with Smooth L1 loss, (D) Conditional Ordinal Regression for Neural Networks (CORN) with effective-number class weighting and differentiable Soft-QWK loss, and (E) a 3-Seed Regularized Ensemble with label smoothing ($\alpha = 0.08$) and test-time augmentation (TTA).

Our featured model, **Variant E: 3-Seed Ensemble + TTA**, achieves the highest classification accuracy (**83.82%**, 95% CI: [80.73%, 86.73%], vs. 83.45% for the best single seed), with Quadratic Weighted Kappa (QWK: **0.8859**, 95% CI: [0.8503, 0.9170]) and Mean Absolute Error (MAE: **0.2200**, 95% CI: [0.1727, 0.2691]) comparable to the best single seed (Seed 44: QWK 0.8901, MAE 0.2200). Its primary benefit is stability across seeds (single-seed mean 82.73% $\rightarrow$ 83.82%), reducing variance across stochastic training trajectories rather than yielding a large gain over the best seed. The ensemble maintains a catastrophic triage error rate ($|y - \hat{y}| \ge 2$) of **4.36%**. Concurrently, other formulations demonstrate specialized strengths: Variant D (CORN) delivers the highest Severe NPDR sensitivity (**65.7%**, 95% CI: [48.3%, 82.9%]), resolving the minority-class collapse observed in unweighted CORAL (0.0% recall) and improving upon the 36.84% reported by Shakibania et al. [2] (different dataset and task setting; not a head-to-head comparison), while Variant C (Continuous Regression) achieves the lowest overall catastrophic error rate (**3.8%**). Post-hoc logit adjustment on the ensemble offers an adjustable accuracy–sensitivity operating curve without retraining. These results demonstrate that combining lightweight architectures with ordinal-aware formulations provides practical, deployable decision support for clinical screening.

---

## 2. LITERATURE SURVEY

Deep learning for DR screening has evolved from binary classification to multi-stage grading and multimodal explanations. However, most published models optimize nominal cross-entropy, creating metric blindness across disease stages. In this section, we review seven representative studies, identifying their methodologies and clinical limitations.

### 2.1 Analysis of Existing Literature

#### 1. Khalifa et al. (2019) — Deep Transfer Learning Models for Medical DR Detection [1]
Khalifa et al. evaluated six pre-trained CNNs (AlexNet, ResNet-18, SqueezeNet, GoogleNet, VGG-16, VGG-19) on the APTOS 2019 dataset (3,662 fundus images) across the full 5-stage ICDR scale. Although AlexNet reported 97.9% accuracy, a 4-fold offline geometric/color augmentation was executed across the aggregate dataset *prior* to splitting (expanding to 14,648 images), introducing cross-split data leakage. Furthermore, Grade 3 (Severe NPDR) suffered the lowest classification accuracy across all architectures (falling to 67.8% in SqueezeNet).

#### 2. Shakibania et al. (2023) — Dual Branch Network for Detection and Stage Grading of DR [2]
Shakibania et al. designed a two-branch ResNet-50 and EfficientNet-B0 network (~30M parameters) evaluated on standalone APTOS 2019 (88.1% multi-class accuracy, 91.8% QWK) and on a selectively merged multi-center dataset (APTOS 2019, Messidor-2, IDRiD; 89.60% multi-class accuracy, 93.00% QWK). Under unweighted Complement Cross Entropy (CCE), Grade 3 (Severe NPDR) sensitivity collapsed to **36.84%** due to feature overlap and class imbalance. In our experiments, Variant D (CORN) achieves **65.7%** Severe NPDR sensitivity with a single 5.3M-parameter model, representing an absolute improvement of +28.86% with 82% fewer parameters (different dataset and task setting; not a head-to-head comparison).

#### 3. Tymchenko, Marchenko, and Spodarets (2020) — Deep Learning Approach to DR Detection [3]
Tymchenko et al. proposed a multi-target learning paradigm that jointly trained nominal classification (Cross-Entropy + Focal loss), ordinal regression ($K-1$ rank thresholds with binary focal loss), and continuous regression (MAE/MSE) heads on heavy ensembles of EfficientNet-B4/B5 and SE-ResNeXt-50. Pre-trained on EyePACS (35,126 images) and fine-tuned on APTOS 2019, IDRiD, and Messidor, their ensemble ranked 54th out of 2,943 teams on Kaggle with an official test QWK of 0.925. However, joint multi-task training on heavy multi-model ensembles did not isolate the independent contribution of ordinal rank loss versus continuous regression on a single compact architecture.

#### 4. Mardianta, Arifianto, and Fatichah (2025) — CNN with SMOTE and CLAHE Applied to Fundus Images [4]
Mardianta et al. addressed class imbalance on APTOS 2019 using CLAHE and Synthetic Minority Over-sampling Technique (SMOTE) with a pre-trained Xception CNN, reporting 99.55% binary accuracy and 95.26% 5-class accuracy. However, synthesizing retinal feature vectors via linear SMOTE interpolation risks generating non-anatomical artifacts and data leakage, while treating 5-class grading as nominal cross-entropy without evaluating clinical QWK.

#### 5. Karthik, Pandiyaraju, and Mynampati (2025) — Explainable AI for DR Detection [5]
Karthik et al. combined EfficientNetV2-B3 with channel attention and fuzzy membership classification on APTOS 2019 (366 test images), reporting 91.5% accuracy by collapsing the five ICDR stages into three coarse categories (Normal, Mild/Moderate, Severe/Proliferative). Even under this simplified 3-class scheme, their severe/proliferative sensitivity reached only 58% (missing 42% of sight-threatening cases). We preserve the full 5-tier clinical standard while attaining **65.7%** Severe NPDR sensitivity.

#### 6. Khokhar et al. (2026) — Interpretable DR Grading with CNN-Transformer Ensembles and VLMs [6]
Khokhar et al. benchmarked six CNN and Transformer backbones (ResNet-50 reaching 0.919 QWK, Swin reaching 0.916 QWK) under 5-fold cross-validation with weighted soft voting and paired medical Vision-Language Models (LLaVA-Med-v1.5, 7B parameters) for rationale generation on APTOS 2019. Although multimodal explanations are valuable, the large parameter footprint limits point-of-care mobile deployment. Our framework achieves competitive diagnostic accuracy (**0.8859** QWK, **83.82%** accuracy) using a compact 5.3M-parameter backbone.

#### 7. Manoj and Bhosale (2024) — Lesion Segmentation to Facilitate Referral Recommendation [7]
Manoj and Bhosale combined U-Net segmentation of six retinal lesion structures on IDRiD with upstream DR classification and ChatGPT-generated referral recommendations. However, segmentation requires dense pixel annotations, and unverified language model referrals risk patient safety when upstream nominal classification makes multi-grade errors. Our approach directly bounds catastrophic errors ($d \ge 2$) to **3.8%–4.36%** without language model dependency.

### 2.2 Literature Summary and Comparative Mapping

Table 1 summarizes the comparison between existing studies and our work across architectural and clinical criteria.

#### TABLE 1: Literature Survey Summary and Comparative Architecture Mapping

| Paper | Architecture / Method | Dataset | Output Task | Key Metric | Major Limitation | Relation to Our Work |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **Khalifa et al. [1]** | AlexNet, ResNet-18, VGG16/19, CCE Loss | APTOS 2019 (3,662 imgs) | 5-stage Grading | Acc: 97.9% | Pre-split 4x augmentation leakage; Severe recall dropped to 67.8% | Evaluates locked split without leakage (0.8859 QWK, 83.82% Acc) |
| **Shakibania et al. [2]** | Dual Branch (ResNet50 + B0), CCE Loss | APTOS 2019, Messidor-2, IDRiD | 5-stage Grading | Acc: 89.60%, QWK: 0.930, Severe Rec: 36.84% | Severe sensitivity collapsed to 36.84%; ~30M params | Variant D reaches **65.7%** Severe Recall using 5.3M params (different dataset and task setting; not a head-to-head comparison) |
| **Tymchenko et al. [3]** | Multi-target (Nominal + Ordinal + Reg.), Heavy Ens. | EyePACS (35k), APTOS 2019 | 5-stage Grading | Test QWK: 0.925 | Heavy ensemble (B4/B5/ResNeXt); heads not ablated independently | Isolates pure ordinal and single-backbone design on identical 5.3M model |
| **Mardianta et al. [4]** | Xception + SMOTE + CLAHE | APTOS 2019 | 5-stage Grading | Acc: 95.26% | SMOTE creates synthetic artifacts; nominal loss; no QWK reported | Real-image effective-number weighting; evaluates QWK and ordinal distance |
| **Karthik et al. [5]** | EfficientNetV2-B3 + SE Attention + Fuzzy | APTOS 2019 (366 test) | Coarse 3 classes | Acc: 91.5%, Severe Rec: 58% | Collapsed 5 classes to 3; severe sensitivity only 58% | Preserves all 5 clinical stages with 65.7% severe sensitivity |
| **Khokhar et al. [6]** | Swin, ResNet-50 + 7B VLMs | APTOS 2019 (5-fold CV) | 5-stage Grading | QWK: 0.916–0.919 | Heavy compute footprint; high latency | Lightweight 5.3M-parameter model deployable on laptop hardware |
| **Manoj & Bhosale [7]** | UNet (6 lesions) + Classifier + ChatGPT | IDRiD, APTOS 2019 | 5-stage + Referral | Dice: 0.787–0.806 | Requires dense masks; unverified LLM text generation | Bounds catastrophic errors ($d \ge 2$) to 3.8%–4.36% |
| **This Work** | **EfficientNet-B0 + Variant E (Ensemble) / Variant D (CORN)** | **APTOS 2019 (3,662 imgs)** | **5-stage Grading** | **Acc: 83.82%, QWK: 0.8859, Severe Rec: 65.7%** | Single-dataset benchmark evaluation | **Combines 83.82% accuracy, 0.8859 QWK, and 65.7% severe recall** |

---

## 3. PROBLEM STATEMENT

Standard deep learning models for Diabetic Retinopathy treat stage grading as nominal multiclass classification, creating metric blindness where life-threatening misclassifications incur identical loss penalties to minor adjacent errors. This project investigates distance-aware ordinal neural networks and regularized ensembling on retinal fundus imagery to improve rank consistency, suppress catastrophic multi-grade triage errors ($|y - \hat{y}| \ge 2$), and elevate minority-class severe disease sensitivity under practical resource constraints.

---

## 4. OBJECTIVES

1. Formulate and benchmark rank-consistent ordinal loss architectures (CORAL and CORN) against standard nominal softmax and continuous regression on the 5-point ICDR Diabetic Retinopathy scale.
2. Implement effective-number class weighting and differentiable Soft-QWK loss regularization to address class imbalance without generating synthetic image artifacts.
3. Feature a regularized 3-seed ensemble with test-time augmentation to maximize classification accuracy ($\ge 80\%$) and QWK ($\ge 0.88$) while keeping catastrophic multi-grade misclassifications below 5% on a lightweight EfficientNet-B0 backbone.

---

## 5. SYSTEM ARCHITECTURE DIAGRAM

The diagnostic pipeline processes a raw retinal photograph through contrast enhancement, extracts representations via a regularized backbone, evaluates predictions across ordinal and ensembled heads, and maps outputs into clinical triage rules.

![End-to-End System Architecture Diagram: Preprocessing, EfficientNet-B0 Backbone, Ordinal Inference Pathways, and Clinical Decision Support Triage Rules.](submission/results/system_architecture.png)

### 5.1 Architectural Components

As illustrated in Fig. 1, the system comprises six components:

1. **Image Acquisition:** Digital fundus photographs exhibiting natural variations in illumination and pigmentation are acquired from screening clinics.
2. **Preprocessing Module:** 
   - *Circular Crop Boundary Detection:* Automatically segments the functional retinal globe, removing inactive black camera borders.
   - *Contrast Limited Adaptive Histogram Equalization (CLAHE):* Applied in CIE LAB color space on the luminance channel ($8 \times 8$ grid, clip limit 2.0) to enhance microaneurysms and hemorrhages.
   - *Resampling:* Images are resized to $224 \times 224$ pixels and standardized using ImageNet statistics.
3. **Feature Extraction Backbone (EfficientNet-B0):** The preprocessed image is processed through EfficientNet-B0 (5.3M parameters), yielding a 1280-dimensional feature vector $\phi(\mathbf{x})$ via Global Average Pooling. While Variants A, B, C, and E are fine-tuned end-to-end with all layers unfrozen, Variant D (CORN) freezes the stem and first three MBConv stages (`features[:4]`) to retain generic low-level edge filters and prevent feature drift under minority-class reweighting.
4. **Inference Pathways:**
   - *Ordinal Pathway (Variant D: CORN):* Evaluates 4 conditional binary classifiers regularized by Soft-QWK loss and effective-number weighting to maximize severe disease sensitivity.
   - *Featured Ensemble Pathway (Variant E: 3-Seed Ensemble + TTA):* Averages the softmax outputs of three independently trained models with label smoothing ($\alpha = 0.08$) over 4-way flip TTA, with optional post-hoc logit adjustment.
5. **Implementation & Compute:** All models were trained on a single consumer laptop GPU using under 1 GB of VRAM with mixed-precision arithmetic.
6. **Clinical Decision Support Triage Rules:** Discrete stage predictions map to standardized clinical pathways:
   - **Grade 4 (PDR):** Emergency Vitreoretinal Referral (< 48 Hours)
   - **Grade 3 (Severe NPDR):** Urgent Laser / Anti-VEGF Evaluation (< 2 Weeks)
   - **Grade 2 (Moderate NPDR):** Comprehensive Examination & OCT Scan (< 6 Months)
   - **Grade 1 (Mild NPDR):** Surveillance Follow-up (< 12 Months)
   - **Grade 0 (No DR):** Routine Annual Tele-Screening Recall

---

## 6. IMPLEMENTATION

### 6.1 Dataset Characteristics and Partitioning

We evaluate our pipeline on the APTOS 2019 Blindness Detection dataset (3,662 fundus photographs) under the 5-point ICDR scale. The dataset exhibits substantial natural class imbalance (Fig. 2):
- **Grade 0 (No DR):** 1,805 images (49.29%)
- **Grade 1 (Mild NPDR):** 370 images (10.10%)
- **Grade 2 (Moderate NPDR):** 999 images (27.28%)
- **Grade 3 (Severe NPDR):** 193 images (5.27%)
- **Grade 4 (Proliferative DR):** 295 images (8.06%)

![Dataset Class Distribution across the 3,662 Retinal Fundus Images in APTOS 2019 under the 5-point ICDR Clinical Severity Scale.](submission/results/class_distribution.png)

The images were partitioned using stratified random sampling with a fixed seed (42) into three locked sets:
- **Training Set (70%):** 2,563 images (Grade 0: 1,263; Grade 1: 259; Grade 2: 700; Grade 3: 135; Grade 4: 206)
- **Validation Set (15%):** 549 images (Grade 0: 271; Grade 1: 55; Grade 2: 150; Grade 3: 29; Grade 4: 44)
- **Holdout Test Set (15%):** 550 images (Grade 0: 271; Grade 1: 56; Grade 2: 149; Grade 3: 29; Grade 4: 45)

All images were preprocessed and cached in `./data/aptos2019/preprocessed_224/` for reproducible training I/O. Fig. 3 shows sample preprocessed images across each grade.

![Representative Preprocessed Retinal Fundus Photographs across the Five Clinical ICDR Grades following Circular Cropping and CIE LAB CLAHE Contrast Equalization.](submission/results/sample_images_per_class.png)

### 6.2 Mathematical Formulation of Evaluated Paradigms

#### 1) Variant A: Nominal Softmax Baseline and Metric Blindness
In nominal classification, the embedding $\phi(\mathbf{x}) \in \mathbb{R}^{1280}$ is projected via $\mathbf{W} \in \mathbb{R}^{5 \times 1280}$ to produce logits $\mathbf{z} = [z_0, z_1, z_2, z_3, z_4]^T$. Probabilities are computed via the softmax function:

$$\hat{p}_k = \frac{\exp(z_k)}{\sum_{j=0}^4 \exp(z_j)}, \quad k \in \{0, 1, 2, 3, 4\}$$

The loss is categorical cross-entropy:

$$\mathcal{L}_{CE} = -\sum_{k=0}^4 y_k \log \hat{p}_k$$

**Metric Blindness Property:** Consider true class $y = 4$ ($\mathbf{y} = [0, 0, 0, 0, 1]^T$). For any model outputting probability $\hat{p}_4 = 0.05$ on the true class, the loss is:

$$\mathcal{L}_{CE} = -\log(0.05) \approx 2.9957$$

Whether the remaining 0.95 probability is assigned to Grade 3 ($|4 - 3| = 1$, safe error) or Grade 0 ($|4 - 0| = 4$, dangerous discharge), the cross-entropy loss and gradient $\frac{\partial \mathcal{L}_{CE}}{\partial z_k} = \hat{p}_k - y_k$ treat non-target errors symmetrically.

#### 2) Variant B: Consistent Rank Logits (CORAL)
CORAL [8] decomposes the 5-class problem into $K - 1 = 4$ binary tasks:

$$r_k = \begin{cases} 1 & \text{if } y > k \\ 0 & \text{otherwise} \end{cases}, \quad k \in \{0, 1, 2, 3\}$$

A shared weight vector $\mathbf{w} \in \mathbb{R}^{1280}$ and independent biases $b_k$ form the cumulative probabilities:

$$g_k(\mathbf{x}) = \mathbf{w}^T \phi(\mathbf{x}) + b_k, \quad \hat{P}(y > k \mid \mathbf{x}) = \sigma(g_k(\mathbf{x}))$$

$$\mathcal{L}_{CORAL} = -\sum_{k=0}^3 \left[ r_k \log \sigma(g_k(\mathbf{x})) + (1 - r_k) \log (1 - \sigma(g_k(\mathbf{x}))) \right]$$

In unweighted CORAL, every sample updates all four binary tasks. In our imbalanced dataset (Grade 0: 1,263 vs. Grade 3: 135), negative gradients from healthy retinas dominate the shared projection $\mathbf{w}$, leading to empirical collapse on Grade 3 (0.0% recall).

#### 3) Variant C: Continuous Regression with Smooth L1 Loss
Continuous regression maps $\phi(\mathbf{x})$ to a scalar $\hat{s} \in \mathbb{R}$:

$$\hat{s} = \mathbf{w}^T \phi(\mathbf{x}) + b$$

The network is trained with Smooth L1 loss:

$$\mathcal{L}_{SmoothL1}(y, \hat{s}) = \begin{cases} 0.5 (y - \hat{s})^2 & \text{if } |y - \hat{s}| < 1.0 \\ |y - \hat{s}| - 0.5 & \text{otherwise} \end{cases}$$

Discrete predictions are obtained by clipping and rounding: $\hat{y} = \text{round}(\text{clip}(\hat{s}, 0.0, 4.0))$. Distance penalties reduce large multi-grade errors.

#### 4) Variant D (CORN): Conditional Ordinal Regression with Class Weighting
CORN [13] models the conditional probability of exceeding rank $k$, given that rank $k-1$ was reached:

$$q_k(\mathbf{x}) = P(y > k \mid y > k-1) = \sigma(g_k(\mathbf{x})), \quad k \in \{0, 1, 2, 3\}$$

Unconditional cumulative probabilities follow via the product rule:

$$P(y > k \mid \mathbf{x}) = \prod_{j=0}^k q_j(\mathbf{x})$$

A sample with label $y$ participates only in binary tasks up to its label:

$$\mathcal{L}_{CORN}(\mathbf{x}, y) = \frac{1}{\min(y+1, K-1)} \sum_{k=0}^{\min(y, K-2)} \text{BCE}(g_k(\mathbf{x}), \mathbb{I}(y > k))$$

To counteract remaining imbalance, we incorporate effective-number class weighting (Cui et al. [14], $\beta = 0.9999$): $W_0 = 0.2269, W_1 = 1.0529, W_2 = 0.3987, W_3 = 2.0075, W_4 = 1.3140$, alongside differentiable Soft-QWK loss regularization:

$$\mathcal{L}_{total} = \mathcal{L}_{CORN} + \lambda_{QWK} \cdot \mathcal{L}_{SoftQWK}, \quad \lambda_{QWK} = 0.20$$

#### 5) Variant E: 3-Seed Ensemble + TTA with Label Smoothing
For applications prioritizing overall diagnostic accuracy, we implement a multi-seed ensembled architecture:
- **Label Smoothing:** Targets are smoothed with $\alpha = 0.08$:
  $$y_k^{LS} = (1 - \alpha) y_k + \frac{\alpha}{K}$$
  reducing overconfidence on ambiguous retinal lesions.
- **Multi-Seed Ensembling and TTA:** Three models are trained independently across random seeds $\{42, 43, 44\}$ for 12 epochs each. At inference time, 4-way flip TTA (original, horizontal, vertical, both) is applied, and softmax probabilities are averaged across the three models.
- **Post-Hoc Logit Adjustment (Sensitivity Analysis):** Unnormalized logits can be adjusted by class priors $\pi_y$:
  $$\tilde{f}_y(\mathbf{x}) = f_y(\mathbf{x}) - \tau \cdot \log(\pi_y)$$
  where $\tau \ge 0$. In this study, temperature sweeps across $\tau \in \{0.0, 0.5, 1.0, 1.5\}$ are evaluated on the test set strictly as a post-hoc sensitivity analysis to document the accuracy–sensitivity trade-off.

### 6.3 Training and Hyperparameter Configurations

To ensure experimental transparency and reproducibility, Table 2 details the hyperparameter configurations across all five model variants. All models shared an identical pretrained EfficientNet-B0 backbone (5.3M parameters), AdamW optimizer ($lr = 3 \times 10^{-4}$, weight decay $1 \times 10^{-2}$), batch size of 16, and Cosine Annealing learning rate schedule. However, training protocols differed in targeted ways: Variants A–C were trained for 8 epochs under standard augmentations with all layers unfrozen on a single run (seed 42). Variant D (CORN) introduced early-block freezing (stages 0–3 frozen to stabilize low-level edge features), class-balanced effective-number weighting, and a WeightedRandomSampler over 8 epochs. Variant E employed an extended 12-epoch training budget per seed, label smoothing ($\alpha = 0.08$), safe-light augmentations, and sequential 3-seed ensembling with test-time augmentation.

#### TABLE 2: Training and Hyperparameter Configurations Across Model Variants

| Hyperparameter / Setting | Variant A (Softmax) | Variant B (CORAL) | Variant C (Regression) | Variant D (CORN) | Variant E (Ensemble) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Target Epochs** | 8 | 8 | 8 | 8 | 12 (per seed) |
| **Optimizer** | AdamW | AdamW | AdamW | AdamW | AdamW |
| **Learning Rate** | $3 \times 10^{-4}$ | $3 \times 10^{-4}$ | $3 \times 10^{-4}$ | $3 \times 10^{-4}$ | $3 \times 10^{-4}$ |
| **Weight Decay** | $1 \times 10^{-2}$ | $1 \times 10^{-2}$ | $1 \times 10^{-2}$ | $1 \times 10^{-2}$ | $1 \times 10^{-2}$ |
| **Batch Size** | 16 | 16 | 16 | 16 | 16 |
| **LR Scheduler** | Cosine Annealing | Cosine Annealing | Cosine Annealing | Cosine Annealing | Cosine Annealing |
| **Frozen Layers** | None (All unfrozen) | None (All unfrozen) | None (All unfrozen) | Stages 0–3 frozen | None (All unfrozen) |
| **Augmentation** | Flips, Rot $\pm 25^\circ$, Jitter | Flips, Rot $\pm 25^\circ$, Jitter | Flips, Rot $\pm 25^\circ$, Jitter | Flips, Rot $\pm 25^\circ$, Jitter | Flips, Affine $\pm 15^\circ$, Light Jitter |
| **Label Smoothing** | None | N/A (Binary BCE) | N/A (Smooth L1) | None | $\alpha = 0.08$ |
| **Imbalance Strategy** | Natural distribution | Natural distribution | Natural distribution | Cui et al. Weights + Sampler | Natural distribution |
| **Number of Seeds** | 1 (Seed 42) | 1 (Seed 42) | 1 (Seed 42) | 1 (Seed 42) | 3 (Seeds 42, 43, 44) |

---

## 7. RESULTS

All models were evaluated on the locked holdout test set of 550 images, which was not used during training or checkpoint selection.

### 7.1 Comprehensive Comparison Across Model Variants

Table 3 presents the complete performance comparison across all five paradigms. Table 4 breaks down sensitivity, precision, and F1-scores with 95% bootstrap confidence intervals (1,000 resamples).

#### TABLE 3: Comprehensive Performance Comparison Across Model Variants on APTOS 2019 Holdout Test Set (N = 550)

| Model Variant | QWK (Primary) [95% CI] | Accuracy (%) | MAE [95% CI] | Catastrophic Error ($d \ge 2$) [95% CI] | Severe F1 (Grade 3) | PDR F1 (Grade 4) | Exact Match ($d=0$) | Off-by-1 ($d=1$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Variant A: Softmax Baseline** | 0.8724 [0.8385, 0.9044] | 80.18% | 0.2673 [0.2182, 0.3200] | 5.64% [3.64%, 7.64%] | 0.3810 [0.2221, 0.5313] | 0.5169 [0.3830, 0.6377] | 80.2% | 14.2% |
| **Variant B: CORAL Ordinal** | 0.7273 [0.6874, 0.7641] | 59.64% | 0.6964 [0.6200, 0.7710] | 26.40% [22.90%, 30.20%] | 0.0000 [0.0000, 0.0000] | 0.3692 [0.2857, 0.4502] | 59.6% | 14.0% |
| **Variant C: Continuous Regression** | 0.8788 [0.8490, 0.9039] | 75.64% | 0.2873 [0.2400, 0.3364] | **3.80%** [2.40%, 5.50%] | 0.2528 [0.1333, 0.3704] | 0.3589 [0.2034, 0.5067] | 75.6% | 20.5% |
| **Variant D (CORN): Conditional Ordinal** | 0.8482 [0.8150, 0.8792] | 69.09% | 0.3764 [0.3200, 0.4291] | 5.10% [3.30%, 7.10%] | 0.2724 [0.1746, 0.3735] | 0.4584 [0.3124, 0.5909] | 69.1% | 25.8% |
| **Variant E: 3-Seed Ensemble + TTA** | **0.8859** [0.8503, 0.9170] | **83.82%** | **0.2200** [0.1727, 0.2691] | 4.36% [2.73%, 6.18%] | **0.4561** [0.2999, 0.6134] | **0.5570** [0.4179, 0.6739] | **83.8%** | **11.8%** |

*Note: Best value in each column is shown in bold. Quadratic Weighted Kappa (QWK) is the primary clinical ordinal metric. Variant E catastrophic error is 4.36%.*

#### TABLE 4: Per-Class Diagnostic Performance with 95% Bootstrap Confidence Intervals (1,000 Resamples)

| Class Grade | Model Variant | Recall / Sensitivity [95% CI] | Precision [95% CI] | F1-Score [95% CI] |
|:---|:---|:---:|:---:|:---:|
| **0 (No DR)** | Variant A (Softmax Baseline) | 97.1% [94.9%, 98.9%] | 96.7% [94.5%, 98.6%] | 0.9690 [0.9541, 0.9825] |
| | Variant B (CORAL Ordinal) | **100.0%** [100.0%, 100.0%] | 74.9% [70.7%, 79.3%] | 0.8560 [0.8281, 0.8843] |
| | Variant C (Continuous Regression) | 96.7% [94.4%, 98.6%] | 96.0% [93.5%, 98.1%] | 0.9635 [0.9478, 0.9779] |
| | Variant D (CORN) | 97.4% [95.4%, 99.2%] | 95.7% [93.3%, 97.9%] | 0.9655 [0.9502, 0.9799] |
| | **Variant E (3-Seed Ensemble + TTA)** | 98.2% [96.4%, 99.6%] | 96.0% [93.7%, 98.0%] | **0.9708** [0.9569, 0.9837] |
| **1 (Mild NPDR)** | Variant A (Softmax Baseline) | 55.5% [42.4%, 68.4%] | 59.7% [46.8%, 73.3%] | 0.5728 [0.4602, 0.6777] |
| | Variant B (CORAL Ordinal) | 1.8% [0.0%, 6.1%] | 42.3% [0.0%, 100.0%] | 0.0344 [0.0000, 0.1132] |
| | Variant C (Continuous Regression) | 50.2% [37.7%, 63.6%] | 50.9% [37.5%, 63.5%] | 0.5035 [0.3860, 0.6087] |
| | Variant D (CORN) | **66.1%** [52.6%, 79.2%] | 47.6% [36.5%, 59.3%] | 0.5515 [0.4409, 0.6491] |
| | **Variant E (3-Seed Ensemble + TTA)** | 46.4% [33.3%, 59.6%] | **72.2%** [58.6%, 86.5%] | 0.5652 [0.4367, 0.6800] |
| **2 (Moderate NPDR)** | Variant A (Softmax Baseline) | 74.6% [67.3%, 81.9%] | 76.3% [69.0%, 83.1%] | 0.7538 [0.6978, 0.8053] |
| | Variant B (CORAL Ordinal) | 11.9% [7.0%, 17.5%] | 71.9% [52.9%, 88.9%] | 0.2039 [0.1241, 0.2857] |
| | Variant C (Continuous Regression) | 68.5% [60.6%, 75.8%] | 72.6% [64.4%, 80.0%] | 0.7044 [0.6370, 0.7638] |
| | Variant D (CORN) | 28.0% [21.1%, 35.3%] | **81.0%** [69.8%, 91.3%] | 0.4151 [0.3284, 0.5000] |
| | **Variant E (3-Seed Ensemble + TTA)** | **89.3%** [84.1%, 94.0%] | 77.0% [70.6%, 83.3%] | **0.8272** [0.7782, 0.8682] |
| **3 (Severe NPDR)** | Variant A (Softmax Baseline) | 41.7% [23.3%, 60.0%] | 35.4% [20.0%, 51.6%] | 0.3810 [0.2221, 0.5313] |
| | Variant B (CORAL Ordinal) | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0000 [0.0000, 0.0000] |
| | Variant C (Continuous Regression) | 38.4% [20.0%, 56.8%] | 19.1% [9.2%, 29.0%] | 0.2528 [0.1333, 0.3704] |
| | Variant D (CORN) | **65.7%** [48.3%, 82.9%] | 17.3% [10.6%, 24.8%] | 0.2724 [0.1746, 0.3735] |
| | **Variant E (3-Seed Ensemble + TTA)** | 44.8% [27.8%, 64.0%] | **46.4%** [29.6%, 65.4%] | **0.4561** [0.2999, 0.6134] |
| **4 (Proliferative DR)** | Variant A (Softmax Baseline) | 52.3% [37.2%, 67.3%] | 51.2% [36.6%, 65.1%] | 0.5169 [0.3830, 0.6377] |
| | Variant B (CORAL Ordinal) | **86.3%** [75.6%, 95.1%] | 23.6% [17.2%, 30.2%] | 0.3692 [0.2857, 0.4502] |
| | Variant C (Continuous Regression) | 27.1% [14.3%, 41.7%] | 54.6% [33.3%, 75.0%] | 0.3589 [0.2034, 0.5067] |
| | Variant D (CORN) | 41.0% [26.2%, 55.1%] | 52.8% [36.8%, 69.4%] | 0.4584 [0.3124, 0.5909] |
| | **Variant E (3-Seed Ensemble + TTA)** | 50.0% [35.7%, 64.9%] | **62.9%** [47.2%, 77.8%] | **0.5570** [0.4179, 0.6739] |

### 7.2 Confusion Matrix and Error Distribution Analysis

Fig. 4 shows the side-by-side confusion matrices across all five models, and Fig. 5 displays prediction error distance distributions ($|y - \hat{y}|$).

![Side-by-Side Confusion Matrices on the Holdout Test Set (N = 550) for Variant A (Softmax Baseline), Variant B (CORAL Ordinal), Variant C (Continuous Regression), Variant D (CORN), and Variant E (3-Seed Ensemble + TTA).](submission/results/all_confusion_matrices.png)

![Prediction Error Distance Distribution across the 550 Holdout Test Images comparing All Five Model Variants with corresponding MAE metrics.](submission/results/error_distance_histogram.png)

Key observations:
- **Variant E (Fifth Panel):** Exhibits high diagonal concentration, achieving an exact-match rate of **83.8%** and holding off-by-one errors to **11.8%**, matching the lowest single-seed MAE (**0.2200**) with the highest overall accuracy (**83.82%**).
- **Variant D (Fourth Panel):** Achieves the highest recall on Grade 3 Severe NPDR (**65.7%**), preventing the diagnostic omissions seen in Variant B.
- **Variant C (Third Panel):** Bounds catastrophic multi-grade triage errors ($d \ge 2$) to **3.8%**, yielding the lowest catastrophic error rate.
- **Variant B (Second Panel):** Demonstrates severe class collapse, with 26.40% catastrophic errors and 0.0% Grade 3 recall.

### 7.3 Statistical Significance Testing

Paired bootstrap tests (1,000 iterations) and Wilcoxon signed-rank tests confirmed:
- **Variant E vs. Variant A (Softmax Baseline):** $\Delta QWK = +0.0135$ (95% CI: $[-0.0113, +0.0383]$, $p = 0.1230$) and $\Delta \text{Accuracy} = +3.64\%$ (95% CI: $[+1.09\%, +6.18\%]$, $p = 0.0050$). A Wilcoxon signed-rank test on absolute error distances confirmed a statistically significant reduction in prediction error ($W = 625.0, p = 0.0152$).
- **Variant D (CORN) vs. Variant B (CORAL):** $\Delta QWK = +0.1212$ (95% CI: $[+0.0924, +0.1504]$, $p < 0.001$), with Wilcoxon $W = 4180.0, p = 9.35 \times 10^{-21}$, confirming substantial minority-class improvement.
- **Variant C vs. Variant A:** $\Delta QWK = +0.0062$ (95% CI: $[-0.0142, +0.0265]$, $p = 0.2640$).

### 7.4 Multi-Seed Ensemble and Sensitivity Analysis

Table 5 details individual seed performances, single-seed summary statistics, and ensemble configurations. Table 6 documents the temperature sweep for post-hoc logit adjustment, and Fig. 6 displays the confusion matrices across accuracy optimization configurations.

#### TABLE 5: Classification Performance Across Ensemble Configurations on Holdout Test Set (N = 550)

| Model Configuration | Accuracy [95% CI] | QWK [95% CI] | MAE [95% CI] | Catastrophic Error ($d \ge 2$) | Grade 3 Severe F1 | Grade 4 PDR F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant A (Baseline Softmax)** | 80.18% [76.73%, 83.45%] | 0.8724 [0.8385, 0.9044] | 0.2673 [0.2182, 0.3200] | 5.64% [3.64%, 7.64%] | 0.3810 [0.2221, 0.5313] | 0.5169 [0.3830, 0.6377] |
| **Single Seed 42 (LS 0.08)** | 82.00% [78.73%, 85.10%] | 0.8709 [0.8330, 0.9039] | 0.2473 [0.1982, 0.3000] | 5.27% [3.45%, 7.27%] | 0.4828 [0.3225, 0.6452] | 0.5195 [0.3662, 0.6364] |
| **Single Seed 43 (LS 0.08)** | 82.73% [79.45%, 85.82%] | 0.8846 [0.8501, 0.9165] | 0.2291 [0.1818, 0.2800] | 4.18% [2.73%, 5.82%] | 0.4068 [0.2456, 0.5667] | 0.5316 [0.3929, 0.6512] |
| **Single Seed 44 (LS 0.08)** | 83.45% [80.18%, 86.55%] | 0.8901 [0.8562, 0.9198] | 0.2200 [0.1764, 0.2691] | 4.18% [2.55%, 5.82%] | 0.5085 [0.3333, 0.6667] | 0.6000 [0.4615, 0.7161] |
| **Single-Seed Mean $\pm$ SD** | 82.73% $\pm$ 0.73% | 0.8819 $\pm$ 0.0099 | 0.2321 $\pm$ 0.0139 | 4.54% $\pm$ 0.63% | 0.4660 $\pm$ 0.0529 | 0.5504 $\pm$ 0.0434 |
| **Variant E: 3-Seed Ens + TTA ($\tau = 0.0$)** | **83.82%** [80.73%, 86.73%] | 0.8859 [0.8503, 0.9170] | **0.2200** [0.1727, 0.2691] | 4.36% [2.73%, 6.18%] | 0.4561 [0.2999, 0.6134] | 0.5570 [0.4179, 0.6739] |
| **Variant E: 3-Seed Ens + TTA ($\tau = 1.0$)** | 79.27% [76.00%, 82.55%] | 0.8787 [0.8451, 0.9091] | 0.2691 [0.2200, 0.3164] | 4.55% [2.73%, 6.36%] | 0.3721 [0.2432, 0.5056] | **0.5778** [0.4444, 0.6875] |

*Note: Single-seed summary reports mean $\pm$ sample SD ($n=3$, $ddof=1$). Seed 44 alone achieves QWK 0.8901, MAE 0.2200, and 4.18% catastrophic error. Best value in each column is shown in bold.*

#### TABLE 6: Post-Hoc Logit Adjustment Temperature Sweep on 3-Seed Ensemble (Sensitivity Analysis)

| Temperature ($\tau$) | Accuracy | QWK | MAE | Severe Recall (Grade 3) | PDR Recall (Grade 4) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0 (Unadjusted, Raw Acc Focus)** | **83.82%** | 0.8859 | **0.2200** | 44.8% | 50.0% |
| **0.5 (Balanced Trade-off)** | 82.18% | **0.8929** | 0.2291 | 51.7% | 56.8% |
| **1.0 (Standard Prior Adjustment)** | 79.27% | 0.8787 | 0.2691 | 55.2% | **59.1%** |
| **1.5 (Minority-Tail Focus)** | 74.91% | 0.8438 | 0.3364 | **69.0%** | 50.0% |

![Confusion Matrices for Accuracy Optimization Configurations on the Holdout Test Set (N = 550) comparing Baseline Softmax, Label-Smoothed Seed 42, and 3-Seed Ensembles (tau = 0.0 and tau = 1.0).](submission/results/confusion_matrix_accuracy_ensemble.png)

### 7.5 Granular Per-Grade Diagnostic Dynamics and Clinical Referral Triage

#### Per-Grade Diagnostic Performance Dynamics
A granular inspection of per-class metrics (Table 4) reveals that Variant E's accuracy advantage (+3.64% over Variant A) stems predominantly from performance improvements on the two most prevalent classes: Moderate NPDR (Grade 2 recall rising from 74.6% to 89.3%) and No DR (Grade 0 recall improving from 97.1% to 98.2%). In contrast, high-acuity minority grades show negligible gain: Grade 3 Severe NPDR recall increases only marginally from 41.7% to 44.8%, while Grade 4 Proliferative DR recall remains essentially unchanged (50.0% vs. 52.3% in baseline Variant A). Furthermore, Grade 1 Mild NPDR recall drops noticeably from 55.5% to 46.4%, with mild lesions frequently subsumed into the dominant Grade 0 or Grade 2 predictions. Consequently, Variant E functions primarily as a stabilizer for majority disease categories rather than an effective detector for rare sight-threatening lesions.

#### Clinical Referral Triage Performance
In practical tele-ophthalmology screening, patients with severe or proliferative retinopathy ($y \in \{3, 4\}$) require urgent specialist referral, whereas healthy individuals and early non-proliferative cases ($y \in \{0, 1, 2\}$) are managed via routine community monitoring. We define *under-triage rate* as the proportion of true vision-threatening patients incorrectly classified as non-referral ($\hat{y} \le 2$ when $y \ge 3$), and *over-referral rate* as the proportion of non-referral patients incorrectly routed to tertiary care ($\hat{y} \ge 3$ when $y \le 2$). Table 7 presents the clinical triage performance across all evaluated models on the 550 holdout test images ($N_{\text{severe}} = 73$, $N_{\text{routine}} = 477$).

#### TABLE 7: Clinical Referral Triage Performance Across Model Variants on Holdout Test Set (N = 550)

| Model Variant | Under-Triage Count ($y \ge 3 \to \hat{y} \le 2$) | Under-Triage Rate (%) | Over-Referral Count ($y \le 2 \to \hat{y} \ge 3$) | Over-Referral Rate (%) | Clinical Trade-off Profile |
|:---|:---:|:---:|:---:|:---:|:---|
| **Variant A: Softmax Baseline** | 21 / 73 | 28.77% | 27 / 477 | 5.66% | Unweighted multiclass baseline |
| **Variant B: CORAL Ordinal** | 8 / 73 | 10.96% | 96 / 477 | 20.13% | High false referrals from class collapse |
| **Variant C: Continuous Regression** | 23 / 73 | 31.51% | 30 / 477 | 6.29% | Conservative; bounds multi-grade drift |
| **Variant D (CORN): Conditional Ordinal** | 11 / 73 | **15.07%** | 82 / 477 | 17.19% | High-sensitivity screening specialist |
| **Variant E: 3-Seed Ens + TTA ($\tau = 0.0$)** | 22 / 73 | 30.14% | 12 / 477 | **2.52%** | Minimum false referrals; confirmatory focus |
| **Variant E: 3-Seed Ens + TTA ($\tau = 1.0$)** | 14 / 73 | 19.18% | 44 / 477 | 9.22% | Balanced trade-off via logit adjustment |

*Note: Sight-threatening cases (Grades 3–4): N = 73; non-referral cases (Grades 0–2): N = 477. Best values in bold.*

### 7.6 Strengths and Limitations of the 3-Seed Ensemble

#### 1) Architectural and Operational Strengths
1. **High Overall Accuracy and Multi-Seed Stability:** Variant E achieves the highest classification accuracy (**83.82%**, vs. 83.45% for the best single seed), with QWK (**0.8859**) and MAE (**0.2200**) comparable to the best single seed (Seed 44: QWK 0.8901, MAE 0.2200). Its primary benefit is mitigating stochastic training variance across seeds (single-seed mean 82.73% $\rightarrow$ 83.82%) rather than delivering large gains over the top seed.
2. **Lowest Over-Referral Rate:** Variant E yields the lowest over-referral rate (**2.52%**, 12/477), substantially reducing unnecessary secondary clinic visits and clinic overburden.
3. **Resource Efficiency and Tunability:** The ensemble reuses the identical 5.3M-parameter EfficientNet-B0 backbone with no external data. Post-hoc logit adjustment provides an adjustable operating curve between accuracy and severe recall without retraining.

#### 2) Clinical and Computational Limitations
1. **Elevated Under-Triage Rate:** At $\tau = 0.0$, Severe NPDR recall is only 44.8% and under-triage is **30.14%** (22/73 missed severe/PDR cases), compared to 15.07% (11/73) for Variant D.
2. **Modest Gain Over Best Seed:** The accuracy gain over Seed 44 (83.45% to 83.82%) is small, and individual seeds exhibit competitive QWK (Seed 44: 0.8901) and catastrophic error rates (Seeds 43 and 44: 4.18% vs. 4.36% for the ensemble).
3. **Inference Overhead:** Evaluating three ensemble members with 4-way TTA requires approximately $3\times$ FLOPs compared to a single forward pass.
4. **Dataset Scope:** All evaluations are conducted on a single 550-image test split with 29 Severe NPDR cases.

#### 3) Future Research Directions
Future iterations will evaluate: (a) ensembling heterogeneous backbones (e.g., ConvNeXt with EfficientNet); (b) integrating CORN ordinal heads directly into the ensemble members; (c) higher input resolutions (e.g., 384x384); (d) cross-validation and calibration analysis (ECE); (e) external validation on Messidor-2 and IDRiD; and (f) knowledge distillation into a single compact inference student model.

---

## 8. DISCUSSION AND CLINICAL TRIAGE

Our empirical results demonstrate that no single model configuration universally optimizes all clinical criteria:
1. **Confirmatory Screening and Triage Efficiency:** Variant E (3-Seed Ensemble + TTA) is optimal when overall classification accuracy (**83.82%**) and high specificity across common grades are required. As quantified in Table 7, Variant E achieves the lowest over-referral rate (**2.52%**, 12/477), minimizing unnecessary hospital visits and reducing clinical burden on overstretched eye clinics. However, its unadjusted under-triage rate (**30.14%**) necessitates post-hoc logit adjustment ($\tau = 1.0$, reducing under-triage to 19.18%) or dual-route screening when deployed for primary screening.
2. **Early Disease Detection and Sight Preservation:** Variant D (CORN) is preferable when maximizing sensitivity for vision-threatening Severe NPDR (**65.7%**) and minimizing missed sight-threatening diagnoses is paramount, reducing under-triage to **15.07%** (11/73).
3. **Safety Envelope:** Variant C (Continuous Regression) provides the lowest catastrophic multi-grade error rate (**3.8%**).

Deploying automated AI screening in real-world clinical practice requires selecting architectures and loss formulations aligned with specific institutional triage priorities and risk thresholds.

---

## 9. CONCLUSION

In this study, we addressed the limitation of nominal cross-entropy in Diabetic Retinopathy staging. Evaluating nominal, regression, ordinal, and ensembled paradigms on the APTOS 2019 benchmark using an EfficientNet-B0 backbone yielded three principal conclusions:

1. **Featured High-Accuracy Ensemble:** **Variant E: 3-Seed Ensemble + TTA** achieved the project's highest classification accuracy (**83.82%**, 95% CI: [80.73%, 86.73%], vs. 83.45% for the best single seed), with QWK (**0.8859**) and MAE (**0.2200**) comparable to the best single seed, reducing stochastic initialization variance across seeds (single-seed mean 82.73% $\rightarrow$ 83.82%) while maintaining a catastrophic triage error rate of **4.36%**. Post-hoc logit adjustment reached a peak QWK of **0.8929** at $\tau = 0.5$ in sensitivity testing.
2. **Minority-Class Protection:** Variant D (CORN) resolved the minority collapse of unweighted CORAL, attaining **65.7%** Severe NPDR sensitivity (vs. 36.84% reported by Shakibania et al. [2], different dataset and task setting; not a head-to-head comparison) with an 82% smaller parameter footprint.
3. **Distance Regularization:** Continuous regression (Variant C) achieved the tightest error distribution, holding catastrophic multi-grade errors to **3.8%**.

All models operate on lightweight 5.3M-parameter backbones on consumer laptop hardware under 1 GB of VRAM. Future work will investigate cross-cohort validation, heterogeneous ensembling, and knowledge distillation.

---

## 10. REFERENCES

[1] N. E. M. Khalifa, M. H. N. Taha, A. E. Hassanien, and I. M. Selim, "Deep transfer learning models for medical diabetic retinopathy detection," Acta Inform. Med., vol. 27, no. 5, pp. 327–331, Dec. 2019, doi: 10.5455/aim.2019.27.327-331.

[2] N. Shakibania, S. Khosravi, H. Danyali, and M. S. Helfroush, "Dual branch deep learning network for detection and stage grading of diabetic retinopathy," arXiv preprint arXiv:2308.09945, Aug. 2023.

[3] B. Tymchenko, P. Marchenko, and D. Spodarets, "Deep learning approach to diabetic retinopathy detection," arXiv preprint arXiv:2003.02261, Mar. 2020.

[4] M. A. Mardianta, A. Arifianto, and C. Fatichah, "Diabetic retinopathy detection based on convolutional neural networks with SMOTE and CLAHE techniques applied to fundus images," arXiv preprint arXiv:2504.05696, Apr. 2025.

[5] R. Karthik, R. Pandiyaraju, and T. Mynampati, "Explainable AI for diabetic retinopathy detection using deep learning with attention mechanisms and fuzzy logic-based interpretability," arXiv preprint arXiv:2511.16294, Nov. 2025.

[6] S. Khokhar, M. A. A. Baig, M. U. G. Khan, and H. A. Al-Nawaiseh, "From pixels to explanations: Interpretable diabetic retinopathy grading with CNN-Transformer ensembles, visual explainability and vision-language models," arXiv preprint arXiv:2604.23079, Apr. 2026.

[7] V. Manoj and S. Bhosale, "Detection and classification of diabetic retinopathy using deep learning algorithms for segmentation to facilitate referral recommendation," arXiv preprint arXiv:2401.02759, Jan. 2024.

[8] W. Cao, V. Mirjalili, and S. Raschka, "Rank consistent ordinal regression for neural networks with application to age estimation," Pattern Recognition Letters, vol. 140, pp. 325–331, Dec. 2020.

[9] J. Cohen, "A coefficient of agreement for nominal scales," Educational and Psychological Measurement, vol. 20, no. 1, pp. 37–46, 1960.

[10] C. P. Wilkinson, F. L. Ferris, R. E. Klein, P. P. Lee, C. D. Agardh, M. Davis, D. Dills, A. Kampik, R. Pararajasegaram, and J. T. Verdaguer, "Proposed international clinical diabetic retinopathy and diabetic macular edema disease severity scales," Ophthalmology, vol. 110, no. 9, pp. 1677–1682, Sep. 2003.

[11] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 770–778.

[12] APTOS 2019 Blindness Detection, Kaggle Competition Dataset, Asia Pacific Tele-Ophthalmology Society, 2019. [Online]. Available: https://www.kaggle.com/c/aptos2019-blindness-detection

[13] X. Shi, W. Cao, and S. Raschka, "Deep neural networks for rank-consistent ordinal regression based on conditional probabilities," Pattern Recognition Letters, vol. 152, pp. 110–116, 2021.

[14] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, "Class-balanced loss based on effective number of samples," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 9268–9277.

[15] A. K. Menon, S. Jayasumana, A. S. Rawat, H. Jain, A. Veit, and S. Kumar, "Long-tail learning via logit adjustment," in Proc. Int. Conf. Learn. Represent. (ICLR), 2021.
