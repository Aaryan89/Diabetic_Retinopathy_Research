# Literature Review: Papers 3 and 4

**Reviewer Assignment:** Member 2  
**Scope:** Multi-Target Ordinal Formulations, CLAHE Image Enhancement, and Imbalance Handling  
**Target Citation Style:** IEEE (Numeric, bracketed, corresponding to standard conference transactions format)  
**Calibrated Section Depth:** ~500–700 words total, matching the literature review depth observed in `sample_report.pdf`.

---

## 1. Paper 3: Deep Learning Approach to Diabetic Retinopathy Detection

### 1.1 Full Citation (IEEE Style)
[3] B. Tymchenko, P. Marchenko, and D. Spodarets, "Deep learning approach to diabetic retinopathy detection," in *Proc. 15th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP 2020)*, vol. 5: VISAPP, pp. 883–891, 2020, doi: 10.5220/0009169308830891.

### 1.2 Problem Addressed
Automated diabetic retinopathy (DR) grading is severely hampered by label noise, subtle microvascular lesion boundaries, and extreme inter-expert grader disagreement (as established in landmark ophthalmic studies where even retinal specialists disagree on border cases). Furthermore, standard multi-class neural networks fail to capture the continuous pathological evolution of DR across its five stages (No DR, Mild NPDR, Moderate NPDR, Severe NPDR, and Proliferative DR), resulting in unstable convergence on threshold-sensitive clinical metrics such as Cohen’s Quadratic Weighted Kappa (QWK).

### 1.3 Method and Architecture Summary
The authors propose a multi-target learning paradigm that simultaneously trains three distinct output heads on top of a shared convolutional feature extractor:
1. A *Nominal Classification Head* trained with categorical Cross-Entropy and Focal Loss to discriminate discrete severity classes;
2. An *Ordinal Regression Head* trained with binary Focal Loss over $K-1$ rank-ordered sub-tasks (predicting whether disease severity $y > k$ for each threshold $k \in \{0, 1, 2, 3\}$);
3. A *Continuous Regression Head* minimizing Mean Absolute Error (MAE) and Mean Squared Error (MSE) against continuous scalar targets, subsequently rounded or threshold-optimized at inference.
The architecture leverages an ensemble of heavy backbones: EfficientNet-B4 ($380\times380$), EfficientNet-B5 ($456\times456$), and SE-ResNeXt-50 ($32\times4\text{d}$). The models were pre-trained on the EyePACS 2015 dataset (35,126 fundus photographs) before fine-tuning on APTOS 2019, IDRiD, and Messidor. Regularization included label smoothing adapted for both classification and ordinal heads, extensive geometric/color augmentations, and Test-Time Augmentation (TTA). SHAP (Shapley Additive exPlanations) was utilized to provide post-hoc interpretability.

### 1.4 Dataset(s) and Metrics Used (Actual Reported Numbers)
- **Datasets:** EyePACS 2015 (35,126 images for pre-training), APTOS 2019 Blindness Detection competition dataset (3,662 training images), IDRiD (413 images), and Messidor (1,200 images relabeled by a panel of ophthalmologists).
- **Reported Performance Numbers:**
  - *Kaggle Competition Holdout Benchmark:* The proposed approach ranked **54th out of 2,943 teams** globally, achieving an official test-set Quadratic Weighted Kappa score of **0.92546**.
  - *Local Validation (without Test-Time Augmentation, Table 1):*
    - EfficientNet-B4: QWK = **0.965**, Macro F1 = **0.811**, Accuracy = **0.903**, Macro Precision = **0.812**, Macro Recall = **0.976**.
    - EfficientNet-B5: QWK = **0.963**, Macro F1 = **0.815**, Accuracy = **0.907**, Macro Precision = **0.807**, Macro Recall = **0.977**.
    - SE-ResNeXt-50: QWK = **0.957**, Macro F1 = **0.793**, Accuracy = **0.892**, Macro Precision = **0.795**, Macro Recall = **0.971**.
    - Tri-Model Ensemble: QWK = **0.969**, Macro F1 = **0.825**, Accuracy = **0.912**, Macro Precision = **0.825**, Macro Recall = **0.979**.
  - *Local Validation (with Test-Time Augmentation, Table 2):*
    - EfficientNet-B4 + TTA: QWK = **0.966**, Macro F1 = **0.806**, Accuracy = **0.902**.
    - Tri-Model Ensemble + TTA: QWK = **0.971**, Macro F1 = **0.829**, Accuracy = **0.915**, Macro Precision = **0.828**, Macro Recall = **0.979**.

### 1.5 Limitations and Gaps
- **Explicitly Acknowledged by Authors:** The computational overhead of the proposed solution is substantial: training an ensemble of heavy backbones (EfficientNet-B4/B5 and SE-ResNeXt-50) at native resolutions ($380\times380$ to $456\times456$) requires multi-GPU infrastructure and large memory bandwidth. Inference latency is further amplified by multi-crop Test-Time Augmentation, preventing real-time point-of-care mobile screening.
- **Identified Gap (Our Analysis):** Because the authors jointly trained the classification, ordinal, and continuous regression heads within a unified multi-task loss alongside massive EyePACS pre-training, they did not isolate or ablate the independent contribution of each individual head. Consequently, the study leaves open the fundamental research question: how much of the performance advantage is strictly attributable to ordinal rank loss versus continuous regression versus multi-task regularization?

### 1.6 Relevance to Our Ordinal Regression Thesis
Tymchenko et al. provide compelling evidence that ordinal regression heads and threshold-based objectives improve QWK on APTOS 2019. Our study directly answers the ablation gap left by their work: we isolate and compare pure nominal classification, pure ordinal regression (CORAL/CORN), and continuous regression on an identical backbone under controlled experimental conditions, dissecting the exact mechanics of ordinal superiority.

---

## 2. Paper 4: Diabetic Retinopathy Detection Based on Convolutional Neural Networks with SMOTE and CLAHE Techniques Applied to Fundus Images

### 2.1 Full Citation (IEEE Style)
[4] S. Mardianta, Affandy, C. Supriyanto, M. A. Soeleman, and A. Wijaya, "Diabetic retinopathy detection based on convolutional neural networks with SMOTE and CLAHE techniques applied to fundus images," arXiv preprint arXiv:2504.05696, Apr. 2025.

### 2.2 Problem Addressed
Retinal fundus photography frequently exhibits poor contrast, non-uniform illumination, and low visibility of early microvascular lesions (e.g., subtle microaneurysms and tiny intraretinal hemorrhages). Compounding this imaging challenge, clinical DR datasets suffer from acute class imbalance, where severe and proliferative stages represent a tiny fraction of total clinical scans. This distribution skew causes standard convolutional neural networks to bias predictions heavily toward the healthy majority class, severely depressing minority-class recall.

### 2.3 Method and Architecture Summary
The authors present a pipeline combining Contrast Limited Adaptive Histogram Equalization (CLAHE) with Synthetic Minority Over-sampling Technique (SMOTE) ahead of deep feature learning:
1. *Contrast Enhancement:* CLAHE is applied to raw fundus images to amplify local contrast and delineate subtle retinal boundaries without over-amplifying background sensor noise.
2. *Imbalance Rectification:* SMOTE is introduced to synthesize artificial samples for under-represented disease grades by computing nearest neighbors in feature space and linearly interpolating new instances along line segments connecting minority samples.
3. *Classification Backbone:* A pre-trained Xception CNN architecture is fine-tuned to classify retinal images into binary (Normal vs. DR) and multi-class (0 to 4) categories using categorical cross-entropy loss.

### 2.4 Dataset(s) and Metrics Used (Actual Reported Numbers)
- **Dataset:** APTOS 2019 Blindness Detection dataset (partitioned into training and validation sets).
- **Reported Performance Numbers:**
  - *Binary Classification (Normal vs. DR, Table 1):*
    - Accuracy = **99.55%**
    - Precision = **99.55%**
    - Recall (Sensitivity) = **99.55%**
  - *Multi-Class Severity Classification (5 classes, Table 1):*
    - Accuracy = **95.26%**
    - Precision = **95.26%**
    - Recall = **95.17%**
    - F1-Score = **95.21%** (harmonic mean of precision and recall)

### 2.5 Limitations and Gaps
- **Explicitly Acknowledged by Authors:** The paper acknowledges that CLAHE introduces sensitive hyperparameter dependencies (clip limit threshold and tile grid size), which must be hand-tuned. Furthermore, applying SMOTE to high-dimensional image representations incurs elevated computational cost and risks introducing synthetic noise.
- **Identified Gap (Our Analysis):** Synthesizing raw pixels or high-dimensional features via linear SMOTE interpolation fails to respect anatomical retinal geometry, generating artificial fundus artifacts that lack clinical validity. Furthermore, the paper reports suspiciously identical 99.55% scores across binary metrics and 95.26% across multi-class metrics, strongly pointing toward data leakage (e.g., executing SMOTE or augmentation prior to dataset splitting). Crucially, the authors formulate 5-class grading as nominal categorical cross-entropy, failing to report Quadratic Weighted Kappa (QWK) or analyze the severity distance of misclassifications.

### 2.6 Relevance to Our Ordinal Regression Thesis
While Mardianta et al. validate CLAHE as an effective contrast-enhancement technique for fundus photography—which we adopt directly in our Phase 2 data pipeline—their attempt to fix class imbalance via synthetic sample generation (SMOTE) ignores the intrinsic ordinal geometry of DR. By replacing heuristic SMOTE oversampling with mathematically principled ordinal regression (CORAL/CORN), our method inherently handles class-boundary ambiguity without generating artifactual synthetic images.

---

## 3. Cross-Paper Synthesis
While Tymchenko et al. [3] and Mardianta et al. [4] both tackle the challenge of five-stage DR classification on the APTOS 2019 dataset, they approach the problem from opposing ends of the machine learning pipeline: [3] targets the loss function and prediction head by integrating multi-target ordinal heads and heavy ensembling, whereas [4] targets input preprocessing through CLAHE contrast enhancement and SMOTE oversampling. However, [4] relies on nominal softmax classification that fails to evaluate ordinal distance metrics like QWK, while [3] demonstrates that ordinal and regression objectives are crucial for competitive ranking on Kaggle benchmarks (achieving 0.92546 QWK). Synthesizing the strengths of both works, our methodology integrates the rigorous CLAHE preprocessing validated in [4] with the ordinal regression loss formulation advocated in [3], providing a clean, single-backbone ablation that demonstrates how ordinal distance penalties outperform nominal classification without the need for artificial data synthesis or heavy compute ensembles.
