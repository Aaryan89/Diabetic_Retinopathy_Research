# Literature Review: Papers 5 and 6

**Reviewer Assignment:** Member 3  
**Scope:** Explainability, Attention Mechanisms, Fuzzy Membership, and Multimodal Vision-Language Ensembles  
**Target Citation Style:** IEEE (Numeric, bracketed, corresponding to standard conference transactions format)  
**Calibrated Section Depth:** ~500–700 words total, matching the literature review depth observed in `sample_report.pdf`.

---

## 1. Paper 5: Explainable AI for Diabetic Retinopathy Detection Using Deep Learning with Attention Mechanisms and Fuzzy Logic-Based Interpretability

### 1.1 Full Citation (IEEE Style)
[5] A. Karthik, V. Pandiyaraju, and S. Mynampati, "Explainable AI for diabetic retinopathy detection using deep learning with attention mechanisms and fuzzy logic-based interpretability," arXiv preprint arXiv:2511.16294, Nov. 2025.

### 1.2 Problem Addressed
Standard deep convolutional neural networks applied to diabetic retinopathy (DR) operate as opaque black boxes, generating overconfident, discrete multi-class outputs that clinicians hesitate to trust. In real clinical screening, pathological severity is fundamentally a continuous biological spectrum; forcing hard probabilistic boundaries between contiguous stages (e.g., Mild vs. Moderate NPDR) causes severe prediction fragility near decision thresholds and obscures model uncertainty.

### 1.3 Method and Architecture Summary
The authors propose an explainable pipeline integrating channel attention, fuzzy classification, and post-hoc visual attribution:
1. *Feature Extraction & Attention:* An EfficientNetV2-B3 convolutional backbone is augmented with Squeeze-and-Excitation (SE) channel-attention blocks that dynamically recalibrate feature maps to highlight clinically decisive microvascular lesions (e.g., microaneurysms and hard exudates).
2. *Custom Fuzzy Output Layer:* Rather than utilizing a conventional softmax operator, the network implements a fuzzy membership layer that produces continuous, multi-grade membership degrees, capturing diagnostic ambiguity and transition states.
3. *Loss Formulation & Imbalance Handling:* The network is trained with a hybrid objective combining Focal Loss and Label Smoothing over 100 epochs with mixed precision. Class imbalance is mitigated via generative adversarial network (GAN) synthetic oversampling and geometric transformations.
4. *Visual Interpretability:* Gradient-weighted Class Activation Mapping (Grad-CAM) generates heatmaps to verify whether the network fixates on true pathological lesions rather than imaging artifacts. For quantitative evaluation, the 5 stages are grouped into three clinical categories: No DR, Mild/Moderate DR, and Severe/Proliferative DR.

### 1.4 Dataset(s) and Metrics Used (Actual Reported Numbers)
- **Dataset:** APTOS 2019 Blindness Detection dataset (partitioned into a 70–15–15 train–validation–test split, with a holdout test partition containing 366 images).
- **Reported Performance Numbers:**
  - *Overall Classification Accuracy (3-category grouped evaluation):* **91.5%** (reported as 0.91).
  - *Test Set Performance Breakdown (Table I, 366 samples):*
    - Class 0 (No DR, Support = 199): Precision = **0.98**, Recall = **0.99**, F1-Score = **0.99**.
    - Class 1 & 2 (Mild/Moderate DR, Support = 117): Precision = **0.82**, Recall = **0.91**, F1-Score = **0.87**.
    - Class 3 & 4 (Severe/Proliferative DR, Support = 50): Precision = **0.81**, Recall = **0.58**, F1-Score = **0.67**.
  - *Macro-Averaged Metrics:* Precision = **0.87**, Recall = **0.83**, F1-Score = **0.84**.
  - *Weighted-Averaged Metrics:* Precision = **0.91**, Recall = **0.91**, F1-Score = **0.91**.

### 1.5 Limitations and Gaps
- **Explicitly Acknowledged by Authors:** The paper acknowledges that GAN-based synthetic sample generation incurs significant training instability and computational expense, occasionally introducing synthetic hallucinations. Furthermore, despite fuzzy modeling, recall on the clinically urgent Severe/Proliferative DR class remained alarmingly low at 0.58 (missing 42% of high-risk patients).
- **Identified Gap (Our Analysis):** To mask poor multi-class separation, the authors collapsed the standardized five-tier International Clinical Diabetic Retinopathy scale into three coarse categories (combining Mild with Moderate, and Severe with Proliferative). Even under this simplified 3-class scheme, the model achieved only 58% sensitivity on the sight-threatening cohort. More fundamentally, the heuristic fuzzy membership functions produce soft assignments without enforcing mathematical rank monotonicity or penalizing multi-grade distance errors, and the authors fail to report Quadratic Weighted Kappa (QWK).

### 1.6 Relevance to Our Ordinal Regression Thesis
Karthik et al. recognize that DR progression is non-binary and continuous, but their heuristic fuzzy membership layer fails to prevent high false-negative rates on urgent cases (42% missed). Our ordinal regression framework (CORAL/CORN) mathematically replaces heuristic fuzzy memberships with rank-consistent threshold probabilities, enforcing monotonic severity progression across all 5 distinct stages and preventing catastrophic triage undercalls.

---

## 2. Paper 6: From Pixels to Explanations: Interpretable Diabetic Retinopathy Grading with CNN-Transformer Ensembles, Visual Explainability and Vision-Language Models

### 2.1 Full Citation (IEEE Style)
[6] P. B. Khokhar, C. Gravino, F. Palomba, S. Y. Yayilgan, and S. Shaikh, "From pixels to explanations: Interpretable diabetic retinopathy grading with CNN-transformer ensembles, visual explainability and vision-language models," arXiv preprint arXiv:2604.23079, Apr. 2024.

### 2.2 Problem Addressed
Automated DR grading systems predominantly output discrete severity labels without verifiable clinical rationales, limiting trust in ophthalmologic screening programs. Furthermore, standard saliency heatmaps (e.g., Grad-CAM) can be diffuse, noisy, and ungrounded, failing to explain subtle lesion configurations. Meanwhile, individual model architectures exhibit distinct inductive biases, making single-backbone grading vulnerable to dataset noise and fine-grained boundary confusion.

### 2.3 Method and Architecture Summary
The authors implement an extensive multi-stage grading and multimodal explanation framework:
1. *Systematic Backbone Benchmarking:* Six distinct CNN and vision transformer backbones are benchmarked on APTOS 2019 under five-fold cross-validation: ResNet-50, ConvNeXt-Tiny, Swin Transformer, DeiT-Small, EfficientNet-B3, and DenseNet-121.
2. *Ensemble Fusion:* To mitigate individual model blind spots, multiple ensembling strategies are evaluated: hard voting, weighted soft voting, stacking, and hybrid class-level fusion (tailoring backbones to specific severity grades).
3. *Multimodal Explanation Pipeline:* Saliency heatmaps generated via Grad-CAM++ are paired with medical Vision-Language Models (VLMs)—specifically LLaVA-Med-v1.5-Mistral-7B and LLaVA-1.5-1B. The predicted grade distributions and spatial localization cues serve as structured prompts to generate natural language clinical rationales describing specific retinal lesions.

### 2.4 Dataset(s) and Metrics Used (Actual Reported Numbers)
- **Dataset:** APTOS 2019 Blindness Detection dataset evaluated under five-fold cross-validation.
- **Reported Performance Numbers:**
  - *Single-Backbone Benchmarking (5-Fold CV Mean ± SD, Table 1):*
    - ResNet-50: QWK = **0.919 ± 0.014**, Macro-F1 = **0.728 ± 0.063**, Weighted-F1 = **0.858 ± 0.030**, Macro-AUC = **0.962 ± 0.011**.
    - ConvNeXt-Tiny: QWK = **0.914 ± 0.026**, Macro-F1 = **0.732 ± 0.045**, Weighted-F1 = **0.848 ± 0.015**, Macro-AUC = **0.956 ± 0.012**.
    - Swin Transformer: QWK = **0.916 ± 0.006**, Macro-F1 = **0.705 ± 0.056**, Weighted-F1 = **0.835 ± 0.017**, Macro-AUC = **0.954 ± 0.009**.
    - DeiT-Small: QWK = **0.901 ± 0.032**, Macro-F1 = **0.697 ± 0.073**, Weighted-F1 = **0.823 ± 0.031**, Macro-AUC = **0.949 ± 0.010**.
    - EfficientNet-B3: QWK = **0.841 ± 0.035**, Macro-F1 = **0.624 ± 0.030**, Weighted-F1 = **0.778 ± 0.014**, Macro-AUC = **0.928 ± 0.007**.
    - DenseNet-121: QWK = **0.835 ± 0.038**, Macro-F1 = **0.612 ± 0.041**, Weighted-F1 = **0.765 ± 0.022**, Macro-AUC = **0.921 ± 0.012**.
  - *Ensemble Class-Wise Performance (Weighted Soft Voting, Table 5):*
    - Class 0 (No DR): Precision = **0.983 ± 0.012**, Recall = **0.997 ± 0.008**, F1-Score = **0.990 ± 0.007**.
  - *VLM Rationale Generation Evaluation (Table 6):*
    - LLaVA-Med-v1.5-Mistral-7B: Grading Accuracy = **0.844**, Kappa = **0.752**, Macro-AUC = **0.950**, Clinical Coverage = **0.700**, CLIPScore = **0.344**.
    - LLaVA-1.5-1B: Grading Accuracy = **0.844**, Kappa = **0.752**, Macro-AUC = **0.950**, Clinical Coverage = **0.602**, CLIPScore = **0.340**.

### 2.5 Limitations and Gaps
- **Explicitly Acknowledged by Authors:** The enormous computational footprint required to run an ensemble of six heavy CNN/Transformer backbones combined with a 7B-parameter vision-language model prevents deployment on mobile or edge screening devices. Furthermore, paired statistical Wilcoxon tests (Table 4) revealed that the proposed hybrid class-level fusion yielded no statistically significant improvement over simple soft voting ($p = 0.625$ to $1.000$), and VLMs exhibited occasional textual hallucinations of non-existent lesions.
- **Identified Gap (Our Analysis):** Despite benchmarking advanced vision transformers and large language models, the foundational classifiers were trained exclusively with standard categorical cross-entropy. By treating severity grades as nominal categories, the individual backbones suffered suboptimal ordinal calibration (e.g., EfficientNet-B3 achieving only 0.841 QWK). The authors compensated for this loss-level deficiency by stacking multiple models and 7B VLMs, introducing massive architectural complexity to solve what is fundamentally an ordinal loss formulation problem.

### 2.6 Relevance to Our Ordinal Regression Thesis
Khokhar et al. confirm that Quadratic Weighted Kappa (QWK) is the vital metric for clinical DR grading, yet they rely on heavy ensembles and generative VLMs to boost agreement. Our study shows that by substituting standard cross-entropy with ordinal regression (CORAL/CORN), a single lightweight backbone can achieve competitive ordinal calibration, eliminating the parameter bloat and inference latency of multi-model ensembles.

---

## 3. Cross-Paper Synthesis
Both Karthik et al. [5] and Khokhar et al. [6] recognize that standard black-box categorical classification is ill-suited for real-world ophthalmology, attempting to overcome its limitations through external interpretability layers—fuzzy membership modeling in [5] and multimodal VLM text generation in [6]. However, both studies retain nominal cross-entropy as their underlying training mechanism: [5] is forced to collapse the 5 stages into 3 coarse groups yet still suffers 42% false-negatives on severe cases, while [6] requires an ensemble of six distinct CNN and Transformer models to mitigate boundary instability. Our project directly bridges these insights by showing that clinical safety and grade calibration must be solved at the loss function level: ordinal regression provides a mathematically rigorous, rank-ordered decision boundary that respects disease severity progression on a single efficient backbone, without requiring heuristic class-collapsing or parameter-heavy generative ensembles.
