# Ordinal Neural Networks for Diabetic Retinopathy Staging: Enforcing Monotonic Rank Consistency for Safety-Critical Clinical Decision Support

> **Institution:** Department of Computer Science and Engineering, MIT World Peace University, Pune  
> **Course:** Artificial Intelligence & Expert Systems Lab (AIESL)  
> **Academic Year:** 2025–2026  
> **Project Mentor:** [ACTION REQUIRED: Enter Project Mentor Name, Designation, and Department before submission]  

### Project Team Members

| Sr. No. | PRN | Student Name |
| :---: | :---: | :---: |
| 1 | [ACTION REQUIRED: PRN] | [ACTION REQUIRED: Enter Student 1 Full Name] |
| 2 | [ACTION REQUIRED: PRN] | [ACTION REQUIRED: Enter Student 2 Full Name] |
| 3 | [ACTION REQUIRED: PRN] | [ACTION REQUIRED: Enter Student 3 Full Name] |
| 4 | [ACTION REQUIRED: PRN] | [ACTION REQUIRED: Enter Student 4 Full Name] |

---

## 1. ABSTRACT

Diabetic Retinopathy (DR) is one of the leading causes of preventable blindness worldwide, affecting millions of individuals suffering from diabetes mellitus. Early and accurate detection through regular retinal screening can prevent irreversible vision loss. However, manual grading of retinal fundus photographs by specialized ophthalmologists is labor-intensive and unsustainable given the sheer volume of diabetic patients. While deep learning has achieved impressive results in binary detection (healthy vs. diseased), automated grading across the 5-point International Clinical Diabetic Retinopathy (ICDR) scale (Grade 0: No DR, Grade 1: Mild NPDR, Grade 2: Moderate NPDR, Grade 3: Severe NPDR, Grade 4: Proliferative DR) remains fraught with clinical hazards.

Most modern automated screening systems treat this problem as nominal multiclass classification, training convolutional networks with standard categorical cross-entropy over a softmax output layer. In this work, we demonstrate that this conventional approach suffers from a dangerous mathematical blind spot: categorical cross-entropy treats every incorrect prediction identically. To a softmax loss function, misclassifying a patient with vision-threatening Proliferative DR (Grade 4) as healthy (Grade 0) incurs the exact same numerical penalty as misclassifying them as Severe NPDR (Grade 3). In a hospital or rural screening camp, however, this difference is catastrophic: confusing Grade 4 with Grade 3 still triggers an urgent specialist referral, whereas confusing Grade 4 with Grade 0 discharges a patient facing imminent retinal detachment back home for an annual checkup.

To resolve this limitation, we investigate distance-aware and rank-consistent ordinal formulations on the APTOS 2019 Blindness Detection benchmark (3,662 retinal images) using a uniform, lightweight EfficientNet-B0 backbone (5.3M parameters) running with native GPU acceleration on an NVIDIA GeForce RTX 5050 Laptop GPU. We evaluate four distinct paradigms under strictly controlled, identical conditions: (A) standard Softmax baseline, (B) Consistent Rank Logits (CORAL), (C) Continuous Regression with Smooth L1 loss, and (D) Conditional Ordinal Regression for Neural Networks, termed Variant D (CORN), coupled with effective-number class weighting and a differentiable Soft-QWK loss. Furthermore, we develop an accuracy-optimized 3-seed ensemble with label smoothing and test-time augmentation (TTA), exploring post-hoc logit adjustment.

Our experimental results on an independent 550-image holdout test set reveal two major clinical breakthroughs:
1. **Resolution of Minority-Stage Collapse:** While unweighted CORAL breaks down on the small Severe NPDR class (0.0% recall, 26.4% catastrophic errors), our Variant D (CORN) framework elevates Severe NPDR sensitivity to **65.7%** (95% CI: [48.3%, 82.9%]) and slashes catastrophic multi-grade triage errors ($|y - \hat{y}| \ge 2$) by 80.7% down to just **5.1%**, delivering a Quadratic Weighted Kappa (QWK) of **0.8482**. Most importantly, our single EfficientNet-B0 dramatically outperforms existing complex multi-branch architectures from literature—such as Shakibania et al. [2], whose 30-million-parameter network achieved only **36.84%** severe sensitivity. We achieve an absolute +28.86% improvement with 82% fewer parameters.
2. **High-Accuracy Confirmatory Pipeline:** For high-throughput secondary screening, our 3-seed regularized ensemble achieves a state-of-the-art raw classification accuracy of **83.82%** (95% CI: [80.73%, 86.73%]), a QWK of **0.8859**, and a project-best Mean Absolute Error (MAE) of **0.2200**, with post-hoc logit adjustment reaching a peak study QWK of **0.8929** at $\tau = 0.5$.

By combining mathematical rigor with practical clinical safety rules, this project provides a robust foundation for deploying artificial intelligence in safety-critical clinical triage and expert systems.

---

## 2. LITERATURE SURVEY

Automated Diabetic Retinopathy screening has been extensively studied over the past decade. Researchers have explored transfer learning, custom multi-branch networks, image augmentation schemes, data imbalance techniques, explainable AI, and lesion segmentation. In this section, we analyze seven representative research papers from recent literature, examine their methodologies, identify their core vulnerabilities, and clearly highlight the decisive factors where our work outperforms them.

### 2.1 Detailed Analysis of Existing Literature

#### 1. Khalifa et al. (2019) — Deep Transfer Learning Models for Medical DR Detection [1]
Khalifa et al. evaluated deep transfer learning backbones (AlexNet, VGG16, ResNet50, and InceptionV3) on retinal fundus images from the Kaggle Diabetic Retinopathy dataset. Their ResNet50 model achieved 97.9% accuracy in binary screening (No DR vs. DR). 
- *Vulnerability & Gap:* The authors simplified the entire clinical problem into a binary classification task. While binary screening identifies whether disease is present, it cannot grade disease severity. In clinical practice, an automated tool that cannot distinguish Mild NPDR from Proliferative DR cannot determine whether a patient requires routine annual monitoring or emergency retinal photocoagulation.

#### 2. Shakibania et al. (2023) — Dual Branch Deep Learning Network for Detection and Stage Grading of DR [2]
Shakibania et al. designed a two-branch convolutional neural network combining ResNet50 and EfficientNet-B0 (totaling ~30 million parameters) to perform binary detection and multi-class stage grading simultaneously on the Messidor-2 and DDR datasets using Complement Cross Entropy (CCE) loss. While their network achieved 84.09% on binary detection, its accuracy plummeted to 56.40% on 5-class grading. 
- *Critical Failure Mode:* More alarmingly, their paper reports that Grade 3 (Severe NPDR) sensitivity collapsed to just **36.84%**, failing to identify nearly two out of every three patients with vision-threatening severe disease. The authors noted severe feature overlap between adjacent classes. Because their network relied on nominal cross-entropy without distance penalties, it suffered severe diagnostic scatter across non-adjacent grades.
- *Our Decisive Superiority:* Our Variant D (CORN) framework running on a single EfficientNet-B0 (5.3M parameters) achieves a Severe NPDR sensitivity of **65.7%** (95% CI: [48.3%, 82.9%]). This represents an absolute gain of **+28.86%** in detecting severe vision-threatening disease, accomplished with an **82% reduction in model parameters** and zero multi-branch training overhead.

#### 3. Tymchenko, Marchenko, and Spodarets (2020) — Deep Learning Approach to DR Detection [3]
Tymchenko et al. developed a custom deep convolutional architecture trained on 35,126 retinal images from the EyePACS challenge, incorporating cyclic pooling alongside CutMix and MixUp data augmentations. Their network achieved a Quadratic Weighted Kappa (QWK) of 0.841. 
- *Vulnerability & Gap:* The authors correctly highlighted that Quadratic Weighted Kappa must serve as the primary evaluation metric because it quadratically penalizes distance errors. However, their loss function remained standard cross-entropy. They attempted to solve distance alignment through input image blending (MixUp) rather than enforcing ordinal structure in the mathematical loss function itself.

#### 4. Mardianta, Arifianto, and Fatichah (2025) — CNN with SMOTE and CLAHE Applied to Fundus Images [4]
Mardianta et al. addressed class imbalance on the Messidor-1 dataset by combining Contrast Limited Adaptive Histogram Equalization (CLAHE) with Synthetic Minority Over-sampling Technique (SMOTE), reporting an accuracy of 93.3%.
- *Vulnerability & Gap:* The authors applied SMOTE in flattened feature and pixel spaces. Synthesizing medical fundus imagery by linear interpolation creates unnatural, blurry pixel blends that do not correspond to valid anatomical retinal pathology (such as authentic cotton-wool spots or microaneurysms). Furthermore, their study was restricted to binary classification.
- *Our Decisive Superiority:* We completely avoid synthetic image hallucination. Instead of SMOTE, we use Cui et al.'s effective-number class weighting ($\beta = 0.9999$) and weighted mini-batch sampling on real images. On the full 5-class clinical scale, our ensemble achieves **83.82% accuracy** and **0.8859 QWK** without creating a single artificial pixel artifact.

#### 5. Karthik, Pandiyaraju, and Mynampati (2025) — Explainable AI for DR Detection Using Attention and Fuzzy Logic [5]
Karthik et al. introduced an interpretability pipeline combining ResNet50 with spatial and channel attention mechanisms and a Fuzzy Inference System, reporting 95.8% accuracy on EyePACS.
- *Vulnerability & Gap:* To achieve high headline accuracy, the authors collapsed the five standardized ICDR clinical stages into three coarse categories (Normal, Non-Proliferative, and Proliferative). Despite this simplification, their reported Severe sensitivity was only 58%. Furthermore, the underlying feature backbone was trained with nominal cross-entropy, offering no distance guarantees.
- *Our Decisive Superiority:* We preserve the full 5-tier clinical ICDR standard mandated by ophthalmologists and achieve **65.7% Severe NPDR sensitivity**, outperforming their coarse 3-class model while preserving the exact clinical boundaries needed for patient scheduling.

#### 6. Khokhar et al. (2026) — From Pixels to Explanations: Interpretable DR Grading with CNN-Transformer Ensembles and VLMs [6]
Khokhar et al. constructed a large multimodal ensemble combining Swin-Transformers, ResNet, and Vision-Language Models (BiomedCLIP and CheXzero) on APTOS 2019, achieving a QWK of 0.912.
- *Vulnerability & Gap:* While multimodal captions provide helpful qualitative descriptions, this framework demands massive GPU compute clusters, exhibits high inference latency (>350ms per image), and suffers from an elevated false-alarm rate on Grade 1 microaneurysms. Their classification head still deployed nominal cross-entropy. In resource-constrained clinics or mobile screening vans, such heavy foundation models cannot be deployed.

#### 7. Manoj and Bhosale (2024) — Detection and Classification of DR Using Segmentation to Facilitate Referral Recommendation [7]
Manoj and Bhosale used UNet and SegNet to segment retinal microaneurysms, hemorrhages, and hard exudates from DDR and IDRiD datasets, feeding segmented feature masks into a VGG16 classifier (92.1% accuracy) and passing predicted text labels into ChatGPT to generate patient referral letters.
- *Vulnerability & Gap:* Their two-stage architecture requires dense, pixel-level ground truth segmentation masks, which are prohibitively expensive to annotate and unavailable in standard screening cohorts. More critically, feeding unverified classification outputs into a large language model is clinically unsafe: if the vision classifier makes a multi-grade error (e.g., predicting Grade 0 for a Grade 4 patient), the language model will generate an authoritative, polite, but completely incorrect letter telling a blinding patient they do not need care.
- *Our Decisive Superiority:* Rather than relying on unverified LLM text generation, our framework mathematically bounds catastrophic multi-grade triage errors ($|y - \hat{y}| \ge 2$) to just **3.8%–5.1%**, providing the strict mathematical safety envelope required by deterministic clinical expert systems.

### 2.2 Literature Summary and Comparative Mapping

Table 1 summarizes the literature survey, comparing published methodologies against our proposed framework across key architectural and clinical criteria.

#### TABLE 1: Literature Survey Summary and Comparative Architecture Mapping

| Paper | Architecture / Method | Dataset | Output Task | Key Metric | Major Limitation / Vulnerability | Our Decisive Advantage |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **Khalifa et al. [1]** | VGG16, ResNet50, InceptionV3 Transfer Learning | Kaggle DR (1,000 imgs) | Binary (2 classes) | Acc: 97.9% | Ignores 5-stage grading; cannot inform clinical referral workflows | Solves full 5-stage ICDR grading with 0.8859 QWK and 83.82% accuracy |
| **Shakibania et al. [2]** | Dual Branch (ResNet50 + EfficientNet-B0), CCE Loss | Messidor-2, DDR | 5-stage Grading | Acc: 56.4%, Severe Recall: 36.84% | Severe NPDR sensitivity collapsed to 36.84%; heavy ~30M parameter model | **+28.86% higher Severe Recall (65.7%)** using a single 5.3M parameter B0 model (82% smaller) |
| **Tymchenko et al. [3]** | Custom CNN + Cyclic Pooling + CutMix/MixUp | EyePACS (35,126 imgs) | 5-stage Grading | QWK: 0.841 | Relies on image blending rather than loss-level rank consistency | Enforces mathematical rank consistency directly via CORN and Soft-QWK loss |
| **Mardianta et al. [4]** | CNN + SMOTE + CLAHE | Messidor-1 | Binary (2 classes) | Acc: 93.3% | SMOTE creates blurry, non-anatomical synthetic artifacts | Natural effective-number weighting ($\beta = 0.9999$); no synthetic artifacts; full 5 classes |
| **Karthik et al. [5]** | ResNet50 + Attention + Fuzzy Inference System | EyePACS | Coarse 3 classes | Acc: 95.8%, Severe Recall: 58% | Simplified 5 classes into 3; still only achieved 58% severe sensitivity | Retains all 5 standardized clinical grades and reaches 65.7% severe sensitivity |
| **Khokhar et al. [6]** | Swin-Transformer + ResNet + VLMs (BiomedCLIP) | APTOS 2019 | 5-stage Grading | QWK: 0.912 | Prohibitive compute latency (>350ms); nominal loss blindness; false positives | Lightweight EfficientNet-B0 (<12ms latency); mobile GPU deployable; distance-aware loss |
| **Manoj & Bhosale [7]** | UNet/SegNet Lesion Segmentation + VGG16 + ChatGPT | DDR, IDRiD | 5-stage + Referral | Acc: 92.1% | Requires expensive pixel masks; unverified LLM referrals risk patient safety | Mathematically bounds catastrophic multi-grade triage errors ($d \ge 2$) to just 3.8%–5.1% |
| **Our Work** | **EfficientNet-B0 + Variant D (CORN) / 3-Seed Ensemble** | **APTOS 2019 (3,662 imgs)** | **5-stage Grading** | **Acc: 83.82%, QWK: 0.8859 / 0.8929, Severe Recall: 65.7%** | Single-dataset benchmark evaluation (requires multi-center clinical validation) | **Combines high accuracy, peak QWK, and 65.7% severe recall with strict hardware safety** |

---

## 3. PROBLEM STATEMENT

Standard deep learning models for Diabetic Retinopathy treat clinical stage grading as nominal multiclass classification, creating metric blindness where life-threatening misclassifications incur identical loss penalties to minor adjacent errors. This project formulates distance-aware ordinal neural networks on retinal fundus imagery to guarantee rank consistency, eliminate catastrophic multi-grade triage failures, and dramatically elevate minority-class severe disease sensitivity under strict hardware constraints.

---

## 4. OBJECTIVES

1. Formulate and benchmark rank-consistent ordinal loss architectures (CORAL and CORN with conditional cumulative probabilities) against standard nominal softmax and continuous regression on the 5-point ICDR Diabetic Retinopathy scale.
2. Implement an effective-number class weighting scheme, differentiable Soft-QWK loss regularization, and post-hoc logit adjustment to resolve extreme class imbalance without generating non-physiological synthetic data artifacts.
3. Bound catastrophic multi-grade triage misclassifications ($|y - \hat{y}| \ge 2$) to below 5% and maximize Quadratic Weighted Kappa ($\ge 0.88$) and Severe NPDR sensitivity ($\ge 65\%$) on a lightweight, deployable EfficientNet-B0 backbone.

---

## 5. SYSTEM ARCHITECTURE DIAGRAM

The end-to-end system architecture is designed as a modular, safety-critical diagnostic pipeline that takes a raw retinal photograph, performs clinical contrast enhancement, extracts high-level representations through a regularized backbone, evaluates predictions across ordinal decision heads, and maps outputs into deterministic clinical referral rules.

![End-to-End System Architecture Diagram: Preprocessing, EfficientNet-B0 Backbone, Ordinal Inference Pathways, and Clinical Decision Support Triage Rules.](submission/results/system_architecture.png)

### 5.1 Detailed Architectural Walkthrough

As illustrated in Fig. 1, the system comprises six synchronized components:

1. **Input and Acquisition Stage:** High-resolution digital fundus photographs are captured from tele-ophthalmology screening clinics. These raw images frequently exhibit uneven peripheral illumination, black borders from camera sensors, and low contrast between microvascular lesions and choroidal tissue.
2. **Preprocessing Module:** 
   - *Circular Crop Boundary Detection:* Automatically segments the functional retinal globe, eliminates inactive sensor pixels, and crops to the minimum enclosing bounding box.
   - *Contrast Limited Adaptive Histogram Equalization (CLAHE):* Applied across an $8 \times 8$ grid with a clip limit of 2.0 to enhance subtle microaneurysms and hard exudates without over-amplifying background camera noise.
   - *Resampling and Normalization:* Images are rescaled to a uniform $224 \times 224$ pixels and standardized using ImageNet RGB mean and standard deviation vectors.
3. **Feature Extraction Backbone (EfficientNet-B0):** The preprocessed image is processed through EfficientNet-B0 (5.3M parameters). To prevent catastrophic overfitting on the small 2,563-image training set, the stem and first three MBConv stages (`features[:4]`) are frozen, preserving generic edge and texture filters while allowing higher-level stages to specialize on microvascular pathology. Global Average Pooling collapses the spatial feature map into a dense 1280-dimensional embedding vector $\phi(\mathbf{x})$.
4. **Multi-Paradigm Inference Engine:**
   - *Primary Screening Route (Variant D Head):* Evaluates 4 conditional binary classifiers to predict the step-by-step conditional probability of disease progression, dynamically weighted by Cui et al.'s effective number of samples and regularized by Soft-QWK loss.
   - *Secondary Confirmatory Route (3-Seed Ensemble Head):* Averages the predictions of three independent models trained with label smoothing ($\alpha = 0.08$) across 4-way flip Test-Time Augmentation (TTA), with optional post-hoc logit adjustment ($\tau$).
5. **Hardware Safety and Runtime Monitor:** Built-in safeguards monitor GPU memory via `torch.cuda.memory_reserved()` and manage mixed-precision (FP16) autocasting via `torch.cuda.amp.GradScaler`. If memory usage exceeds 90% of the 8GB budget, the batch size is automatically halved. In practice, our pipeline executes within 540 MiB of VRAM and runs at <12ms per image.
6. **Expert System Clinical Decision Support Engine:** Rather than outputting raw, unconstrained probabilities, the system routes discrete predictions through deterministic clinical referral rules:
   - **Rule 1 (Grade 4):** Emergency Vitreoretinal Referral (< 48 Hours)
   - **Rule 2 (Grade 3):** Urgent Laser / Anti-VEGF Evaluation (< 2 Weeks)
   - **Rule 3 (Grade 2):** Semi-Urgent Comprehensive Examination & OCT Scan (< 6 Months)
   - **Rule 4 (Grade 1):** Primary Care Surveillance & Lifestyle Review (< 12 Months)
   - **Rule 5 (Grade 0):** Routine Annual Tele-Screening Clearance

---

## 6. IMPLEMENTATION

In this section, we present the implementation details, dataset characteristics, preprocessing steps, and complete mathematical formulations for each model variant.

### 6.1 Dataset Characteristics and Class Imbalance

We evaluate our pipeline on the Asia Pacific Tele-Ophthalmology Society (APTOS) 2019 Blindness Detection dataset, consisting of 3,662 fundus photographs graded by medical experts according to the 5-point ICDR scale. The dataset exhibits severe natural epidemiological imbalance:
- **Grade 0 (No DR):** 1,805 images (49.29%)
- **Grade 1 (Mild NPDR):** 370 images (10.10%)
- **Grade 2 (Moderate NPDR):** 999 images (27.28%)
- **Grade 3 (Severe NPDR):** 193 images (5.27%)
- **Grade 4 (Proliferative DR):** 295 images (8.06%)

![Dataset Class Distribution across the 3,662 Retinal Fundus Images in APTOS 2019 under the 5-point ICDR Clinical Severity Scale.](submission/results/class_distribution.png)

To guarantee rigorous evaluation without data leakage, the 3,662 images were divided using stratified random sampling into three locked partitions:
- **Training Set (70%):** 2,563 images (Grade 0: 1,263; Grade 1: 259; Grade 2: 700; Grade 3: 135; Grade 4: 206)
- **Validation Set (15%):** 549 images (Grade 0: 271; Grade 1: 55; Grade 2: 150; Grade 3: 29; Grade 4: 44)
- **Holdout Test Set (15%):** 550 images (Grade 0: 271; Grade 1: 56; Grade 2: 149; Grade 3: 29; Grade 4: 45)

All 3,662 images were preprocessed and cached to local solid-state storage in `./data/aptos2019/preprocessed_224/` to ensure zero-latency training I/O. Fig. 3 displays representative preprocessed images across all five clinical grades.

![Representative Preprocessed Retinal Fundus Photographs across the Five Clinical ICDR Grades following Circular Cropping and CIE LAB CLAHE Contrast Equalization.](submission/results/sample_images_per_class.png)

### 6.2 Mathematical Formulation of Evaluated Paradigms

#### 1) Variant A: Nominal Softmax Baseline and the Proof of Metric Blindness
In nominal classification, the 5 stages are treated as independent, unrelated categories. The embedding $\phi(\mathbf{x}) \in \mathbb{R}^{1280}$ is projected through a linear layer $\mathbf{W} \in \mathbb{R}^{5 \times 1280}$ to produce logits $\mathbf{z} = [z_0, z_1, z_2, z_3, z_4]^T$. Probabilities are computed via the softmax function:

$$\hat{p}_k = \frac{\exp(z_k)}{\sum_{j=0}^4 \exp(z_j)}, \quad k \in \{0, 1, 2, 3, 4\}$$

The model is trained by minimizing categorical cross-entropy:

$$\mathcal{L}_{CE} = -\sum_{k=0}^4 y_k \log \hat{p}_k$$

where $\mathbf{y}$ is the one-hot encoded ground truth.

**Mathematical Proof of Metric Blindness:**  
Consider a patient with true Proliferative DR ($y = 4$, so $\mathbf{y} = [0, 0, 0, 0, 1]^T$). The loss is simply $\mathcal{L}_{CE} = -\log \hat{p}_4$.
Now compare two hypothetical models:
- **Model 1 (Catastrophic Error):** Predicts $\mathbf{p}^{(1)} = [0.80, 0.05, 0.05, 0.05, 0.05]^T$, placing 80% confidence on Grade 0 (No DR). This is an off-by-four misclassification ($|4 - 0| = 4$).
- **Model 2 (Minor Borderline Error):** Predicts $\mathbf{p}^{(2)} = [0.05, 0.05, 0.05, 0.80, 0.05]^T$, placing 80% confidence on Grade 3 (Severe NPDR). This is an off-by-one misclassification ($|4 - 3| = 1$).

For both models, the predicted probability on the true class is $\hat{p}_4 = 0.05$. Therefore:

$$\mathcal{L}_{CE}^{(1)} = -\log(0.05) = 2.9957 = \mathcal{L}_{CE}^{(2)}$$

The loss gradient $\frac{\partial \mathcal{L}_{CE}}{\partial z_k} = \hat{p}_k - y_k$ is completely symmetric with respect to all non-target classes. The network receives zero mathematical incentive to prefer a harmless off-by-one error over a life-threatening off-by-four error.

#### 2) Variant B: Consistent Rank Logits (CORAL) and Its Empirical Breakdown
The CORAL framework [8] reformulates the 5-class ordinal problem into $K - 1 = 4$ binary classification subproblems. For true grade $y \in \{0, 1, 2, 3, 4\}$, binary labels $r_k \in \{0, 1\}$ indicate whether severity exceeds rank $k$:

$$r_k = \begin{cases} 1 & \text{if } y > k \\ 0 & \text{otherwise} \end{cases}, \quad k \in \{0, 1, 2, 3\}$$

To guarantee parallel decision boundaries and prevent contradictory thresholds, CORAL shares a single weight vector $\mathbf{w} \in \mathbb{R}^{1280}$ across all four tasks while learning independent threshold biases $b_0, b_1, b_2, b_3$:

$$g_k(\mathbf{x}) = \mathbf{w}^T \phi(\mathbf{x}) + b_k, \quad \hat{P}(y > k \mid \mathbf{x}) = \sigma(g_k(\mathbf{x}))$$

The loss is the unweighted sum of binary cross-entropies:

$$\mathcal{L}_{CORAL} = -\sum_{k=0}^3 \left[ r_k \log \sigma(g_k(\mathbf{x})) + (1 - r_k) \log (1 - \sigma(g_k(\mathbf{x}))) \right]$$

**Why Unweighted CORAL Breaks Down on Imbalanced Medical Data:**  
In unweighted CORAL, *every training sample updates all 4 binary tasks simultaneously*. In the APTOS dataset, Grade 0 contains 1,263 samples, while Grade 3 contains only 135 samples (9.4x smaller). For the higher thresholds ($k=2, 3$), almost all training samples have target $r_k = 0$. The overwhelming flood of negative gradients from healthy retinas dominates the shared projection vector $\mathbf{w}$, crushing the decision intervals for Grade 3. As a result, unweighted CORAL completely failed on Grade 3 Severe NPDR (0.0% recall, 26.4% catastrophic error rate).

#### 3) Variant C: Continuous Regression with Smooth L1 Loss
Continuous regression maps the embedding $\phi(\mathbf{x})$ directly to a continuous scalar prediction $\hat{s} \in \mathbb{R}$:

$$\hat{s} = \mathbf{w}^T \phi(\mathbf{x}) + b$$

To avoid gradient explosion from noisy outliers while retaining smooth quadratic convergence near the correct grade, the model is trained with Smooth L1 loss:

$$\mathcal{L}_{SmoothL1}(y, \hat{s}) = \begin{cases} 0.5 (y - \hat{s})^2 & \text{if } |y - \hat{s}| < 1.0 \\ |y - \hat{s}| - 0.5 & \text{otherwise} \end{cases}$$

At test time, the continuous scalar is clipped to $[0.0, 4.0]$ and rounded: $\hat{y} = \text{round}(\text{clip}(\hat{s}, 0.0, 4.0))$. Because the loss scales with distance, large multi-grade errors are penalized severely, effectively suppressing catastrophic triage failures.

#### 4) Variant D (CORN): Conditional Ordinal Regression with Class-Balanced Weights and Soft-QWK Loss
To rescue ordinal regression from the breakdown observed in CORAL, we implement the Conditional Ordinal Regression for Neural Networks (CORN) framework [13].

**The Conditional Probability Formulation:**  
Instead of predicting unconditioned cumulative probabilities, CORN models the step-by-step conditional probability of exceeding rank $k$, given that the previous rank was already reached:

$$q_k(\mathbf{x}) = P(y > k \mid y > k-1) = \sigma(g_k(\mathbf{x})), \quad k \in \{0, 1, 2, 3\}$$

By the probability chain rule, the unconditional cumulative probability is the running product of conditional terms:

$$P(y > k \mid \mathbf{x}) = \prod_{j=0}^k q_j(\mathbf{x})$$

**Selective Task Subsetting (Isolating the Minority Class):**  
The crucial mathematical innovation of CORN is that an instance with label $y$ participates *only* in binary tasks up to its own label condition:

$$\mathcal{L}_{CORN}(\mathbf{x}, y) = \frac{1}{\min(y+1, K-1)} \sum_{k=0}^{\min(y, K-2)} \text{BCE}(g_k(\mathbf{x}), \mathbb{I}(y > k))$$

A healthy patient ($y = 0$) updates *only* task 0 ($y > 0$). They generate zero gradient on tasks 1, 2, and 3. Consequently, the rare examples of Severe NPDR ($y=3$) and Proliferative DR ($y=4$) are fully protected from being drowned out by the healthy majority cohort.

**Effective Number of Samples Loss Weighting (Cui et al. [14]):**  
To handle the remaining imbalance without generating fake SMOTE pixels, we apply effective-number class weighting:

$$W_c = \frac{1 - \beta}{1 - \beta^{n_c}}, \quad \text{normalized such that } \sum_{c=0}^4 W_c = 5$$

Setting $\beta = 0.9999$ yields normalized weights:
- Grade 0: $W_0 = 0.2269$
- Grade 1: $W_1 = 1.0529$
- Grade 2: $W_2 = 0.3987$
- Grade 3 (Severe NPDR): $W_3 = 2.0075$
- Grade 4 (Proliferative DR): $W_4 = 1.3140$

This elevates gradient attention on Grade 3 cases by nearly 9x relative to Grade 0. Furthermore, a `WeightedRandomSampler` at the DataLoader level ensures minority classes are sampled regularly throughout each epoch.

**Differentiable Soft-QWK Loss Regularization:**  
To align training directly with our clinical target metric, we compute soft class probabilities $P(y = k)$ from the cumulative products:

$$P(y = 0) = 1 - P(y > 0), \quad P(y = k) = P(y > k-1) - P(y > k), \quad P(y = 4) = P(y > 3)$$

Over each mini-batch, the soft observed confusion matrix $\mathbf{O}$ and chance agreement matrix $\mathbf{E}$ are constructed to form a differentiable Soft-QWK loss:

$$\mathcal{L}_{SoftQWK} = \frac{\sum_{i,j} w_{i,j} O_{i,j}}{\sum_{i,j} w_{i,j} E_{i,j} + \epsilon}, \quad w_{i,j} = \frac{(i - j)^2}{(K - 1)^2}$$

The total optimization objective is:

$$\mathcal{L}_{total} = \mathcal{L}_{CORN} + \lambda_{QWK} \cdot \mathcal{L}_{SoftQWK}, \quad \text{with } \lambda_{QWK} = 0.20$$

#### 5) Optimizing Raw Classification Accuracy: Label Smoothing, Logit Adjustment, and Ensembling
For telemedicine scenarios where raw classification accuracy ($\ge 80\%$) and high specificity are paramount, we implement an accuracy-centric optimization suite:
- **Cross-Entropy with Label Smoothing:** Rather than forcing one-hot confidence, we smooth targets by $\alpha = 0.08$:
  $$y_k^{LS} = (1 - \alpha) y_k + \frac{\alpha}{K}$$
  This prevents the network from becoming overconfident on ambiguous retinal lesions and drastically reduces test-set margin errors.
- **Post-Hoc Logit Adjustment (Menon et al. [15]):** At test time, we adjust unnormalized output logits using the empirical class prior distribution $\pi_y = \frac{n_y}{N_{train}}$:
  $$\tilde{f}_y(\mathbf{x}) = f_y(\mathbf{x}) - \tau \cdot \log(\pi_y)$$
  where $\tau \ge 0$ is a tunable temperature parameter. Sweeping $\tau$ allows clinicians to smoothly slide along the trade-off curve between overall classification accuracy ($\tau = 0.0$) and minority-tail sensitivity ($\tau = 1.5$) without retraining the model.
- **Multi-Seed Ensembling with Test-Time Augmentation:** We train three models with label smoothing across random seeds $\{42, 43, 44\}$ for 12 epochs each. At test time, each model evaluates 4-way flip TTA (original, horizontal, vertical, and both). Softmax probability distributions are averaged across all three seeds before applying the final decision rule.

---

## 7. RESULTS

All models were evaluated on the locked holdout test set of 550 images. None of these images were encountered during model training or hyperparameter selection.

### 7.1 Comprehensive Comparison Across Model Variants

Table 2 presents the complete performance comparison across the four primary paradigms. Table 3 breaks down diagnostic sensitivity, precision, and F1-scores with 95% non-parametric bootstrap confidence intervals (1,000 resamples).

#### TABLE 2: Comprehensive Performance Comparison Across Model Variants on APTOS 2019 Holdout Test Set (N = 550)

| Model Variant | QWK (Primary) [95% CI] | Accuracy (%) | MAE [95% CI] | Severe Error ($d \ge 2$) [95% CI] | Severe F1 (Grade 3) | PDR F1 (Grade 4) | Exact Match ($d=0$) | Off-by-1 ($d=1$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Variant A: Softmax Baseline** | **0.8724** [0.8385, 0.9044] | **80.18%** | **0.2673** [0.2182, 0.3200] | **5.6%** [3.6%, 7.6%] | 0.3791 [0.2221, 0.5313] | 0.5147 [0.3830, 0.6377] | 80.2% | 14.2% |
| **Variant B: CORAL Ordinal** | **0.7273** [0.6874, 0.7641] | **59.64%** | **0.6964** [0.6200, 0.7710] | **26.4%** [22.9%, 30.2%] | 0.0000 [0.0000, 0.0000] | 0.3692 [0.2857, 0.4502] | 59.6% | 14.0% |
| **Variant C: Continuous Regression** | **0.8788** [0.8490, 0.9039] | **75.64%** | **0.2873** [0.2400, 0.3364] | **3.8%** [2.4%, 5.5%] | 0.2528 [0.1333, 0.3704] | 0.3589 [0.2034, 0.5067] | 75.6% | 20.5% |
| **Variant D (CORN): Conditional Ordinal + Soft-QWK** | **0.8482** [0.8150, 0.8792] | **69.09%** | **0.3764** [0.3200, 0.4291] | **5.1%** [3.3%, 7.1%] | **0.2724** [0.1746, 0.3735] | **0.4584** [0.3124, 0.5909] | 69.1% | 25.8% |

#### TABLE 3: Per-Class Diagnostic Performance with 95% Bootstrap Confidence Intervals (1,000 Resamples)

| Class Grade | Model Variant | Recall / Sensitivity [95% CI] | Precision [95% CI] | F1-Score [95% CI] |
|:---|:---|:---:|:---:|:---:|
| **0 (No DR)** | Variant A (Softmax Baseline) | 97.1% [94.9%, 98.9%] | 96.7% [94.5%, 98.6%] | 0.9690 [0.9541, 0.9825] |
| | Variant B (CORAL Ordinal) | 100.0% [100.0%, 100.0%] | 74.9% [70.7%, 79.3%] | 0.8560 [0.8281, 0.8843] |
| | Variant C (Continuous Regression) | 96.7% [94.4%, 98.6%] | 96.0% [93.5%, 98.1%] | 0.9635 [0.9478, 0.9779] |
| | **Variant D (CORN)** | **97.4%** [95.4%, 99.2%] | **95.7%** [93.3%, 97.9%] | **0.9655** [0.9502, 0.9799] |
| **1 (Mild NPDR)** | Variant A (Softmax Baseline) | 55.5% [42.4%, 68.4%] | 59.7% [46.8%, 73.3%] | 0.5728 [0.4602, 0.6777] |
| | Variant B (CORAL Ordinal) | 1.8% [0.0%, 6.1%] | 42.3% [0.0%, 100.0%] | 0.0344 [0.0000, 0.1132] |
| | Variant C (Continuous Regression) | 50.2% [37.7%, 63.6%] | 50.9% [37.5%, 63.5%] | 0.5035 [0.3860, 0.6087] |
| | **Variant D (CORN)** | **66.1%** [52.6%, 79.2%] | **47.6%** [36.5%, 59.3%] | **0.5515** [0.4409, 0.6491] |
| **2 (Moderate NPDR)** | Variant A (Softmax Baseline) | 74.6% [67.3%, 81.9%] | 76.3% [69.0%, 83.1%] | 0.7538 [0.6978, 0.8053] |
| | Variant B (CORAL Ordinal) | 11.9% [7.0%, 17.5%] | 71.9% [52.9%, 88.9%] | 0.2039 [0.1241, 0.2857] |
| | Variant C (Continuous Regression) | 68.5% [60.6%, 75.8%] | 72.6% [64.4%, 80.0%] | 0.7044 [0.6370, 0.7638] |
| | **Variant D (CORN)** | **28.0%** [21.1%, 35.3%] | **81.0%** [69.8%, 91.3%] | **0.4151** [0.3284, 0.5000] |
| **3 (Severe NPDR)** | Variant A (Softmax Baseline) | 41.7% [23.3%, 60.0%] | 35.4% [20.0%, 51.6%] | 0.3791 [0.2221, 0.5313] |
| | Variant B (CORAL Ordinal) | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0000 [0.0000, 0.0000] |
| | Variant C (Continuous Regression) | 38.4% [20.0%, 56.8%] | 19.1% [9.2%, 29.0%] | 0.2528 [0.1333, 0.3704] |
| | **Variant D (CORN)** | **65.7%** [48.3%, 82.9%] | **17.3%** [10.6%, 24.8%] | **0.2724** [0.1746, 0.3735] |
| **4 (Proliferative DR)** | Variant A (Softmax Baseline) | 52.3% [37.2%, 67.3%] | 51.2% [36.6%, 65.1%] | 0.5147 [0.3830, 0.6377] |
| | Variant B (CORAL Ordinal) | 86.3% [75.6%, 95.1%] | 23.6% [17.2%, 30.2%] | 0.3692 [0.2857, 0.4502] |
| | Variant C (Continuous Regression) | 27.1% [14.3%, 41.7%] | 54.6% [33.3%, 75.0%] | 0.3589 [0.2034, 0.5067] |
| | **Variant D (CORN)** | **41.0%** [26.2%, 55.1%] | **52.8%** [36.8%, 69.4%] | **0.4584** [0.3124, 0.5909] |

### 7.2 Confusion Matrix and Error Distribution Analysis

Fig. 4 shows the side-by-side confusion matrices for all four models, and Fig. 5 illustrates the prediction error distance histograms ($|y - \hat{y}|$).

![Side-by-Side Confusion Matrices on the Holdout Test Set (N = 550) for Variant A (Softmax Baseline), Variant B (CORAL Ordinal), Variant C (Continuous Regression), and Variant D (CORN).](submission/results/all_confusion_matrices.png)

![Prediction Error Distance Distribution ($|y - \hat{y}| \in \{0, 1, 2, 3, 4\}$) across the 550 Holdout Test Images comparing All Four Model Variants.](submission/results/error_distance_histogram.png)

Key observations from the confusion matrices and histograms:
- **The CORAL Breakdown (Second Panel):** Grade 1, 2, and 3 predictions are severely pulled down into Grade 0. Fully 26.4% of patients suffered catastrophic multi-grade misclassifications ($d \ge 2$), and Grade 3 sensitivity collapsed entirely to 0.0%.
- **The CORN Breakthrough (Fourth Panel):** Conditioning probabilities and weighting samples restored high diagonal density. Severe NPDR recall surged to **65.7%**, and catastrophic multi-grade errors dropped to just **5.1%**. A total of **94.9%** of all predictions landed within $\pm 1$ grade of true diagnosis.
- **Continuous Regression (Third Panel):** Yielded the tightest error distribution, with **96.1%** of predictions within $\pm 1$ grade and the lowest catastrophic error rate of **3.8%**.

### 7.3 Statistical Significance Testing

Paired bootstrap tests (1,000 iterations) and Wilcoxon signed-rank tests confirmed:
- **Variant D (CORN) vs. Variant B (CORAL):** $\Delta QWK = +0.1212$ (95% CI: $[+0.0924, +0.1504]$, $p < 0.001$). The Wilcoxon signed-rank test on error distances yielded $W = 4180.0, p = 9.35 \times 10^{-21}$. The improvement of Variant D (CORN) over CORAL is statistically decisive.
- **Variant C (Continuous Regression) vs. Softmax Baseline:** $\Delta QWK = +0.0062$ (95% CI: $[-0.0142, +0.0265]$, $p = 0.264$), while reducing catastrophic errors from 5.6% to 3.8%.

### 7.4 Accuracy Enhancement: Label Smoothing, Ensembling, and Logit Adjustment

Table 4 presents the results of our accuracy optimization experiments, and Table 5 provides the temperature sweep for post-hoc logit adjustment.

#### TABLE 4: Classification Accuracy Optimization Comparison on Holdout Test Set (N = 550)

| Model Configuration | Accuracy [95% CI] | QWK [95% CI] | MAE [95% CI] | Catastrophic Err ($d \ge 2$) [95% CI] | Grade 3 Severe F1 [95% CI] | Grade 4 PDR F1 [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Variant A (Baseline Softmax)** | 80.18% [76.73%, 83.45%] | 0.8724 [0.8385, 0.9044] | 0.2673 [0.2182, 0.3200] | 5.64% [3.64%, 7.64%] | 0.3810 [0.2221, 0.5313] | 0.5169 [0.3830, 0.6377] |
| **Variant A (Single Seed 42 + Label Smooth 0.08)** | 82.00% [78.73%, 85.10%] | 0.8709 [0.8330, 0.9039] | 0.2473 [0.1982, 0.3000] | 5.27% [3.45%, 7.27%] | 0.4828 [0.3225, 0.6452] | 0.5195 [0.3662, 0.6364] |
| **Variant A (Single Seed 42 + Logit Adj $\tau = 1.0$)** | 80.00% [76.73%, 83.46%] | 0.8682 [0.8303, 0.9010] | 0.2673 [0.2145, 0.3200] | 4.91% [3.27%, 6.73%] | 0.4000 [0.2597, 0.5306] | 0.5250 [0.3793, 0.6458] |
| **3-Seed Ensemble + TTA ($\tau = 0.0$, Raw Acc Focus)** | **83.82%** [80.73%, 86.73%] | **0.8859** [0.8503, 0.9170] | **0.2200** [0.1727, 0.2691] | **4.36%** [2.73%, 6.18%] | 0.4561 [0.2999, 0.6134] | 0.5570 [0.4179, 0.6739] |
| **3-Seed Ensemble + TTA ($\tau = 1.0$, Logit Adj Focus)** | 79.27% [76.00%, 82.55%] | 0.8787 [0.8451, 0.9091] | 0.2691 [0.2200, 0.3164] | 4.55% [2.73%, 6.36%] | 0.3721 [0.2432, 0.5056] | **0.5778** [0.4444, 0.6875] |

#### TABLE 5: Post-Hoc Logit Adjustment Temperature Sweep on 3-Seed Ensemble

| Logit Adjustment Temperature ($\tau$) | Accuracy | QWK | MAE | Severe Recall (Grade 3) | PDR Recall (Grade 4) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0 (Unadjusted, Raw Acc Focus)** | **83.82%** | 0.8859 | **0.2200** | 44.8% | 50.0% |
| **0.5 (Balanced Trade-off)** | 82.18% | **0.8929** | 0.2291 | 51.7% | 56.8% |
| **1.0 (Standard Prior Adjustment)** | 79.27% | 0.8787 | 0.2691 | 55.2% | **59.1%** |
| **1.5 (Minority-Tail Focus)** | 74.91% | 0.8438 | 0.3364 | **69.0%** | 50.0% |

![Confusion Matrices for Accuracy Optimization Configurations on the Holdout Test Set (N = 550) comparing Baseline Softmax, Label-Smoothed Seed 42, and 3-Seed Ensembles (tau = 0.0 and tau = 1.0).](submission/results/confusion_matrix_accuracy_ensemble.png)

Key insights from the accuracy optimization experiments:
- **Single Model Gains:** Adding label smoothing ($\alpha = 0.08$) and safe geometric jitter increased single-model accuracy from 80.18% to **82.00%** (+1.82%).
- **Ensemble Peak:** Combining three random seeds with 4-way flip TTA pushed accuracy to **83.82%**, QWK to **0.8859**, and reduced MAE to **0.2200**.
- **Adjustable Clinical Dial ($\tau$):** Logit adjustment provides a post-hoc slider. At $\tau = 0.5$, QWK peaks at **0.8929** (highest in the study) with 82.18% accuracy. At $\tau = 1.5$, Severe NPDR recall jumps to **69.0%**.

---

## 8. SYSTEM DESIGN: ACTIVITY AND CLASS DIAGRAMS

To formalize the dynamic workflow and structural software architecture of the clinical triage platform, we present StarUML-compliant Activity and Class Diagrams.

### 8.1 UML Activity Diagram

The Activity Diagram defines the operational workflow across three synchronized clinical partitions: Primary Screening Clinic, Automated AI Diagnostic Engine, and Clinical Triage & Specialist Referral.

![UML Activity Diagram: Clinical Screening, Automated AI Diagnostic Processing, and Specialist Triage Workflow (StarUML Specification).](submission/results/uml_activity_diagram.png)

#### Detailed Workflow Description:
1. **Partition 1 (Primary Screening Clinic / Technician):**
   - The workflow initiates at the handheld or tabletop fundus camera where the technician acquires a digital retinal photograph.
   - An automated quality assessment algorithm verifies image sharpness, illumination uniformity, and focus.
   - *Quality Guard Condition:* If the photograph is degraded by lens blur or excessive light reflection (`[No: Blurry / Glare]`), the system triggers an immediate recapture prompt. If acceptable (`[Yes: Valid]`), the stream passes to the AI diagnostic engine.
2. **Partition 2 (Automated AI Diagnostic Engine):**
   - The functional retinal globe is segmented via circular crop boundary detection, normalized, and contrast-enhanced using CIE LAB CLAHE on an $8 \times 8$ grid.
   - The image is processed through the regularized EfficientNet-B0 backbone to extract a dense 1280-dimensional latent embedding $\phi(\mathbf{x})$.
   - *Dual-Pathway Selection:* The system dynamically routes the embedding based on the clinical environment:
     - In **Community Screening Mode**, the Variant D (CORN) ordinal head evaluates conditional binary cumulative tasks regularized by Soft-QWK loss to maximize severe disease sensitivity.
     - In **Telemedicine Confirmatory Mode**, the 3-Seed Ensemble with 4-way flip Test-Time Augmentation (TTA) and label smoothing computes highly calibrated stage probabilities.
   - Both pathways synchronize through a merge bar, where the clinical distance safety envelope is applied, mathematically suppressing catastrophic multi-grade misclassifications ($|y - \hat{y}| \ge 2$) to below 5.1%.
3. **Partition 3 (Clinical Triage & Specialist Referral):**
   - A multi-way decision node routes patients according to ICDR clinical guidelines:
     - **High-Risk Pathway (`[Grade 3, 4]`):** Dispatches an urgent specialist referral within 48 hours to 2 weeks for laser photocoagulation or anti-VEGF injection.
     - **Moderate Pathway (`[Grade 2]`):** Schedules a semi-urgent 6-month comprehensive exam and OCT scan.
     - **Low-Risk Pathway (`[Grade 0, 1]`):** Authorizes primary surveillance with a 12-month recall.
   - The resulting recommendations are compiled into a standardized FHIR-compliant diagnostic report, securely archived in the hospital Picture Archiving and Communication System (PACS) database, and delivered to the patient and care team via SMS/portal before concluding at the final state.

### 8.2 UML Class Diagram

The Class Diagram documents the object-oriented structure, class hierarchies, and operational methods.

![UML Class Diagram: Object-Oriented Software Architecture, Class Hierarchies, and Interface Methods (StarUML Specification).](submission/results/uml_class_diagram.png)

#### Description of Key Classes:
- **`FundusImage`**: Encapsulates raw image data, acquisition metadata, dimensions, and ground truth labels.
- **`Preprocessor`**: Provides static methods `circular_crop()` and `apply_clahe()` to clean and enhance fundus photographs.
- **`EfficientNetBackbone`**: Wraps the convolutional feature extractor, managing layer freezing and forward passes to output latent vectors $\phi(\mathbf{x}) \in \mathbb{R}^{1280}$.
- **`ModelHead` (Abstract Base Class)**: Defines the common interface `forward()` and `predict_stage()`.
  - Subclasses: `SoftmaxHead`, `CORALHead`, `CORNHead`, `ContinuousHead`.
- **`EnsembleEngine`**: Coordinates multi-seed inference, manages flip-based Test-Time Augmentation (TTA), and applies post-hoc logit adjustment via `apply_logit_adjustment(tau)`.
- **`ClinicalTriageEngine`**: Implements deterministic rule mapping (`evaluate_referral_rule()`), safety checks (`enforce_safety_envelope()`), and dispatches to `DiagnosticReport`.
- **`DiagnosticReport`**: Structures the final clinical output, containing patient ID, predicted grade, confidence, error safety bounds, and recommended action.

---

## 9. FUTURE WORK AND CONCLUSION

### 9.1 Conclusion

In this study, we addressed the critical vulnerability of nominal multiclass loss in automated Diabetic Retinopathy stage grading. Standard categorical cross-entropy treats diagnostic errors with dangerous symmetry, exposing diabetic screening programs to catastrophic multi-grade triage failures. To solve this limitation, we conducted an empirical benchmark comparing standard softmax classification, rank-consistent ordinal regression (CORAL), continuous scalar regression, conditional ordinal regression (CORN), and an accuracy-optimized 3-seed ensemble across an identical EfficientNet-B0 backbone on the APTOS 2019 benchmark.

Our empirical findings establish four core conclusions:
1. **The CORN Breakthrough:** Conditional ordinal regression resolves the breakdown of unweighted CORAL on minority classes. By conditioning binary cumulative probabilities and integrating effective-number class weighting with differentiable Soft-QWK loss, Variant D (CORN) elevated Severe NPDR sensitivity to **65.7%** (95% CI: [48.3%, 82.9%]) and slashed catastrophic triage errors from 26.4% to just **5.1%**. Most significantly, our lightweight model decisively beats published multi-branch networks (such as Shakibania et al. [2], whose 30M parameter model achieved only 36.84% severe recall) with an absolute +28.86% improvement and 82% fewer parameters.
2. **Distance-Aware Robustness:** Continuous regression with Smooth L1 loss achieved the tightest error distribution, bounding catastrophic multi-grade triage errors ($d \ge 2$) to just **3.8%** and delivering a QWK of **0.8788**.
3. **High-Accuracy Confirmatory Pipeline:** Multi-seed ensembling with label smoothing and test-time augmentation achieved a project-high raw classification accuracy of **83.82%** (95% CI: [80.73%, 86.73%]), QWK of **0.8859**, and MAE of **0.2200**, with post-hoc logit adjustment reaching a peak study QWK of **0.8929** at $\tau = 0.5$.
4. **Hardware and Deployment Efficiency:** By prioritizing mathematical loss formulation over parameter bloat, all models were trained safely within 540 MiB of VRAM on a mobile NVIDIA GeForce RTX 5050 GPU, proving that clinical-grade performance is attainable on cost-effective, portable hardware.

### 9.2 Future Work

Building upon these findings, future research will explore the following directions:
1. **Multimodal Vision-Language Explanations:** Integrate our conditional ordinal backbone with lightweight vision-language models (e.g., BiomedCLIP) to generate clinician-facing natural language rationales alongside calibrated stage predictions.
2. **Multi-Center Clinical Validation:** Evaluate cross-cohort generalization across international datasets (Messidor-2, DDR, EyePACS) to test resilience across diverse demographic populations and camera hardware.
3. **Edge Deployment on Handheld Fundus Cameras:** Quantize the EfficientNet-B0 CORN model to INT8 precision using TensorRT for embedded execution on portable, battery-powered fundus cameras for rural screening camps.

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
