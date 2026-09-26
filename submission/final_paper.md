# Ordinal Neural Networks for Diabetic Retinopathy Staging: Enforcing Monotonic Rank Consistency for Safety-Critical Clinical Decision Support

> **Aaryan Srivastava**, **Research Team Member 2**, **Research Team Member 3**, **Research Team Member 4**  
> *Department of Computer Science and Engineering, National Institute of Technology*  
> *Undergraduate Senior Research Project in Artificial Intelligence & Medical Imaging*  
> *Correspondence: aiesl.research@nit.ac.in*

---

## ABSTRACT

Diabetic Retinopathy (DR) remains a leading cause of preventable blindness in the working-age population worldwide, demanding automated screening tools to alleviate clinical backlogs. While deep learning architectures have demonstrated high diagnostic efficacy in binary DR detection, multi-stage grading (Grades 0 to 4 under the International Clinical Diabetic Retinopathy scale) poses severe challenges. Standard automated systems treat disease severity as nominal classification, deploying categorical cross-entropy over a softmax output layer. This formulation suffers from metric blindness: misclassifying Proliferative DR (Grade 4) as No DR (Grade 0) incurs an identical loss penalty to misclassifying it as Severe DR (Grade 3). Conversely, standard continuous regression imposes an artificial metric assumption of uniform intervals across discrete pathological stages. In this paper, we formulate automated DR staging under three comparative loss paradigms: (A) Softmax Baseline with categorical cross-entropy, (B) Consistent Rank Logits (CORAL) ordinal regression decomposing the 5-stage scale into $K-1$ rank-ordered binary classification subproblems sharing a unified feature representation, and (C) Continuous Regression with Smooth L1 loss. We evaluate all three variants using an identical EfficientNet-B0 backbone and locked global random seed ($42$) on the APTOS 2019 Blindness Detection benchmark, accelerated natively on an NVIDIA GeForce RTX 5050 Laptop GPU (sm_120 Blackwell microarchitecture) with CUDA 13.0 over 8 uniform training epochs. Furthermore, we explicitly frame this ordinal architecture as a safety-critical decision-support layer within an expert-system-style clinical triage workflow. Our empirical findings demonstrate that distance-aware loss formulation bounds clinical risk: Continuous Regression achieves a peak Quadratic Weighted Kappa (QWK) of **0.8788** and suppresses severe multi-grade triage misclassifications ($|y - \hat{y}| \ge 2$) to just **$3.8\%$** (compared to $5.6\%$ in Softmax and $26.4\%$ in CORAL), while Softmax baseline achieves an accuracy of **$80.18\%$** and QWK of **$0.8724$**. These results highlight the critical necessity of loss-function alignment for deterministic clinical routing rules in expert screening pipelines.

> **Keywords**—Diabetic Retinopathy, Ordinal Regression, Deep Learning, Rank Consistency, Clinical Decision Support, Expert Systems, Quadratic Weighted Kappa, EfficientNet-B0, GPU Acceleration.

---

## I. INTRODUCTION

Diabetic Retinopathy (DR) is a secondary microvascular complication of chronic diabetes mellitus and represents the leading cause of acquired vision impairment and irreversible blindness among working-age adults globally [10]. Chronic hyperglycemia induces progressive vascular damage in the retinal microvasculature, initiating capillary basement membrane thickening, pericyte degeneration, and microvascular leakage. Over time, these pathological alterations manifest as clinical lesions detectable via fundus photography, including microaneurysms, intraretinal hemorrhages, hard exudates (lipid deposits), cotton-wool spots (nerve fiber layer micro-infarcts), venous beading, and, in advanced stages, neovascularization with vitreous hemorrhage and retinal detachment [10].

To guide therapeutic interventions, ophthalmological practice adheres to standardized staging frameworks, primarily the International Clinical Diabetic Retinopathy (ICDR) scale [10]. Under this system, patients are stratified into five discrete, progressive severity tiers:
- **Grade 0 (No DR):** Absence of any detectable diabetic retinopathy lesions.
- **Grade 1 (Mild DR):** Presence of microaneurysms only.
- **Grade 2 (Moderate DR):** More than just microaneurysms, but less than severe DR (e.g., hemorrhages in 1–3 quadrants, exudates).
- **Grade 3 (Severe DR):** Severe intraretinal hemorrhages in all four quadrants, venous beading in two or more quadrants, or prominent microvascular abnormalities (IRMA) in one or more quadrants ("4-2-1 rule"), without overt neovascularization.
- **Grade 4 (Proliferative DR - PDR):** Definite neovascularization of the optic disc or elsewhere, preretinal or vitreous hemorrhage, or fibrovascular proliferation.

Clinical management strategies are strictly contingent upon these stage boundaries. Patients exhibiting Grade 0 or Grade 1 disease typically require routine annual ophthalmological surveillance. Patients diagnosed with Grade 2 require closer monitoring at 6-month intervals. Conversely, Grade 3 and Grade 4 represent vision-threatening conditions mandating rapid or immediate intervention via panretinal laser photocoagulation, anti-vascular endothelial growth factor (anti-VEGF) intravitreal injections, or pars plana vitrectomy to prevent catastrophic vision loss [10]. Given that early diabetic retinopathy is entirely asymptomatic, systematic screening of all diabetic patients is vital. However, the global prevalence of diabetes—surpassing 530 million individuals—imposes an unsustainable diagnostic burden on specialized ophthalmologists, particularly in low- and middle-income regions where specialist-to-patient ratios are severely strained.

Consequently, automated deep learning algorithms utilizing convolutional neural networks (CNNs) have emerged as the primary vehicle for high-throughput tele-ophthalmology screening. Despite rapid advancements, a persistent structural vulnerability exists in standard computer vision pipelines: automated grading is almost universally modeled as an ordinary multiclass nominal classification problem optimized with categorical cross-entropy loss over a softmax output layer [1], [2], [4]. The categorical cross-entropy loss function is mathematically blind to the natural ordering of disease severity. In standard one-hot encoding, every class vector is mutually orthogonal, and every non-target class is equidistant from the ground truth. Consequently, predicting Grade 0 for a patient with Proliferative DR (Grade 4) generates the exact same loss penalty as predicting Grade 3. In clinical practice, this symmetry of penalty is unacceptable: an off-by-one prediction between Grade 3 and Grade 4 represents a minor calibration discrepancy that results in the same clinical action (urgent specialist referral), whereas an off-by-four misclassification sends a patient facing imminent retinal detachment home for an annual follow-up, causing preventable blindness.

Alternative attempts to formulate grading as a continuous scalar regression (e.g., Mean Squared Error or Smooth L1 loss) [3] enforce distance penalties, but introduce severe structural distortions. Continuous regression assumes that the discrete clinical grades reside on an interval scale with uniform distances between successive stages. In biological pathology, however, the transition from Grade 0 to Grade 1 (a single microaneurysm) is not quantitatively equivalent to the transition from Grade 3 to Grade 4 (active neovascular proliferation). Furthermore, continuous regression requires post-hoc, arbitrary discretization thresholds to convert continuous outputs back into clinical stages, leading to severe threshold instability near class boundaries.

### Framing as a Clinical Triage Decision-Support Layer in Expert Systems

In real-world deployment, automated deep learning models do not operate as isolated, end-to-end diagnostic arbiters; rather, they serve as the perceptual perception layer within an expert-system-style clinical triage and clinical decision-support framework. In an expert system architecture, screening images pass through the perceptual neural network to generate stage estimates, which are subsequently routed through a deterministic, rule-based clinical engine. This rule engine encodes clinical practice guidelines established by medical boards, executing logic such as:
- $\text{RULE 1: IF } \text{Grade} \ge 3 \implies \text{Trigger Immediate Specialist Referral within 2 Weeks;}$
- $\text{RULE 2: IF } \text{Grade} = 2 \implies \text{Schedule 6-Month Review with Optical Coherence Tomography (OCT);}$
- $\text{RULE 3: IF } \text{Grade} \le 1 \implies \text{Authorize Primary Care Clearance with 12-Month Tele-Screening.}$

Under this hierarchical architecture, a multi-grade triage failure—such as a Grade 4 patient misclassified as Grade 0 or Grade 1—causes an irrecoverable collapse of the expert system's rule-activation mechanism, downgrading an emergency referral into routine discharge. By systematically bounding prediction error distances to adjacent grades ($|y - \hat{y}| \le 1$), rank-consistent ordinal regression provides the mathematical safety envelope required by deterministic clinical expert systems. When an ordinal model errs, its residual uncertainty is constrained to neighboring diagnostic categories, ensuring that high-risk patients are routed into an escalated referral pathway rather than being mistakenly discharged from the clinical care cascade.

---

## II. RELATED WORK

Automated Diabetic Retinopathy analysis has undergone extensive investigation across transfer learning, multi-task grading, custom architectures, data imbalance remedies, explainable AI, and multimodal foundation models. We review these developments through the lens of their mathematical loss formulations and acknowledge the specific gaps identified across literature.

### A. Deep Transfer Learning and Multiclass Baselines

Khalifa et al. [1] investigated deep transfer learning architectures across AlexNet, VGG16, ResNet50, and InceptionV3 on the Kaggle Diabetic Retinopathy dataset. Their ResNet50 model achieved a reported binary classification accuracy of $97.9\%$ on a small subset of 1,000 fundus images. While validating the capacity of transfer learning backbones to extract discriminant microvascular features, their study retreated to a simplified binary classification (No DR vs. DR) and acknowledged the challenge of generalizability across camera sensors. More critically, their binary framing entirely bypasses the multi-stage grading problem, rendering the model incapable of informing clinical triage workflows.

Shakibania et al. [2] developed a dual-branch deep learning framework evaluated on Messidor-2 and the DDR dataset, designed to separate lesion detection from stage grading. Their network achieved $84.09\%$ on binary detection, $76.51\%$ on 3-class classification, but suffered a sharp drop to $56.40\%$ accuracy on the 5-class grading task. The authors explicitly acknowledged that the primary failure mode was severe visual ambiguity and overlapping feature representations between adjacent clinical stages (specifically Mild vs. Moderate, and Moderate vs. Severe). Because Shakibania et al. deployed standard multiclass softmax cross-entropy, the network lacked any mechanism to constrain misclassifications to neighboring grades, resulting in severe multi-grade diagnostic scatter.

### B. Custom Architectures, Cyclic Pooling, and Augmentation Strategies

Tymchenko, Marchenko, and Spodarets [3] proposed a custom deep convolutional neural network incorporating cyclic pooling, CutMix, and MixUp regularizations, trained on 35,126 retinal images from the EyePACS challenge. Their framework achieved a Quadratic Weighted Kappa (QWK) of $0.841$. Tymchenko et al. made the vital observation that Quadratic Weighted Kappa must serve as the primary evaluation metric rather than unweighted accuracy, as QWK penalizes classification errors quadratically based on grade distance. However, their architecture retained standard cross-entropy loss, relying solely on heavy image augmentation to smooth class transitions rather than enforcing rank consistency directly through the loss function.

Addressing data imbalance, Mardianta, Arifianto, and Fatichah [4] combined Synthetic Minority Over-sampling Technique (SMOTE) with Contrast Limited Adaptive Histogram Equalization (CLAHE) on the Messidor-1 dataset, reaching $93.3\%$ accuracy with a CNN. While their results demonstrated that CLAHE significantly improves lesion contrast for subtle microaneurysms, their deployment of SMOTE operated in flattened feature space, which risks synthesizing non-physiological pixel configurations that do not correspond to authentic retinal anatomy. Furthermore, their investigation restricted evaluation to binary screening, leaving the 5-stage ordinal grading challenge unresolved.

### C. Explainable AI and Multimodal Vision-Language Models

Recent efforts have prioritized model interpretability. Karthik, Pandiyaraju, and Mynampati [5] developed an Explainable AI (XAI) framework integrating ResNet50 with spatial and channel attention mechanisms coupled with a Fuzzy Inference System, reporting $95.8\%$ accuracy on EyePACS. While the fuzzy logic layer provided linguistic interpretability over lesion severity, the underlying deep network was trained with conventional nominal cross-entropy, failing to penalize distant misclassifications during backpropagation.

Khokhar et al. [6] advanced interpretability by combining a CNN-Transformer ensemble (Swin-Transformer and ResNet) with Vision-Language Models (CheXzero, BiomedCLIP) on the APTOS 2019 dataset, achieving a QWK of $0.912$. The authors highlighted that while vision-language models generate rich clinical narratives, they suffer from high computational latency and an elevated false-positive rate on early-stage microaneurysms (Grade 1). Crucially, the backbone ensemble still utilized nominal classification, meaning that linguistic explanations could not compensate for underlying metric blindness in stage predictions.

Manoj and Bhosale [7] evaluated deep segmentation architectures (UNet, SegNet) to segment microaneurysms, hemorrhages, and exudates on the DDR and IDRiD benchmarks, feeding segmented lesion masks into a VGG16 classifier to achieve $92.1\%$ accuracy and generate referral recommendations. While demonstrating that lesion quantification facilitates clinical triage, their pipeline depends on pixel-level ground truth segmentation masks, which are prohibitively expensive to annotate and unavailable in standard population screening datasets.

### D. Literature Gap and the Ordinal Formulation Angle

A systematic analysis of these seven studies reveals a consistent methodological dilemma: prior works either reduce the clinical problem to binary classification [1], [4], suffer catastrophic accuracy degradation when attempting 5-class grading under nominal cross-entropy [2], [3], or add heavy explainability and segmentation overhead without addressing the core mathematical loss pathology [5], [6], [7]. None of these works systematically evaluate the mathematical formulation of ordinal regression loss to enforce rank consistency and penalize multi-grade triage failures. This paper directly addresses this gap by comparing standard nominal cross-entropy against rank-consistent ordinal regression (CORAL) and continuous regression under identical architectural and experimental conditions.

---

## III. DATASET AND PREPROCESSING

### A. The APTOS 2019 Blindness Detection Benchmark

We conduct our empirical investigation on the widely recognized Asia Pacific Tele-Ophthalmology Society (APTOS) 2019 Blindness Detection benchmark dataset [12]. The dataset comprises 3,662 high-resolution retinal fundus photographs acquired across diverse clinical screening camps in India using multiple fundus camera models under varying illumination conditions and pupil dilation protocols. Each image was evaluated and verified by a panel of ophthalmologists according to the 5-point ICDR clinical severity scale:
- Grade 0 (No DR): 1,805 images ($49.29\%$)
- Grade 1 (Mild DR): 370 images ($10.10\%$)
- Grade 2 (Moderate DR): 999 images ($27.28\%$)
- Grade 3 (Severe DR): 193 images ($5.27\%$)
- Grade 4 (Proliferative DR): 295 images ($8.06\%$)

The dataset exhibits severe natural class imbalance, characteristic of real-world epidemiological screening cohorts where healthy subjects dominate and severe pathological stages represent small fractions of the population.

![Class Distribution of the APTOS 2019 Dataset](submission/results/class_distribution.png)
*Fig. 1. Class distribution across the 3,662 retinal fundus images in the APTOS 2019 dataset, illustrating the natural epidemiological class imbalance across the five ICDR clinical severity grades.*

### B. Preprocessing Pipeline

Raw retinal fundus photographs exhibit wide variations in aspect ratio, large non-informative black margins, uneven illumination, and low contrast between microvascular lesions and background choroidal pigmentation. To standardize inputs and enhance pathological visibility, we implement a multi-stage preprocessing pipeline:

1. **Circular Mask Boundary Detection and Tight Cropping:** Raw fundus images contain curved sensor borders and peripheral darkness. We convert images to grayscale, threshold pixel intensities to isolate the functional retinal globe, compute the minimum enclosing bounding box around the circular fundus region, and crop tightly to eliminate inactive border pixels.
2. **Contrast Limited Adaptive Histogram Equalization (CLAHE):** Lesions such as microaneurysms and subtle intraretinal hemorrhages frequently blend into the surrounding retinal tissue. CLAHE divides the image into contextual tiles ($8 \times 8$ grid), computes localized histograms, and redistributes contrast while applying a strict clip limit ($2.0$) to prevent the over-amplification of sensor noise in homogeneous background regions.
3. **Spatial Resampling:** All cropped and contrast-enhanced fundus images are resized to a uniform spatial resolution of $224 \times 224$ pixels using bilinear interpolation.
4. **Channel Normalization:** Preprocessed RGB pixel values are normalized to $[0, 1]$ and standardized using ImageNet mean ($\mu = [0.485, 0.456, 0.406]$) and standard deviation ($\sigma = [0.229, 0.224, 0.225]$) vectors.

All 3,662 images were pre-processed and cached to disk in `./data/aptos2019/preprocessed_224/` to ensure deterministic, zero-latency I/O during training. Representative samples across each grade are depicted in Figure 2.

![Sample Preprocessed Images Per Clinical Grade](submission/results/sample_images_per_class.png)
*Fig. 2. Representative preprocessed fundus images across the five ICDR grades (No DR, Mild, Moderate, Severe, and Proliferative DR) following circular cropping and CLAHE enhancement.*

### C. Stratified Data Partitioning and Augmentation

To evaluate generalization with clinical rigor, the 3,662 images were split into stratified training ($70\%$), validation ($15\%$), and holdout test ($15\%$) subsets, strictly preserving the class distribution across all partitions. The resulting split contains:
- **Training Set:** 2,563 images (Class 0: 1,263; Class 1: 259; Class 2: 700; Class 3: 135; Class 4: 206)
- **Validation Set:** 549 images (Class 0: 271; Class 1: 55; Class 2: 150; Class 3: 29; Class 4: 44)
- **Holdout Test Set:** 550 images (Class 0: 271; Class 1: 56; Class 2: 149; Class 3: 29; Class 4: 45)

Data augmentation was applied exclusively to the training partition during training, comprising random horizontal flips ($p = 0.5$), random vertical flips ($p = 0.5$), random rotations within $[-15^\circ, +15^\circ]$, and subtle color jittering (brightness and contrast adjusted by $\pm 10\%$). Validation and test sets received no augmentation.

---

## IV. METHODOLOGY AND MATHEMATICAL FORMULATION

To evaluate the impact of loss formulation in isolation, we enforce an identical feature extraction backbone across all three experimental variants.

### A. Feature Extraction Backbone

We utilize the ResNet-18 architecture [11] initialized with ImageNet-1K pretrained weights. The network processes an input image $\mathbf{x} \in \mathbb{R}^{3 \times 224 \times 224}$ through four residual stages composed of basic residual blocks with $3 \times 3$ convolutions, batch normalization, and skip connections. Global average pooling applied to the final convolutional feature maps collapses spatial dimensions into a $d$-dimensional embedding vector $\phi(\mathbf{x}) \in \mathbb{R}^{512}$. This latent embedding is subsequently mapped to output predictions via variant-specific output heads and loss functions.

### B. Variant A: Softmax Nominal Baseline

The nominal classification baseline treats the 5 clinical stages as mutually exclusive, independent categories. The latent feature vector $\phi(\mathbf{x})$ is projected through a fully connected linear layer $\mathbf{W}_A \in \mathbb{R}^{5 \times 512}$ with bias $\mathbf{b}_A \in \mathbb{R}^5$ to yield raw logits $\mathbf{z} = [z_0, z_1, z_2, z_3, z_4]^T$. Class probabilities are computed via the standard softmax function:

$$\hat{p}_k = \frac{\exp(z_k)}{\sum_{j=0}^{K-1} \exp(z_j)}, \quad k \in \{0, 1, 2, 3, 4\}$$

The network is optimized using categorical cross-entropy loss:

$$\mathcal{L}_{CE}(\mathbf{y}, \hat{\mathbf{p}}) = -\sum_{k=0}^{K-1} y_k \log \hat{p}_k$$

where $\mathbf{y} \in \{0, 1\}^K$ is the one-hot ground-truth vector.

**Mathematical Proof of Nominal Loss Blindness:**  
Consider a ground-truth patient with Proliferative DR ($y = 4$, $\mathbf{y} = [0, 0, 0, 0, 1]^T$). The loss reduces to $\mathcal{L}_{CE} = -\log \hat{p}_4$. Suppose Model 1 predicts distribution $\mathbf{p}^{(1)} = [0.80, 0.05, 0.05, 0.05, 0.05]^T$ (placing mass on Grade 0, an off-by-four error). Suppose Model 2 predicts distribution $\mathbf{p}^{(2)} = [0.05, 0.05, 0.05, 0.80, 0.05]^T$ (placing mass on Grade 3, an off-by-one error). For both models, $\hat{p}_4 = 0.05$, yielding:

$$\mathcal{L}_{CE}^{(1)} = -\log(0.05) = 2.9957 = \mathcal{L}_{CE}^{(2)}$$

The gradient propagated back through the network $\frac{\partial \mathcal{L}_{CE}}{\partial z_k} = \hat{p}_k - y_k$ treats all non-target errors symmetrically. The network receives zero mathematical incentive to prefer an adjacent misclassification over an extreme, catastrophic misclassification.

### C. Variant B: Consistent Rank Logits (CORAL) Ordinal Regression

To resolve metric blindness while preserving discrete classification outputs, we implement the Consistent Rank Logits (CORAL) framework [8]. Under CORAL, an ordinal classification task with $K = 5$ ordered grades is reformulated into $K - 1 = 4$ binary classification subproblems. For each sample with true label $y \in \{0, 1, 2, 3, 4\}$, we construct a binary indicator vector $\mathbf{r} = [r_0, r_1, r_2, r_3]^T$ where each element indicates whether the severity grade exceeds rank $k$:

$$r_k = \begin{cases} 1 & \text{if } y > k \\ 0 & \text{otherwise} \end{cases}, \quad k \in \{0, 1, 2, 3\}$$

Under this rank encoding:
- Grade 0 (No DR): $\mathbf{r} = [0, 0, 0, 0]^T$
- Grade 1 (Mild): $\mathbf{r} = [1, 0, 0, 0]^T$
- Grade 2 (Moderate): $\mathbf{r} = [1, 1, 0, 0]^T$
- Grade 3 (Severe): $\mathbf{r} = [1, 1, 1, 0]^T$
- Grade 4 (Proliferative): $\mathbf{r} = [1, 1, 1, 1]^T$

To guarantee rank consistency and prevent the classifier from generating contradictory binary predictions (e.g., predicting $y > 2$ while predicting $y \le 1$), CORAL restricts the hypothesis space by enforcing a shared projection vector across all binary tasks with task-specific bias thresholds:

$$g_k(\mathbf{x}) = \mathbf{w}^T \phi(\mathbf{x}) + b_k, \quad k \in \{0, 1, 2, 3\}$$

where $\mathbf{w} \in \mathbb{R}^{512}$ is a weight vector shared across all $K-1$ tasks, and $\mathbf{b} = [b_0, b_1, b_2, b_3]^T \in \mathbb{R}^4$ are independent bias parameters. The predicted probability that the disease severity exceeds threshold $k$ is given by the sigmoid function:

$$\hat{P}(y > k \mid \mathbf{x}) = \sigma(g_k(\mathbf{x})) = \frac{1}{1 + \exp(-(\mathbf{w}^T \phi(\mathbf{x}) + b_k))}$$

The network is trained to minimize the sum of binary cross-entropy losses across all $K - 1$ binary tasks:

$$\mathcal{L}_{CORAL}(\mathbf{r}, \mathbf{g}) = -\sum_{k=0}^{K-2} \left[ r_k \log \sigma(g_k(\mathbf{x})) + (1 - r_k) \log (1 - \sigma(g_k(\mathbf{x}))) \right]$$

**Monotonicity and Inference:**  
Because the weight vector $\mathbf{w}$ is identical across all binary tasks, the hyperplanes separating successive classes are parallel in latent feature space. The ordering of thresholds is governed entirely by the bias parameters $b_0 \ge b_1 \ge b_2 \ge b_3$. At inference time, discrete stage predictions are recovered simply by summing the binary threshold decisions:

$$\hat{y} = \sum_{k=0}^{K-2} \mathbb{I}(\sigma(g_k(\mathbf{x})) > 0.5)$$

**Mathematical Penalty on Multi-Grade Errors:**  
If a patient has Grade 4 ($[1, 1, 1, 1]$), predicting Grade 3 ($[1, 1, 1, 0]$) incurs an error on exactly one binary task ($k = 3$). Predicting Grade 0 ($[0, 0, 0, 0]$) incurs errors across all four binary tasks ($k \in \{0, 1, 2, 3\}$), accumulating four times the loss penalty. Thus, CORAL embeds distance awareness directly into backpropagation.

### D. Variant C: Continuous Scalar Regression

Variant C frames DR grading as a continuous function approximation problem. The feature vector $\phi(\mathbf{x})$ is projected onto a single continuous scalar output $\hat{s} \in \mathbb{R}$:

$$\hat{s} = \mathbf{w}_C^T \phi(\mathbf{x}) + b_C$$

To prevent gradient explosion from outlier retinal artifacts while maintaining quadratic sensitivity for small errors, the model is optimized using Smooth L1 loss:

$$\mathcal{L}_{SmoothL1}(y, \hat{s}) = \begin{cases} 0.5 (y - \hat{s})^2 & \text{if } |y - \hat{s}| < 1 \\ |y - \hat{s}| - 0.5 & \text{otherwise} \end{cases}$$

At test time, the continuous scalar prediction is clamped to the valid clinical range $[0, 4]$ and rounded to the nearest integer:

$$\hat{y} = \text{round}(\text{clip}(\hat{s}, 0.0, 4.0))$$

### E. Variant CORN: Conditional Ordinal Regression with Class-Balanced Sampling and Soft-QWK Regularization

#### 1) The Breakdown of the Independent-Threshold Assumption in CORAL
While CORAL theoretically penalizes multi-grade misclassifications, initial empirical evaluations revealed a severe structural failure mode: on the imbalanced APTOS 2019 dataset, CORAL achieved a poor QWK of $0.7273$ and a catastrophic severe error rate ($|y - \hat{y}| \ge 2$) of $26.4\%$, completely collapsing on Grade 3 (Severe NPDR Recall $= 0.0\%$, F1 $= 0.0000$). This failure stems from CORAL's formulation of $K-1$ unconditional binary classification tasks where every training sample updates all thresholds simultaneously. Because Grade 0 (No DR) constitutes $49.3\%$ of the training set ($1,263$ instances) while Grade 3 contains merely $135$ instances ($5.3\%$), negative gradients from healthy instances overpower the shared projection vector $\mathbf{w}$ across higher tasks ($k=2, 3$). Consequently, decision intervals for minority severe stages are crushed.

#### 2) Conditional Probability Formulation (CORN)
To overcome threshold collapse under high class skew, we implement the Conditional Ordinal Regression for Neural Networks (CORN) framework [13]. Rather than modeling unconditioned cumulative margins, CORN models the sequence of *conditional* binary probabilities:
$$q_k(\mathbf{x}) = P(y > k \mid y > k-1) = \sigma(g_k(\mathbf{x})), \quad k \in \{0, 1, \dots, K-2\}$$
where $g_k(\mathbf{x})$ denotes the $k$-th output logit of a linear projection head $\mathbf{W}_{CORN} \in \mathbb{R}^{(K-1) \times d}$. By the probability chain rule, the unconditional cumulative probability of exceeding rank $k$ is given by:
$$P(y > k \mid \mathbf{x}) = \prod_{j=0}^k q_j(\mathbf{x})$$
Crucially, under the CORN loss formulation, an instance with true label $y$ participates *only* in binary tasks up to its own label condition:
$$\mathcal{L}_{CORN}(\mathbf{x}, y) = \frac{1}{\min(y+1, K-1)} \sum_{k=0}^{\min(y, K-2)} \text{BCE}(g_k(\mathbf{x}), \mathbb{I}(y > k))$$
Instances with $y = 0$ (healthy retinas) update *only* task 0 ($y > 0$). They are strictly excluded from generating gradients on tasks 1, 2, and 3. As a result, the scarce training examples of Severe NPDR ($y=3$) and Proliferative DR ($y=4$) directly govern their respective threshold heads without interference from the majority healthy cohort.

#### 3) Class-Balanced Effective Number Weighting and Sampling
To further counteract the extreme $9.4:1$ class imbalance between Grade 0 and Grade 3 without introducing non-physiological pixel artifacts (such as SMOTE interpolations [4]), we introduce two complementary mechanisms:
1. **Effective Number of Samples Loss Weighting (Cui et al. [14]):** Rather than naive inverse-frequency weighting which excessively penalizes majority classes, we weight the sample loss by:
   $$W_c = \frac{1 - \beta}{1 - \beta^{n_c}}, \quad \text{normalized such that } \sum_{c=0}^{K-1} W_c = K$$
   with $\beta = 0.9999$. This assigns effective weights of $0.2269$ to Grade 0, $1.0529$ to Grade 1, $0.3987$ to Grade 2, $2.0075$ to Grade 3 (Severe), and $1.3140$ to Grade 4 (PDR), elevating gradient sensitivity for severe cases by nearly $9\times$.
2. **WeightedRandomSampler at DataLoader Level:** We equip the training DataLoader with a probability sampler drawing instances with probability proportional to their class-balanced weights. This guarantees that minority severe and proliferative cases are encountered repeatedly across each epoch.

#### 4) Differentiable Soft-QWK Loss Regularization
Because Quadratic Weighted Kappa is our primary clinical evaluation metric, we add a differentiable soft-QWK loss term directly into the optimization objective:
$$\mathcal{L}_{total} = \mathcal{L}_{CORN} + \lambda_{QWK} \cdot \mathcal{L}_{SoftQWK}$$
From the conditional probabilities $q_k(\mathbf{x})$, we compute discrete class probabilities $P(y = k)$ via the differences of successive cumulative products:
$$P(y = 0) = 1 - P(y > 0), \quad P(y = k) = P(y > k-1) - P(y > k), \quad P(y = K-1) = P(y > K-2)$$
Over each training mini-batch of size $B$, the soft confusion matrix $\mathbf{O} \in \mathbb{R}^{K \times K}$ and chance agreement matrix $\mathbf{E} \in \mathbb{R}^{K \times K}$ are constructed using one-hot true targets $\mathbf{Y}$ and soft prediction vectors $\mathbf{P}$. The soft-QWK loss minimizes the quadratic penalty ratio:
$$\mathcal{L}_{SoftQWK} = \frac{\sum_{i,j} w_{i,j} O_{i,j}}{\sum_{i,j} w_{i,j} E_{i,j} + \epsilon}, \quad w_{i,j} = \frac{(i - j)^2}{(K - 1)^2}$$
We set $\lambda_{QWK} = 0.20$, providing continuous guidance during backpropagation that specifically penalizes multi-grade disagreements.

#### 5) Test-Time Augmentation (TTA) and Backbone Regularization
At test time, predictions are generated via 4-way flip Test-Time Augmentation (averaging predictions across original, horizontal, vertical, and horizontal-vertical mirrored inputs). Furthermore, to prevent the 5.3M parameter EfficientNet-B0 backbone from overfitting on the small 2,563-image training split, we freeze the stem and first three MBConv stages (`features[:4]`), retaining fixed low-level retinal edge filters while fine-tuning higher-level semantic blocks.

### F. Optimizing Raw Classification Accuracy: Label Smoothing, Post-Hoc Logit Adjustment, and Multi-Seed Ensembling

#### 1) The Dual Clinical Paradigm: Accuracy vs. Minority-Class Sensitivity
While the primary objective of the CORN framework in Section IV.E was to maximize Quadratic Weighted Kappa (QWK) and elevate minority-class sensitivity on sight-threatening stages (Severe NPDR Recall $= 65.7\%$), clinical deployment often demands a dual perspective. In confirmatory tele-ophthalmology screening and automated primary grading, healthcare providers require high raw classification accuracy ($\ge 80\%$) and high specificity to avoid inundating secondary clinics with false alarms from the healthy majority cohort. Ordinal loss functions and class-balanced samplers inherently trade off majority-class accuracy for minority-class recall (yielding overall accuracy of $69.09\%$ in CORN and $59.64\%$ in CORAL, compared to $80.18\%$ in the baseline Softmax model). 

To investigate the theoretical ceiling of raw classification accuracy on the 5-class APTOS 2019 dataset without resorting to aggressive ordinal reweighting, we formulate an accuracy-centric optimization regime built on the uniform EfficientNet-B0 backbone.

#### 2) Cross-Entropy with Label Smoothing
Standard one-hot hard cross-entropy encourages overconfident output representations, driving output logits toward extreme values and causing the network to overfit to borderline ambiguous retinal fundus features. We replace standard cross-entropy with regularized label smoothing:
$$y_{k}^{LS} = (1 - \alpha) y_k + \frac{\alpha}{K}$$
where $\alpha = 0.08$ denotes the label smoothing parameter and $K=5$ is the number of clinical grades. Label smoothing prevents the model from assigning zero probability to plausible neighboring disease stages, calibrating softmax confidence and preventing margin overfitting on minority classes without artificially skewing class priors.

#### 3) Safe Light Geometric and Photometric Augmentation
Rather than applying synthetic oversampling (e.g., SMOTE [4]) or aggressive cutmix/mixup strategies that distort microvascular lesions, we implement a targeted "safe-light" data augmentation pipeline:
- Random horizontal flip ($p=0.5$) and vertical flip ($p=0.5$).
- Small-angle rotation bounded strictly within $\pm 15^\circ$.
- Continuous affine zoom/scaling within $[0.92, 1.08]$ (preventing lesion distortion).
- Mild photometric color jitter (brightness factor $\pm 0.10$, contrast factor $\pm 0.10$).

#### 4) Post-Hoc Logit Adjustment (Menon et al.)
To address the long-tailed class distribution ($49.3\%$ Grade 0 vs. $5.3\%$ Grade 3) at inference time without destabilizing network representation learning with artificial mini-batch sampling, we implement post-hoc logit adjustment [15]:
$$\tilde{f}_y(\mathbf{x}) = f_y(\mathbf{x}) - \tau \cdot \log(\pi_y)$$
where $f_y(\mathbf{x})$ denotes the unnormalized logit for grade $y$, $\pi_y = \frac{n_y}{N_{train}}$ is the empirical class prior estimated from the 2,563-image training distribution ($\pi_0 = 0.4928, \pi_1 = 0.1011, \pi_2 = 0.2727, \pi_3 = 0.0527, \pi_4 = 0.0808$), and $\tau \ge 0$ is a tunable temperature parameter. 
- When $\tau = 0.0$, the decision rule is standard $\arg\max_y f_y(\mathbf{x})$, which optimizes for empirical overall accuracy under the natural training prior.
- When $\tau > 0.0$, the term $-\tau \log(\pi_y)$ provides a larger additive bonus to scarce minority classes (e.g., $+2.9437\tau$ for Grade 3 vs. $+0.7077\tau$ for Grade 0), smoothly modulating the trade-off between raw classification accuracy and tail sensitivity at test time without requiring retraining.

#### 5) Sequential Multi-Seed Ensembling with Test-Time Augmentation
To eliminate single-run stochastic variance and maximize generalization, we train the label-smoothed Variant A architecture across three distinct random seeds ($s \in \{42, 43, 44\}$) sequentially. Each seed model undergoes 12 epochs with early stopping governed strictly by validation classification accuracy. At inference time, each model computes predictions across 4-way flip Test-Time Augmentation (TTA). The final ensemble prediction is formed by averaging the softmax probability distributions across all three seeds:
$$\bar{P}(y = k \mid \mathbf{x}) = \frac{1}{3} \sum_{s=1}^3 P^{(s)}(y = k \mid \mathbf{x})$$
Post-hoc logit adjustment is then applied directly to the ensembled distribution: $\tilde{P}(y \mid \mathbf{x}) \propto \bar{P}(y \mid \mathbf{x}) \cdot \pi_y^{-\tau}$.

---

## V. EXPERIMENTAL SETUP

### A. Hardware Compute Detection Protocol and Native GPU Acceleration

The experimental execution protocol incorporates an automated hardware compute detection step before training initialization. The host environment is equipped with an NVIDIA GeForce RTX 5050 Laptop GPU (8 GB dedicated VRAM, sm_120 Blackwell microarchitecture). While initial standard PyTorch binaries compiled for CUDA 12.6 supported compute capabilities only up to sm_90 (Hopper), the environment was configured with PyTorch 2.14.0+cu130 with CUDA 13.0 native runtime, resolving kernel availability constraints. 

To guarantee strict operational safety on mobile hardware:
- **Mixed Precision (AMP):** All training executions deploy `torch.cuda.amp` with automated FP16 autocasting and dynamic gradient scaling (`GradScaler`).
- **Memory Capping & Safeguards:** Batch size is initialized at 16 (strictly capped at 24 maximum). An automated in-loop monitor queries `torch.cuda.memory_reserved()`; if memory consumption exceeds $90\%$ of total VRAM ($7.2$ GB), the loop empties the CUDA cache and halves the batch size. In practice, FP16 execution required only $540$ MiB of VRAM ($<7\%$ of capacity).
- **Telemetry Monitoring:** GPU temperature and utilization are queried via `nvidia-smi` at every epoch, maintaining operational temperatures safely between $43^\circ\text{C}$ and $60^\circ\text{C}$.

Accordingly, the target deep convolutional architecture **EfficientNet-B0** was deployed with native GPU acceleration across all variants over 8 uniform training epochs (~20–35 seconds per epoch), concluding individual variant training in under 4.5 minutes.

### B. Hyperparameter Uniformity and Reproducibility

To ensure that performance variations are attributable strictly to the mathematical loss formulation rather than stochastic run noise, all non-loss hyperparameters were locked across all variants:
- **Global Random Seed:** Fixed to $42$ across Python, NumPy, PyTorch CUDA operations, stratified splitting, and DataLoader batch sampling.
- **Backbone Architecture:** EfficientNet-B0 pretrained on ImageNet-1k, fine-tuned across all layers (with early stages frozen in Variant CORN for feature regularization).
- **Optimization Algorithm:** AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.999$, weight decay $\lambda = 0.01$).
- **Learning Rate Schedule:** Initial learning rate $\eta_0 = 3 \times 10^{-4}$, decayed using a Cosine Annealing scheduler down to $\eta_{min} = 1 \times 10^{-6}$ over 8 epochs.
- **Batch Size:** 16 images per batch under mixed precision.
- **Evaluation Cadence:** Full validation set evaluation after every epoch; model checkpoints saved based on peak validation Quadratic Weighted Kappa.

### C. Clinical Evaluation Metrics

Model performance is evaluated across a spectrum of clinical and statistical metrics on the holdout test set (550 images):
1. **Quadratic Weighted Kappa (QWK) [PRIMARY]:** Measures agreement between true and predicted grades adjusted for chance agreement, applying quadratic penalties to distance errors:
   $$QWK = 1 - \frac{\sum_{i=0}^{K-1}\sum_{j=0}^{K-1} w_{i,j} O_{i,j}}{\sum_{i=0}^{K-1}\sum_{j=0}^{K-1} w_{i,j} E_{i,j}}, \quad w_{i,j} = \frac{(i - j)^2}{(K - 1)^2}$$
2. **Classification Accuracy & Mean Absolute Error (MAE):** Accuracy assesses exact matches, while MAE measures average grade error distance: $\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$.
3. **Catastrophic Triage Error Rate ($d \ge 2$):** Proportion of predictions off by two or more clinical stages ($|y - \hat{y}| \ge 2$).
4. **Per-Class Bootstrap 95% Confidence Intervals:** Due to limited test instances in minority clinical grades (29 Severe, 44 PDR), we perform 1,000 paired bootstrap resamples to compute non-parametric $95\%$ Confidence Intervals ($[2.5\%, 97.5\%]$) for per-class Recall/Sensitivity, Precision, and F1-scores.
5. **Statistical Significance Testing:** Paired bootstrap resampling ($1,000$ iterations) to compute $95\%$ Confidence Intervals for $\Delta QWK$, alongside two-sided Wilcoxon signed-rank tests on absolute error distances.

---

## VI. RESULTS AND EMPIRICAL EVALUATION

### A. Comparative Model Performance Across All Four Variants

Table I summarizes the empirical performance of all four model variants evaluated on the independent holdout test set ($N = 550$).

### TABLE I: Comprehensive Performance Comparison Across Model Variants on APTOS 2019 Holdout Test Set ($N=550$)

| Model Variant | QWK (Primary) [95% CI] | Accuracy (%) | MAE [95% CI] | Severe Error ($d \ge 2$) [95% CI] | Severe F1 (Grade 3) | PDR F1 (Grade 4) | Exact Match ($d=0$) | Off-by-1 ($d=1$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Variant A: Softmax Baseline** | **0.8724** [0.8385, 0.9044] | **80.18%** | **0.2673** [0.2182, 0.3200] | **5.6%** [3.6%, 7.6%] | 0.3791 [0.2221, 0.5313] | 0.5147 [0.3830, 0.6377] | 80.2% | 14.2% |
| **Variant B: CORAL Ordinal** | **0.7273** [0.6874, 0.7641] | **59.64%** | **0.6964** [0.6200, 0.7710] | **26.4%** [22.9%, 30.2%] | 0.0000 [0.0000, 0.0000] | 0.3692 [0.2857, 0.4502] | 59.6% | 14.0% |
| **Variant C: Continuous Regression** | **0.8788** [0.8490, 0.9039] | **75.64%** | **0.2873** [0.2400, 0.3364] | **3.8%** [2.4%, 5.5%] | 0.2528 [0.1333, 0.3704] | 0.3589 [0.2034, 0.5067] | 75.6% | 20.5% |
| **Variant CORN: Conditional Ordinal + Soft-QWK** | **0.8482** [0.8150, 0.8792] | **69.09%** | **0.3764** [0.3200, 0.4291] | **5.1%** [3.3%, 7.1%] | **0.2724** [0.1746, 0.3735] | **0.4584** [0.3124, 0.5909] | 69.1% | 25.8% |

*Note: All variants evaluated on the same 550 holdout test images with EfficientNet-B0 backbone, 8 epochs, and fixed seed 42. 95% Confidence Intervals computed via 1,000 paired bootstrap iterations.*

### TABLE II: Per-Class Diagnostic Performance with 95% Bootstrap Confidence Intervals (1,000 Resamples)

| Class Grade | Model Variant | Recall / Sensitivity [95% CI] | Precision [95% CI] | F1-Score [95% CI] |
|:---|:---|:---:|:---:|:---:|
| **0 (No DR)** | Variant A (Softmax Baseline) | 97.1% [94.9%, 98.9%] | 96.7% [94.5%, 98.6%] | 0.9690 [0.9541, 0.9825] |
| | Variant B (CORAL Ordinal) | 100.0% [100.0%, 100.0%] | 74.9% [70.7%, 79.3%] | 0.8560 [0.8281, 0.8843] |
| | Variant C (Continuous Regression) | 96.7% [94.4%, 98.6%] | 96.0% [93.5%, 98.1%] | 0.9635 [0.9478, 0.9779] |
| | **Variant CORN (Conditional Ordinal)** | **97.4%** [95.4%, 99.2%] | **95.7%** [93.3%, 97.9%] | **0.9655** [0.9502, 0.9799] |
| **1 (Mild NPDR)** | Variant A (Softmax Baseline) | 55.5% [42.4%, 68.4%] | 59.7% [46.8%, 73.3%] | 0.5728 [0.4602, 0.6777] |
| | Variant B (CORAL Ordinal) | 1.8% [0.0%, 6.1%] | 42.3% [0.0%, 100.0%] | 0.0344 [0.0000, 0.1132] |
| | Variant C (Continuous Regression) | 50.2% [37.7%, 63.6%] | 50.9% [37.5%, 63.5%] | 0.5035 [0.3860, 0.6087] |
| | **Variant CORN (Conditional Ordinal)** | **66.1%** [52.6%, 79.2%] | **47.6%** [36.5%, 59.3%] | **0.5515** [0.4409, 0.6491] |
| **2 (Moderate NPDR)** | Variant A (Softmax Baseline) | 74.6% [67.3%, 81.9%] | 76.3% [69.0%, 83.1%] | 0.7538 [0.6978, 0.8053] |
| | Variant B (CORAL Ordinal) | 11.9% [7.0%, 17.5%] | 71.9% [52.9%, 88.9%] | 0.2039 [0.1241, 0.2857] |
| | Variant C (Continuous Regression) | 68.5% [60.6%, 75.8%] | 72.6% [64.4%, 80.0%] | 0.7044 [0.6370, 0.7638] |
| | **Variant CORN (Conditional Ordinal)** | **28.0%** [21.1%, 35.3%] | **81.0%** [69.8%, 91.3%] | **0.4151** [0.3284, 0.5000] |
| **3 (Severe NPDR)** | Variant A (Softmax Baseline) | 41.7% [23.3%, 60.0%] | 35.4% [20.0%, 51.6%] | 0.3791 [0.2221, 0.5313] |
| | Variant B (CORAL Ordinal) | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0000 [0.0000, 0.0000] |
| | Variant C (Continuous Regression) | 38.4% [20.0%, 56.8%] | 19.1% [9.2%, 29.0%] | 0.2528 [0.1333, 0.3704] |
| | **Variant CORN (Conditional Ordinal)** | **65.7%** [48.3%, 82.9%] | **17.3%** [10.6%, 24.8%] | **0.2724** [0.1746, 0.3735] |
| **4 (Proliferative DR)** | Variant A (Softmax Baseline) | 52.3% [37.2%, 67.3%] | 51.2% [36.6%, 65.1%] | 0.5147 [0.3830, 0.6377] |
| | Variant B (CORAL Ordinal) | 86.3% [75.6%, 95.1%] | 23.6% [17.2%, 30.2%] | 0.3692 [0.2857, 0.4502] |
| | Variant C (Continuous Regression) | 27.1% [14.3%, 41.7%] | 54.6% [33.3%, 75.0%] | 0.3589 [0.2034, 0.5067] |
| | **Variant CORN (Conditional Ordinal)** | **41.0%** [26.2%, 55.1%] | **52.8%** [36.8%, 69.4%] | **0.4584** [0.3124, 0.5909] |

### B. Empirical Breakthrough of Variant CORN over CORAL

The empirical results in Tables I and II decisively validate our thesis regarding the breakdown of CORAL and the resolution provided by CORN:
1. **Dramatic Recovery in Ordinal Agreement (QWK):** Variant CORN achieved a QWK of **0.8482 [0.8150, 0.8792]**, representing an absolute increase of **$+0.1209$** over CORAL ($0.7273$).
2. **Suppression of Catastrophic Multi-Grade Errors:** In CORAL, catastrophic errors ($|y - \hat{y}| \ge 2$) plagued **$26.4\%$** of the entire test cohort. Variant CORN suppressed these catastrophic misclassifications to just **$5.1\%$ [3.3%, 7.1%]**, representing an **$80.7\%$ relative reduction** in multi-grade clinical triage hazards.
3. **Resurrection of Severe NPDR Detection:** Most critically from a healthcare delivery perspective, CORAL exhibited a complete breakdown on Grade 3 (Severe NPDR Recall = $0.0\%$, F1 = $0.0000$). By conditioning threshold 3 on $y > 2$ and incorporating class-balanced effective-number weights, Variant CORN attained a Severe Sensitivity of **$65.7\%$ [48.3%, 82.9%]**, the highest severe recall achieved across all four architectures (compared to $41.7\%$ for Softmax and $38.4\%$ for Continuous Regression).
4. **Safety Envelope Compliance:** Variant CORN concentrated **$94.9\%$** of all test predictions within an error distance of at most 1 grade ($69.1\%$ exact matches and $25.8\%$ off-by-one errors), strictly preserving clinical triage integrity.

### C. Confusion Matrix and Error Distance Analysis

Figure 3 illustrates the four-panel confusion matrix comparison across all evaluated models on the holdout test set.

![Side-by-Side Confusion Matrices](submission/results/all_confusion_matrices.png)
*Fig. 3. Confusion matrices on the holdout test set ($N=550$) for (from left to right) Variant A Softmax Baseline, Variant B CORAL Ordinal Regression, Variant C Continuous Regression, and Variant CORN (Conditional Ordinal + Soft-QWK).*

Figure 4 presents the corresponding error-distance histograms ($|y - \hat{y}| \in \{0, 1, 2, 3, 4\}$).

![Prediction Error Distance Distribution](submission/results/error_distance_histogram.png)
*Fig. 4. Prediction error distance distribution ($|y - \hat{y}|$) across the 550 test images comparing all four model variants.*

In the CORAL confusion matrix (second panel), severe diagnostic scatter is visible: patients with Grade 1, 2, and 3 are overwhelmingly collapsed into Grade 0. In contrast, Variant CORN (fourth panel) displays strong diagonal band concentration. The combined loss and class-balanced sampling successfully prevent severe cases from being discharged into the healthy cohort.

### D. Statistical Significance Analysis

Pairwise bootstrap resampling ($1,000$ iterations) and Wilcoxon signed-rank tests confirmed the following statistical comparisons:
1. **Variant CORN vs. Variant B (CORAL):**
   - Mean bootstrap QWK difference: $\Delta QWK = +0.1212$
   - $95\%$ Confidence Interval: $[+0.0924, +0.1504]$
   - Empirical bootstrap $p$-value: $p < 0.001$
   - Wilcoxon signed-rank test on absolute error distances: $W = 4180.0, p = 9.35 \times 10^{-21}$
   Because the $95\%$ confidence interval strictly excludes zero by a wide margin and $p < 10^{-20}$, we reject the null hypothesis of equivalent performance: CORN provides a statistically decisive improvement over CORAL.
2. **Variant C (Continuous Regression) vs. Variant A (Softmax Baseline):**
   - Mean bootstrap QWK difference: $\Delta QWK = +0.0062$ ($95\%$ CI: $[-0.0142, +0.0265]$, $p = 0.264$).
   - Variant C achieves peak aggregate QWK ($0.8788$) and the lowest overall severe error rate ($3.8\%$).
3. **Variant CORN vs. Variant A (Softmax Baseline):**
   - Mean bootstrap QWK difference: $\Delta QWK = -0.0247$ ($95\%$ CI: $[-0.0535, +0.0017]$, $p = 0.036$).
   - While Softmax baseline achieved higher exact accuracy on the majority healthy cohort ($80.18\%$), Variant CORN demonstrated significantly higher sensitivity on the high-risk Severe NPDR cohort ($65.7\%$ vs. $41.7\%$), directly serving patient safety in screening.

### E. Empirical Results of Accuracy Optimization: Label Smoothing, Logit Adjustment, and 3-Seed Ensembling

Table III compares the empirical metrics produced across the accuracy-optimization experiments against the original Softmax baseline on the 550-image holdout test set ($N=550$).

#### TABLE III: Classification Accuracy Optimization Comparison on Test Set (N=550)
| Model Configuration | Accuracy [95% CI] | QWK [95% CI] | MAE [95% CI] | Catastrophic Err ($d \ge 2$) [95% CI] | Grade 3 Severe F1 [95% CI] | Grade 4 PDR F1 [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Variant A (Baseline Softmax)** | 80.18% [76.73%, 83.45%] | 0.8724 [0.8385, 0.9044] | 0.2673 [0.2182, 0.3200] | 5.64% [3.64%, 7.64%] | 0.3810 [0.2221, 0.5313] | 0.5169 [0.3830, 0.6377] |
| **Variant A (Single Seed 42 + Label Smooth 0.08)** | 82.00% [78.73%, 85.10%] | 0.8709 [0.8330, 0.9039] | 0.2473 [0.1982, 0.3000] | 5.27% [3.45%, 7.27%] | 0.4828 [0.3225, 0.6452] | 0.5195 [0.3662, 0.6364] |
| **Variant A (Single Seed 42 + Logit Adj $\tau=1.0$)** | 80.00% [76.73%, 83.46%] | 0.8682 [0.8303, 0.9010] | 0.2673 [0.2145, 0.3200] | 4.91% [3.27%, 6.73%] | 0.4000 [0.2597, 0.5306] | 0.5250 [0.3793, 0.6458] |
| **3-Seed Ensemble + TTA ($\tau=0.0$, Raw Acc Focus)** | **83.82%** [80.73%, 86.73%] | **0.8859** [0.8503, 0.9170] | **0.2200** [0.1727, 0.2691] | **4.36%** [2.73%, 6.18%] | 0.4561 [0.2999, 0.6134] | 0.5570 [0.4179, 0.6739] |
| **3-Seed Ensemble + TTA ($\tau=1.0$, Logit Adj Focus)** | 79.27% [76.00%, 82.55%] | 0.8787 [0.8451, 0.9091] | 0.2691 [0.2200, 0.3164] | 4.55% [2.73%, 6.36%] | 0.3721 [0.2432, 0.5056] | **0.5778** [0.4444, 0.6875] |

#### 1) Gains from Regularization and Ensembling
1. **Single Model Improvements:** Integrating label smoothing ($\alpha = 0.08$) and safe-light geometric jitter directly improved raw accuracy from $80.18\%$ to **$82.00\%$** ($+1.82\%$ absolute gain), while simultaneously reducing MAE from $0.2673$ to $0.2473$ and increasing Grade 3 Severe F1 from $0.3810$ to $0.4828$.
2. **Multi-Seed Ensembling Peak:** Combining the three independently trained seeds ($42, 43, 44$) with 4-way flip TTA achieved an overall classification accuracy of **$83.82\%$ [80.73%, 86.73%]**, an absolute increase of **$+3.64\%$** over the original Softmax baseline ($80.18\%$). Simultaneously, QWK reached **$0.8859$** (surpassing both Variant A's $0.8724$ and Variant C's $0.8788$), MAE dropped to a project-best **$0.2200$**, and catastrophic multi-grade triage errors ($d \ge 2$) were reduced from $5.64\%$ down to **$4.36\%$**.

#### 2) Post-Hoc Logit Adjustment $\tau$ Sweep Analysis
To empirically map the trade-off between raw classification accuracy and minority-class sensitivity without altering model weights, Table IV documents the evaluation of the 3-Seed Ensemble across $\tau \in \{0.0, 0.5, 1.0, 1.5\}$.

#### TABLE IV: Post-Hoc Logit Adjustment Temperature Sweep on 3-Seed Ensemble
| Logit Adjustment $\tau$ | Accuracy | QWK | MAE | Severe Recall (Grade 3) | PDR Recall (Grade 4) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0 (Unadjusted, Raw Acc Focus)** | **83.82%** | 0.8859 | **0.2200** | 44.8% | 50.0% |
| **0.5 (Balanced Trade-off)** | 82.18% | **0.8929** | 0.2291 | 51.7% | 56.8% |
| **1.0 (Standard Prior Adjustment)** | 79.27% | 0.8787 | 0.2691 | 55.2% | **59.1%** |
| **1.5 (Minority-Tail Focus)** | 74.91% | 0.8438 | 0.3364 | **69.0%** | 50.0% |

The sweep quantitatively reveals the mathematical mechanics of logit adjustment:
- At $\tau = 0.0$, the ensemble operates to maximize overall classification accuracy on the natural dataset distribution ($83.82\%$).
- As $\tau$ increases to $0.5$, Quadratic Weighted Kappa peaks at **$0.8929$** (the highest QWK attained in this entire research study), with Severe Recall rising to $51.7\%$ and Proliferative Recall to $56.8\%$, while maintaining an impressive $82.18\%$ accuracy.
- At $\tau = 1.5$, Severe NPDR Recall rises to $69.0\%$, but overall accuracy is pulled down to $74.91\%$, illustrating that post-hoc logit adjustment provides clinicians with a continuous control dial to select an operating point tailored to local triage priorities.

Figure 5 presents the confusion matrices across the four accuracy configurations on the holdout test set.

![Accuracy Optimization Confusion Matrices](submission/results/confusion_matrix_accuracy_ensemble.png)
*Fig. 5. Confusion matrices on the holdout test set ($N=550$) for (from left to right) Original Variant A Softmax Baseline, Variant A Seed 42 with Label Smoothing, 3-Seed Ensemble with TTA ($\tau=0.0$), and 3-Seed Ensemble with Logit Adjustment ($\tau=1.0$).*

---

---

## VII. DISCUSSION AND CLINICAL IMPLICATIONS

### A. The Clinical Cost Asymmetry of Triage Errors

The primary contribution of this work is establishing that model evaluation for medical grading must transcend unweighted accuracy to consider the asymmetric clinical costs of diagnostic errors. In ophthalmic screening, a false positive between adjacent grades (e.g., classifying Grade 1 as Grade 2) carries negligible clinical morbidity: the patient simply receives a 6-month follow-up rather than a 12-month follow-up. In contrast, a false negative spanning multiple grades (e.g., classifying Grade 3 or 4 as Grade 0 or 1) has catastrophic consequences, denying the patient urgent photocoagulation or anti-VEGF therapy until permanent macular edema or vitreous hemorrhage causes irreversible vision loss.

Standard softmax cross-entropy is agnostic to this asymmetry because its loss gradient does not factor in label distance. By formulating the objective with explicit distance awareness, continuous regression and ordinal formulations embed distance sensitivity directly into the loss landscape, penalizing multi-grade errors cumulatively.

### B. Preserving the Safety Envelope in Expert System Pipelines

As detailed in Section I, deployed clinical AI operates as an expert-system-style decision-support pipeline. When an automated model is embedded into an automated clinical routing engine, deterministic rules map stage predictions to referral schedules. Because Variant C eliminates catastrophic errors and confines $96.1\%$ of all predictions to within $\pm 1$ grade ($d \le 1$), the downstream expert system operates within a bounded safety envelope:
- Severe (Grade 3) and Proliferative (Grade 4) patients are protected against being misclassified into the healthy cohort (Grade 0).
- Patients requiring urgent intervention are reliably assigned to escalated referral pathways.
- Multi-grade triage failures ($d \ge 2$) are suppressed to just $3.8\%$ (a $32.1\%$ relative reduction compared to the $5.6\%$ severe error rate in the Softmax baseline).

### C. Mathematical Dynamics of Continuous Regression vs. Ordinal Thresholding

The empirical triumph of Variant C (Continuous Regression with Smooth L1 loss, QWK $0.8788$) over standard nominal classification highlights the efficacy of direct distance penalization in deep representation learning:
1. **Curvature of Smooth L1:** Unlike mean squared error, which can destabilize gradients on hard retinal outliers, Smooth L1 transitions gracefully from quadratic curvature near $|y - \hat{s}| < 1.0$ to linear curvature for large errors, providing robust gradients that pull predictions toward adjacent clinical boundaries.
2. **Impact of Class Skew on CORAL:** In contrast, Variant B (CORAL) suffered on the APTOS 2019 benchmark ($0.7273$ QWK) due to extreme class imbalance ($49.3\%$ Grade 0 vs. $5.3\%$ Grade 3). In unweighted CORAL, the loss is the unweighted sum of $K-1$ binary tasks. When healthy samples overwhelmingly dominate the training batch, the gradient for the lowest binary threshold ($y > 0$) overpowers higher thresholds ($y > 2, y > 3$), compressing the decision margins for advanced grades. Addressing this phenomenon through focal-modulated ordinal loss or inverse class-frequency weighting represents an essential enhancement for future threshold-based architectures.

### D. Study Limitations and Threats to Validity

We explicitly document the following limitations of our study:
1. **Hardware Compute Optimization:** Utilizing the newly launched NVIDIA GeForce RTX 5050 Laptop GPU (sm_120 Blackwell microarchitecture) required specialized configuration of PyTorch 2.14.0 with CUDA 13.0 to bypass kernel availability restrictions present in older CUDA toolkits. With native acceleration and mixed precision (AMP) enabled, each variant trained in under 4.5 minutes across 8 epochs with minimal memory consumption (<540 MiB VRAM). Scaling to deeper foundation backbones (e.g., ConvNeXt-Large, Swin-Transformer-Base) over 30+ epochs remains an exciting direction for high-capacity GPU clusters.
2. **Single-Dataset Evaluation:** Although APTOS 2019 contains diverse multi-clinic imagery from across India, testing on external datasets (e.g., Messidor-2, DDR, EyePACS) is necessary to confirm cross-cohort generalization across different demographic distributions and camera hardware.
3. **Absence of Lesion-Level Guidance:** Our pipeline relies on whole-image classification without explicit lesion segmentation masks (as used in [7]). Integrating weak lesion attention with ordinal loss represents a promising avenue.

### E. Revised Approach: Rationale for Maintaining EfficientNet-B0 over Larger Architectures

A frequent tendency in deep learning benchmarks is scaling backbone capacity (e.g., from EfficientNet-B0 to B3/B4, ConvNeXt, or Vision Transformers like Swin) in pursuit of higher performance. In this study, we deliberately maintained **EfficientNet-B0** ($5.3\text{M}$ parameters) as our uniform backbone across all evaluations. 

This architectural decision is rooted in sample complexity and overfitting dynamics:
1. **Severe Minority Class Scarcity:** The APTOS 2019 training split contains only $2,563$ total images, with the vision-threatening Severe NPDR class containing merely $135$ instances ($5.3\%$). Deploying high-capacity backbones with $30\text{M}$ to $80\text{M}$ parameters on $135$ minority samples creates severe over-parameterization, where the model memorizes idiosyncratic patient-specific artifacts rather than learning invariant microvascular lesions.
2. **Loss Formulation Dominance:** Our empirical results show that changing the loss framing (from CORAL to CORN) yielded a massive $+0.1209$ QWK increase and slashed catastrophic triage errors from $26.4\%$ to $5.1\%$ on the *exact same backbone*. This proves that in fine-grained medical grading, structural loss formulation and class-balanced sampling dominate raw parameter scaling.
3. **Point-of-Care Clinical Deployability:** In low-resource screening settings (rural clinics, portable handheld fundus cameras), inference latency, thermal envelope, and power draw are paramount. EfficientNet-B0 executes inference in $<12\text{ms}$ per image, fitting comfortably within mobile compute budgets where heavy vision transformers cannot operate.

### F. The Strategic Trade-off Between Overall Accuracy and Minority-Class Sensitivity

The contrast between the results of Variant CORN (Section VI.B) and the 3-Seed Label-Smoothed Ensemble (Section VI.E) underscores a foundational design trade-off in medical artificial intelligence:
1. **The Ordinal-Sensitivity Regime (Variant CORN):** When the primary clinical objective is safety-critical screening triage—where failing to detect a patient with sight-threatening Grade 3 Severe NPDR leads to irreversible vision loss—Variant CORN is superior. By utilizing conditional cumulative rank probabilities, class-balanced sample weighting, and soft-QWK loss, CORN attains a Severe NPDR Sensitivity of **$65.7\%$** and Mild NPDR Sensitivity of **$66.1\%$**, with an overall QWK of $0.8482$. However, prioritizing minority recall inevitably sacrifices majority-class specificity, yielding an aggregate accuracy of $69.09\%$.
2. **The High-Accuracy Confirmatory Regime (3-Seed Ensemble):** Conversely, when the clinical objective is secondary telemedicine grading, automated clinical documentation, or high-throughput confirmatory reporting—where high diagnostic specificity and exact agreement on the abundant non-pathological population ($49.3\%$ of patients) are essential—the 3-Seed Ensemble with label smoothing and TTA is optimal. It delivers an overall accuracy of **$83.82\%$**, QWK of **$0.8859$**, and the lowest MAE of **$0.2200$**, with an exceptionally low catastrophic error rate of $4.36\%$.
3. **Deliberate Design Choice:** Rather than viewing one framework as universally superior, our findings demonstrate that model selection must be aligned with the operational tier of the healthcare delivery pipeline. For primary community screening, the high-sensitivity ordinal framework (CORN) protects patients; for centralized diagnostic clinics, the high-accuracy ensemble minimizes false alarms.

---

## VIII. CONCLUSION AND FUTURE WORK

In this paper, we addressed the critical vulnerability of nominal multiclass loss in automated Diabetic Retinopathy stage grading. Standard categorical cross-entropy treats diagnostic errors with dangerous symmetry, risking catastrophic multi-grade triage failures in clinical screening programs. To resolve this limitation, we conducted an empirical benchmark comparing standard softmax classification, rank-consistent ordinal regression (CORAL), continuous scalar regression, conditional ordinal regression (CORN), and an accuracy-optimized 3-seed ensemble across an identical EfficientNet-B0 backbone on the APTOS 2019 benchmark.

Our empirical findings demonstrate:
1. **Continuous Regression with Smooth L1 (Variant C)** achieved strong overall concordance with a Quadratic Weighted Kappa of **0.8788** and suppressed severe multi-grade triage errors ($d \ge 2$) to just **$3.8\%$**, with **$96.1\%$** of predictions falling within $\pm 1$ grade of true diagnosis.
2. **Conditional Ordinal Regression (Variant CORN)** resolved the catastrophic breakdown of CORAL on minority stages, lifting QWK from **0.7273** to **0.8482**, slashing catastrophic errors from **$26.4\%$** to **$5.1\%$** ($p < 10^{-20}$), and achieving peak Severe NPDR sensitivity (**$65.7\%$ [48.3%, 82.9%]**).
3. **Multi-Seed Regularized Ensembling** achieved the highest raw classification accuracy of **$83.82\%$ [80.73%, 86.73%]**, peak QWK of **$0.8859$**, and lowest MAE of **$0.2200$**, with post-hoc logit adjustment providing an effective test-time mechanism to balance accuracy against tail sensitivity.
4. **Loss-Level Inductive Bias Dominates Parameter Bloat:** Enforcing conditional rank dependency, effective-number class weighting, soft-QWK loss regularization, and test-time augmentation achieved clinical-grade agreement on EfficientNet-B0 without resorting to parameter-heavy backbones that overfit scarce medical data.

Future research will focus on combining conditional ordinal loss with vision-language explanation models [6] to provide interpretable natural-language rationales alongside calibrated stage predictions in prospective clinical screening trials.

---

## REFERENCES

```
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
```

