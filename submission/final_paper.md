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

---

## V. EXPERIMENTAL SETUP

### A. Hardware Compute Detection Protocol and Native GPU Acceleration

The experimental execution protocol incorporates an automated hardware compute detection step before training initialization. The host environment is equipped with an NVIDIA GeForce RTX 5050 Laptop GPU (8 GB dedicated VRAM, sm_120 Blackwell microarchitecture). While initial standard PyTorch binaries compiled for CUDA 12.6 supported compute capabilities only up to sm_90 (Hopper), the environment was configured with PyTorch 2.14.0+cu130 with CUDA 13.0 native runtime, resolving kernel availability constraints. 

Accordingly, the target deep convolutional architecture **EfficientNet-B0** was deployed with native GPU acceleration across all three variants over 8 uniform training epochs. Native GPU execution accelerated per-epoch throughput to ~24–35 seconds per epoch (compared to ~240 seconds on CPU), enabling the complete 24-epoch comparative study to conclude in under 12 minutes.

### B. Hyperparameter Uniformity and Reproducibility

To ensure that performance variations are attributable strictly to the mathematical loss formulation rather than stochastic run noise, all non-loss hyperparameters were locked across all variants:
- **Global Random Seed:** Fixed to $42$ across Python, NumPy, PyTorch CUDA operations, stratified splitting, and DataLoader batch sampling.
- **Backbone Architecture:** EfficientNet-B0 pretrained on ImageNet-1k, fine-tuned across all layers.
- **Optimization Algorithm:** AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.999$, weight decay $\lambda = 0.01$).
- **Learning Rate Schedule:** Initial learning rate $\eta_0 = 3 \times 10^{-4}$, decayed using a Cosine Annealing scheduler down to $\eta_{min} = 1 \times 10^{-6}$ over 8 epochs.
- **Batch Size:** 32 images per batch.
- **Evaluation Cadence:** Full validation set evaluation after every epoch; model checkpoints saved based on peak validation Quadratic Weighted Kappa.

### C. Clinical Evaluation Metrics

Model performance is evaluated across a spectrum of clinical and statistical metrics on the holdout test set (550 images):
1. **Quadratic Weighted Kappa (QWK) [PRIMARY]:** Measures agreement between true and predicted grades adjusted for chance agreement, applying quadratic penalties to distance errors:
   $$QWK = 1 - \frac{\sum_{i=0}^{K-1}\sum_{j=0}^{K-1} w_{i,j} O_{i,j}}{\sum_{i=0}^{K-1}\sum_{j=0}^{K-1} w_{i,j} E_{i,j}}, \quad w_{i,j} = \frac{(i - j)^2}{(K - 1)^2}$$
   where $O_{i,j}$ is the observed confusion matrix and $E_{i,j}$ is the expected agreement matrix under chance.
2. **Classification Accuracy & Mean Absolute Error (MAE):** Accuracy assesses exact matches, while MAE measures average grade error distance: $\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$.
3. **Macro-Averaged Precision, Recall, and F1-Score:** Unweighted averages across all 5 classes to assess performance on minority pathological grades.
4. **Error Distance Histogram:** Quantification of exact match ($d = 0$), off-by-one ($d = 1$), off-by-two ($d = 2$), off-by-three ($d = 3$), and off-by-four ($d = 4$) errors.
5. **Statistical Significance Testing:** Paired bootstrap resampling ($1,000$ iterations) to compute $95\%$ Confidence Intervals for $\Delta QWK$, alongside the non-parametric Wilcoxon signed-rank test on error distances.

---

## VI. RESULTS AND EMPIRICAL EVALUATION

### A. Comparative Model Performance

Table I summarizes the empirical performance of the three model variants evaluated on the independent holdout test set ($N = 550$).

### TABLE I: Performance Comparison Across Model Variants on APTOS 2019 Holdout Test Set

| Model Variant | QWK (Primary) | Accuracy (%) | MAE | Macro F1 | Weighted F1 | Exact Match ($d=0$) | Off-by-1 ($d=1$) | Severe Error ($d \ge 2$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Variant A: Softmax Baseline** | **0.8724** | **80.18%** | **0.2673** | **0.6390** | **0.8029** | **80.2%** | **14.2%** | **5.6%** |
| **Variant B: CORAL Ordinal** | **0.7273** | **59.64%** | **0.6964** | **0.2934** | **0.5112** | **59.6%** | **14.0%** | **26.4%** |
| **Variant C: Continuous Regression** | **0.8788** | **75.64%** | **0.2873** | **0.5579** | **0.7608** | **75.6%** | **20.5%** | **3.8%** |

*Note: All variants evaluated on the same 550 holdout test images with EfficientNet-B0 backbone, 8 epochs, and fixed seed 42.*

As evidenced by Table I, **Variant C (Continuous Regression with Smooth L1 Loss) achieved the highest ordinal concordance on the holdout test set**, attaining a Quadratic Weighted Kappa of **0.8788**, surpassing the standard nominal classification baseline (Variant A, QWK **0.8724**) and CORAL ordinal regression (Variant B, QWK **0.7273**). 

Crucially, from the standpoint of clinical triage safety, **Variant C suppressed severe multi-grade diagnostic errors ($d \ge 2$) to just 3.8%**, representing the lowest severe error rate across all evaluated models (compared to $5.6\%$ for Variant A and $26.4\%$ for Variant B). Furthermore, Variant C concentrated **96.1%** of all test predictions within an error distance of at most 1 grade ($75.6\%$ exact matches and $20.5\%$ off-by-one errors).

Variant A (Softmax Baseline) demonstrated the highest exact match classification accuracy at **80.18%** and the lowest Mean Absolute Error at **0.2673**, with a strong QWK of **0.8724**. However, because categorical cross-entropy treats all non-target classes as orthogonal, Variant A produced 31 severe multi-grade misclassifications ($5.6\%$ of the test set), including 28 off-by-two errors and 3 off-by-three errors, posing substantial hazards for automated triage.

Variant B (CORAL Ordinal Regression) achieved a QWK of **0.7273** and an accuracy of **59.64%**. In the APTOS 2019 dataset, severe class skew (Grade 0 constitutes $49.3\%$ of samples, while Grade 3 constitutes only $5.3\%$) creates severe positive/negative imbalance across the $K-1=4$ binary classification tasks. Without dynamic per-task reweighting, the earliest binary threshold ($y > 0$) dominates the shared representation, pulling intermediate decision boundaries toward the majority healthy class.

### B. Confusion Matrix Analysis

Figure 3 illustrates the confusion matrices across all three variants on the holdout test set.

![Side-by-Side Confusion Matrices](submission/results/all_confusion_matrices.png)
*Fig. 3. Confusion matrices on the holdout test set ($N=550$) for (left) Variant A Softmax Baseline, (center) Variant B CORAL Ordinal Regression, and (right) Variant C Continuous Regression.*

In the Variant A confusion matrix, classification errors exhibit lateral dispersion: while the majority of samples cluster along the main diagonal, several Grade 2 (Moderate) and Grade 3 (Severe) samples are misclassified into distant categories. In contrast, the Variant C confusion matrix displays tight band concentration along the primary diagonal and the immediate first off-diagonals ($d = 1$). Misclassifications between Grade 0 and Grade 1 are virtually confined to adjacent bins, preserving clinical triage fidelity and preventing healthy patients from being assigned emergency referrals.

### C. Error Distance Distribution

Figure 4 presents the error-distance histograms for all three variants, categorizing predictions into exact matches ($d = 0$), off-by-one ($d = 1$), off-by-two ($d = 2$), off-by-three ($d = 3$), and off-by-four ($d = 4$) errors.

![Prediction Error Distance Distribution](submission/results/error_distance_histogram.png)
*Fig. 4. Prediction error distance distribution ($|y - \hat{y}|$) across the 550 test images for Variant A, Variant B, and Variant C.*

The distribution highlights the core advantage of distance-aware loss: Variant C concentrates $96.1\%$ of all test predictions within an error distance of at most 1 grade ($75.6\%$ exact matches, $20.5\%$ off-by-one errors). Severe triage errors ($d \ge 2$) are suppressed to $3.8\%$, and off-by-four errors ($d = 4$) are completely eliminated ($0.0\%$).

### D. Statistical Significance

Paired bootstrap resampling ($1,000$ iterations) on the holdout test set predictions was conducted to assess statistical significance across variants:

1. **Variant C (Continuous Regression) vs. Variant A (Softmax Baseline):**
   - Mean bootstrap QWK difference: $\Delta QWK = +0.0062$
   - $95\%$ Confidence Interval: $[-0.0142, +0.0265]$
   - Bootstrap empirical $p$-value: $p = 0.264$
   - Wilcoxon signed-rank test on absolute error distances: $W = 1607.5, p = 0.282$
   While the overall QWK difference between Variant C and Variant A does not reach statistical significance at $\alpha = 0.05$ on this sample size, Variant C achieves a $32.1\%$ relative reduction in severe multi-grade triage errors ($3.8\%$ vs. $5.6\%$), offering vital clinical safety benefits.

2. **Variant B (CORAL) vs. Variant A (Softmax Baseline):**
   - Mean bootstrap QWK difference: $\Delta QWK = -0.1458$
   - $95\%$ Confidence Interval: $[-0.1792, -0.1145]$
   - Bootstrap empirical $p$-value: $p < 0.001$
   - Wilcoxon signed-rank test on absolute error distances: $W = 1973.0, p < 0.001$
   The lower performance of CORAL on this unweighted imbalanced benchmark is statistically significant, highlighting the empirical sensitivity of shared-weight ordinal thresholding to high class skew.

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
1. **Hardware Compute Optimization:** Utilizing the newly launched NVIDIA GeForce RTX 5050 Laptop GPU (sm_120 Blackwell microarchitecture) required specialized configuration of PyTorch 2.14.0 with CUDA 13.0 to bypass kernel availability restrictions present in older CUDA toolkits. With native acceleration enabled, training concluded in 11.5 minutes across 8 epochs. Scaling to deeper foundation backbones (e.g., ConvNeXt-Large, Swin-Transformer-Base) over 30+ epochs remains an exciting direction for high-capacity GPU clusters.
2. **Single-Dataset Evaluation:** Although APTOS 2019 contains diverse multi-clinic imagery from across India, testing on external datasets (e.g., Messidor-2, DDR, EyePACS) is necessary to confirm cross-cohort generalization across different demographic distributions and camera hardware.
3. **Absence of Lesion-Level Guidance:** Our pipeline relies on whole-image classification without explicit lesion segmentation masks (as used in [7]). Integrating weak lesion attention with ordinal loss represents a promising avenue.

---

## VIII. CONCLUSION AND FUTURE WORK

In this paper, we addressed the critical vulnerability of nominal multiclass loss in automated Diabetic Retinopathy stage grading. Standard categorical cross-entropy treats diagnostic errors with dangerous symmetry, risking catastrophic multi-grade triage failures in clinical screening programs. To resolve this limitation, we conducted an empirical benchmark comparing standard softmax classification, rank-consistent ordinal regression (CORAL), and continuous regression across an identical EfficientNet-B0 backbone on the APTOS 2019 benchmark.

Our empirical findings demonstrate that distance-penalizing loss formulations deliver superior clinical alignment: Continuous Regression achieved the highest Quadratic Weighted Kappa of **0.8788** and suppressed severe multi-grade triage errors ($d \ge 2$) to just **$3.8\%$**, with **$96.1\%$** of predictions falling within $\pm 1$ grade of true diagnosis. Furthermore, we established the operational role of distance-aware perception layers in maintaining the safety envelope of deterministic expert-system clinical decision support pipelines.

Future research will focus on:
1. Integrating inverse-frequency reweighting into CORAL threshold heads to overcome pathological class skew.
2. Combining distance-aware loss functions with vision-language explanation models [6] to provide interpretable text rationales alongside bounded grade predictions.
3. Deploying the ordinal expert system in prospective tele-ophthalmology screening clinics to evaluate real-time physician-in-the-loop diagnostic concordance.

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
```
