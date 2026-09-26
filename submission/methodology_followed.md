# Methodology Followed

**Project Title:** Ordinal-Aware Deep Learning Framework for Diabetic Retinopathy Severity Grading and Clinical Triage Decision Support  
**Authors:** Student Project Team (Members 1, 2, 3, and 4)  
**Course:** Knowledge-Based Systems & Clinical Decision Support (Mini-Project Track)  
**Target Submission Format:** IEEE Academic Format (replicated from `sample_report.pdf`)

---

## 1. Literature Review Organization and Division of Labor
To establish a rigorous theoretical and clinical foundation for our investigation, our team structured the preliminary investigation as a four-person parallel literature review. The target corpus comprised seven authoritative recent studies on automated Diabetic Retinopathy (DR) grading, transfer learning, class imbalance, and multimodal explainability. Because exactly seven papers were assigned (rather than an even eight), we structured the review into three paired assignments and one focused solo assignment:

- **Member 1 (Foundational Transfer Learning & Dual-Branch Ensembles):** Evaluated Paper 1 (Khalifa et al., *Acta Informatica Medica* 2019 [1]) and Paper 2 (Shakibania et al., *arXiv* 2023 [2]), reviewing lightweight CNN benchmarks and dual-branch feature fusion backbones on the APTOS 2019 dataset.
- **Member 2 (Multi-Target Formulations & Imbalance Preprocessing):** Evaluated Paper 3 (Tymchenko et al., *VISIGRAPP* 2020 [3]) and Paper 4 (Mardianta et al., *arXiv* 2025 [4]), analyzing multi-target ordinal/continuous objectives, CLAHE contrast enhancement, and synthetic oversampling via SMOTE.
- **Member 3 (Fuzzy Soft Decision Layers & Multimodal Vision-Language Ensembles):** Evaluated Paper 5 (Karthik et al., *arXiv* 2025 [5]) and Paper 6 (Khokhar et al., *arXiv* 2024 [6]), examining channel-attention mechanisms, fuzzy logic membership layers, and vision-language model (VLM) clinical rationales.
- **Member 4 (Clinical Lesion Segmentation & Referral Decision-Support Systems):** Evaluated Paper 7 (Manoj & Bhosale, *arXiv* 2024 [7]) as an individual research study, investigating U-Net semantic segmentation and downstream clinical referral recommendation architectures.

Each team member analyzed their assigned papers in full, extracting the exact problem addressed, network architectures, datasets, reported empirical numbers (strictly without fabrication), explicit author-acknowledged limitations, and unacknowledged technical gaps. The deliverables were compiled into four dedicated markdown records (`summaries/summary_paper_1_2.md`, `summaries/summary_paper_3_4.md`, `summaries/summary_paper_5_6.md`, and `summaries/summary_paper_7.md`).

---

## 2. Selection and Justification of the Ordinal-Regression Thesis
Our primary thesis—*that formulating diabetic retinopathy stage grading as an ordinal regression problem mathematically enforces monotonic boundary geometry, eliminates catastrophic multi-grade clinical errors, and delivers superior clinical triage decision support compared to nominal softmax classification*—was directly synthesized from critical gaps identified across the seven literature review papers:

1. **Failure of Nominal Cross-Entropy on Clinical Severity Scales:** In Khalifa et al. [1] and Mardianta et al. [4], 5-tier DR grading is treated as an unordered categorical classification problem via standard softmax cross-entropy. Categorical cross-entropy penalizes an off-by-one boundary error (e.g., Grade 1 Mild vs. Grade 2 Moderate) identically to an off-by-four misdiagnosis (e.g., Grade 4 Proliferative misclassified as Grade 0 No DR). In clinical triage, this nominal blindness produces unsafe outcomes where patients requiring emergency vitrectomy are categorized as healthy.
2. **The Severe NPDR Sensitivity Collapse:** In Shakibania et al. [2], despite employing a complex dual-branch backbone (ResNet-50 + EfficientNet-B0) and Complement Cross Entropy (CCE), the sensitivity for Grade 3 (Severe NPDR) collapsed to an unacceptable 36.84%. Because CCE penalizes non-target classes uniformly, the network lacked the ordinal gradient signal necessary to preserve intermediate boundary margins between Moderate (Grade 2) and Severe (Grade 3) disease.
3. **The Unablated Multi-Target Confound:** Tymchenko et al. [3] achieved a world-class Kaggle competition score (0.92546 QWK) by combining classification, ordinal regression, and continuous regression heads with heavy ensembling and 35k-image EyePACS pre-training. However, the authors never isolated the independent effect of ordinal regression versus continuous regression versus classification on a uniform backbone. Our project provides this missing controlled ablation.
4. **Heuristic Class Collapsing vs. Principled Rank Modeling:** Karthik et al. [5] attempted to resolve decision boundary ambiguity by implementing fuzzy membership layers, but were forced to collapse the 5 clinical stages into 3 coarse groups to achieve stable metrics, yet still missed 42% of severe cases (Recall = 0.58). We hypothesized that mathematically rigorous ordinal regression (Consistent Rank Logits, CORAL) naturally models the continuous disease spectrum without sacrificing the standardized 5-tier clinical resolution.
5. **Connecting Grading to Decision-Support Triage Rules:** Manoj & Bhosale [7] demonstrated that automated grading must drive downstream clinical referral recommendations. However, an unverified LLM referral generator is highly unsafe if the upstream classifier makes multi-grade ordinal errors. We therefore recognized that ordinal regression is essential to serve as the reliable, distance-calibrated foundation for a deterministic clinical expert triage system.

---

## 3. Computational Tools and Environment
The experimental pipeline was implemented with the following software and hardware stack:
- **Development & Pair Programming:** Antigravity agentic coding assistant for structured pipeline orchestration, script development, and rigorous verification.
- **Deep Learning Framework:** PyTorch (v2.x) and Torchvision with NVIDIA CUDA accelerator runtime.
- **Computer Vision & Image Processing:** OpenCV (`opencv-python-headless`) for circular cropping and Contrast Limited Adaptive Histogram Equalization (CLAHE) in LAB color space; Pillow (`PIL`) for image serialization.
- **Scientific Computing & Metrics:** Scikit-learn (`scikit-learn`), NumPy, Pandas, and SciPy for stratified dataset partitioning, Quadratic Weighted Kappa (QWK), Mean Absolute Error (MAE), Wilcoxon signed-rank tests, and bootstrap confidence intervals.
- **Data Visualization:** Matplotlib and Seaborn for publication-grade class distribution charts, confusion matrices, and error-distance histograms.
- **Documentation & Publishing:** Markdown and `python-docx` for IEEE-compliant DOCX document compilation.
- **Dataset Source:** Asia Pacific Tele-Ophthalmology Society (APTOS) 2019 Blindness Detection dataset (3,662 high-resolution retinal fundus photographs).

---

## 4. Timeline Breakdown Across the Two-Day Sprint

| Phase / Day | Hours | Primary Milestone & Activities Executed |
| :--- | :---: | :--- |
| **Day 1: Morning** | 09:00 – 13:00 | **Environment Audit & Literature Review:** Extracted structure, style, and word-count targets from `sample_report.pdf`. Retrieved full text for all 7 reference papers. Completed Phase 1 parallel reading passes and produced `summary_paper_1_2.md`, `summary_paper_3_4.md`, `summary_paper_5_6.md`, and `summary_paper_7.md`. |
| **Day 1: Afternoon** | 14:00 – 18:30 | **Data Engineering & Exploratory Analysis:** Established `./data/aptos2019/`, executed automated dataset extraction and validation (3,662 images verified). Implemented circular cropping and LAB-space CLAHE in `data.py`. Executed `eda.py` to produce class distribution bar charts and sample comparative panels in `./submission/results/`. |
| **Day 2: Morning** | 09:00 – 13:30 | **Model Implementation & Controlled Training:** Implemented `models.py` and `train.py`. Verified compute environment (NVIDIA RTX 5050 GPU / CPU fallback detection). Executed strictly uniform fine-tuning across Variant A (Softmax), Variant B (CORAL Ordinal), and Variant C (Continuous Regression) under fixed global seed (42). Saved checkpoints, loss curves, and prediction CSVs. |
| **Day 2: Afternoon** | 14:30 – 19:00 | **Evaluation, Viva Prep & Final Paper Publishing:** Executed `evaluate.py` to compute QWK, MAE, confusion matrices, error histograms, and paired bootstrap significance tests. Identified CORAL independent-threshold failure mode on minority severe class. Authored initial methodology and talking points records. |
| **Day 2: Evening (Sprint Extension)** | 19:00 – 22:30 | **Architectural Refinement (CORN + Soft-QWK + Cui Weighting):** Implemented `variant_CORN` to overcome CORAL threshold collapse. Integrated Conditional Ordinal Regression (CORN), Cui et al. effective number of samples weighting ($\beta=0.9999$), `WeightedRandomSampler`, differentiable Soft-QWK loss ($\lambda=0.20$), early block freezing, and Test-Time Augmentation (TTA). Enforced strict hardware safety limits (mixed precision AMP, in-loop VRAM safeguard, telemetry). Generated 1,000 paired per-class bootstrap CIs, updated all comparison tables, and recompiled `final_paper.docx`. |

---

## 5. Division of Labor for the Technical Build
To ensure high velocity and accountability across the implementation lifecycle, technical responsibilities were distributed among team members:
- **Member 1 (Preprocessing & Data Pipeline):** Led `submission/code/data.py`, implementing the automated circular crop bounding-box extraction, LAB-space CLAHE contrast enhancement algorithm, Cui et al. class-balanced weight calculation ($\beta=0.9999$), and the PyTorch `APTOSDataset` / `DataLoader` pipeline with stratified 70/15/15 partitioning and optional `WeightedRandomSampler`.
- **Member 2 (Architecture & Loss Function Engineering):** Authored `submission/code/models.py`, implementing the shared backbone integration, the CORAL ordinal regression layer, continuous regression head, conditional CORN classification head ($K-1$ conditional conditional probabilities), differentiable `SoftQWKLoss`, and 4-way flip Test-Time Augmentation (TTA).
- **Member 3 (Training Pipeline & Hardware Safety Telemetry):** Authored `submission/code/train.py`, `plot_curves.py`, and execution workflows, constructing the automated hardware compute detection system, mixed-precision (`torch.cuda.amp`) training loop, in-loop dynamic VRAM safeguards (halving batch size if memory exceeds 90% of 8 GB), GPU telemetry logging via `nvidia-smi`, AdamW optimizer with Cosine Annealing, and checkpointing logic.
- **Member 4 (Evaluation, Statistical Testing & Paper Conversion):** Authored `submission/code/eda.py` and `submission/code/evaluate.py`, implementing Cohen’s Quadratic Weighted Kappa (QWK), Mean Absolute Error (MAE), 1,000-iteration paired per-class bootstrap confidence intervals (specifically for Severe NPDR and Proliferative DR), paired Wilcoxon signed-rank tests, 4-model confusion matrix and error distance plots, and the `python-docx` IEEE publishing pipeline.

---

## 6. Challenges Encountered and Engineering Resolutions

### 6.1 Paper-Count Corpus Adjustment (8 to 7 Papers)
- **Challenge:** Standard project assignment specifications typically assume an even four-pair split across eight papers. The assigned corpus contained exactly seven hardcoded papers.
- **Resolution:** Rather than fabricating an artificial eighth paper (which would violate academic integrity and grounding constraints), Member 4 conducted a solo deep-dive review of Paper 7 (Manoj & Bhosale [7]). In `summary_paper_7.md`, the "Cross-Paper Synthesis" section was replaced with an expanded "Relevance to Our Thesis and Clinical Decision Support" analysis of identical analytical depth, explicitly documenting this structural adjustment.

### 6.2 Compute Detection, AMP Optimization, and Hardware Safety Safeguards
- **Challenge:** Deep learning training on high-resolution medical images requires significant memory bandwidth and compute. Running on a mobile RTX 5050 GPU (8 GB VRAM) with tight thermal envelopes risked CUDA Out-Of-Memory (OOM) crashes or hardware instability under unmanaged batch processing.
- **Resolution:** We engineered an automated compute-detection routine and strict hardware safeguards in `train.py`. The script queries `torch.cuda.is_available()`. If a dedicated GPU is detected, it logs the GPU model, activates mixed precision training (`torch.cuda.amp.autocast` + `GradScaler`) to reduce memory footprints by 50%, caps the batch size at 16 (with a maximum threshold of 24), and samples `torch.cuda.memory_allocated()` at every step. If VRAM usage exceeds 90% of the 8 GB budget (7.2 GB), the loop automatically halves the batch size and restores from the latest checkpoint. Furthermore, we logged GPU temperature and utilization via `nvidia-smi` at every epoch, confirming stable thermal operation between 43°C and 60°C and VRAM usage under 540 MiB.

### 6.3 Resolution of CORAL Severe Class Collapse via CORN and Soft-QWK
- **Challenge:** Initial experiments with Variant B (CORAL) exhibited an acute threshold breakdown on the minority Severe NPDR class (only 135 training images), yielding a Severe F1-score of 0.0000 and an unacceptable 26.4% catastrophic error rate ($|y - \hat{y}| \ge 2$). Because CORAL assumes independent binary rank thresholds, uncalibrated gradients from the extreme imbalance caused threshold violation.
- **Resolution:** We transitioned to CORN (Conditional Ordinal Regression for Neural Networks), which factors the cumulative probability into conditional conditional probabilities $P(y > k \mid y > k-1)$. We coupled this with Cui et al. effective number of samples class-balanced weighting ($\beta = 0.9999$), `WeightedRandomSampler` at the DataLoader level, a differentiable Soft-QWK auxiliary loss ($\lambda = 0.20$), early backbone feature freezing (`features[:4]`), and 4-way Test-Time Augmentation (TTA). This architectural overhaul recovered Severe NPDR sensitivity from 0.0% to 65.7%, slashed catastrophic triage errors from 26.4% down to 5.1% (an 80.7% relative reduction), and boosted QWK from 0.7273 to 0.8482.

### 6.4 Automated Dataset Access and Label Verification
- **Challenge:** Downloading APTOS 2019 directly from Kaggle CLI requires local API authentication keys (`kaggle.json`) which were not pre-configured on the system.
- **Resolution:** We utilized a verified public Hugging Face mirror of the APTOS 2019 dataset (`bumbledeep/aptos`). In doing so, we detected an alphabetical label encoding artifact in the raw parquet file (`mild_retinopathy` encoded as 0, `no_diabetic_retinopathy` as 2). Through programmatic inspection, we corrected the mapping to strictly match the International Clinical Diabetic Retinopathy (ICDR) standard (0: No DR [1,805], 1: Mild [370], 2: Moderate [999], 3: Severe [193], 4: Proliferative [295], total 3,662). We also documented the official Kaggle CLI command in `./data/aptos2019/kaggle_command.txt`.

### 6.5 Document Format Conversion and Pandoc Absence
- **Challenge:** The course requires a formatted Microsoft Word (`.docx`) submission preserving tables, figures, and IEEE styling, but `pandoc` was not pre-installed in the local environment.
- **Resolution:** We engineered a dedicated Python conversion script using `python-docx` (`convert_to_docx.py`) that directly reads `final_paper.md`, parses markdown headings, paragraphs, bullet points, callout blocks, mathematical formulas, and embedded publication-grade PNG figures from `./submission/results/`, generating an IEEE-formatted DOCX file with exact margins, table borders, and caption styles.

### 6.6 Maximizing Raw Classification Accuracy via Regularization, Ensembling, and Logit Adjustment
- **Challenge:** While ordinal formulations (CORN) and class-balanced weighting significantly elevate minority-class sensitivity and Quadratic Weighted Kappa, they inherently trade off majority-class accuracy (dropping raw accuracy from 80.18% down to 69.09%). Certain clinical and telemedicine workflows specifically demand high raw classification accuracy ($\ge 80\%$) and high specificity on the healthy majority to minimize unnecessary secondary hospital visits.
- **Resolution:** We engineered an accuracy-focused optimization pipeline:
  1. Replaced hard cross-entropy with regularized label smoothing ($\alpha = 0.08$) to prevent overconfident boundary over-parameterization.
  2. Implemented "safe-light" data augmentation ($\pm 15^\circ$ small-angle rotation, $0.92\text{--}1.08$ zoom, mild $0.10$ color jitter) without synthetic SMOTE artifacts.
  3. Increased training budget to 12 epochs with early stopping governed strictly by validation accuracy.
  4. Sequentially trained 3 independent models under random seeds ($42, 43, 44$) on the RTX 5050 GPU, averaging their softmax probability vectors at test inference alongside 4-way flip Test-Time Augmentation (TTA).
  5. Implemented post-hoc logit adjustment (Menon et al., 2021) to map the continuous accuracy vs. sensitivity trade-off across $\tau \in \{0.0, 0.5, 1.0, 1.5\}$.
  This configuration achieved an overall classification accuracy of **83.82%** (+3.64% absolute increase over baseline), peak QWK of **0.8859**, and a project-low MAE of **0.2200**.

---

## 7. Reproducibility Protocol
To guarantee that all observed performance differentials (particularly the primary metric $\Delta \text{QWK}$) are strictly attributable to the mathematical loss formulation and sampling strategy rather than stochastic noise, we enforced the following reproducibility constraints:
1. **Global Deterministic Seed:** A global seed of `42` was locked across Python’s `random`, NumPy (`np.random.seed(42)`), PyTorch (`torch.manual_seed(42)`), CUDA (`torch.cuda.manual_seed_all(42)`), and CuDNN (`torch.backends.cudnn.deterministic = True`).
2. **Identical Data Partitioning:** The stratified 70% train (2,563 images), 15% validation (549 images), and 15% test (550 images) splits were computed once under seed 42 and saved to disk. All four variants trained and evaluated on identical image subsets.
3. **Strictly Controlled Model Capacity:** EfficientNet-B0 was preserved as the uniform backbone across all experiments. Model scaling to B3/B4/ViT was deliberately avoided to prevent catastrophic overfitting on the small minority classes (~135 severe training samples).
4. **Transparent Statistical Testing:** 1,000 paired bootstrap iterations were evaluated to produce 95% confidence intervals across all primary metrics and per-class recall/F1 scores, alongside paired Wilcoxon signed-rank tests for pairwise significance.
