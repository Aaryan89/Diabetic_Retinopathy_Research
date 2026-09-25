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
| **Day 2: Morning** | 09:00 – 13:30 | **Model Implementation & Controlled Training:** Implemented `models.py` and `train.py`. Verified compute environment (NVIDIA GPU / CPU fallback detection). Executed strictly uniform fine-tuning across Variant A (Softmax), Variant B (CORAL Ordinal), and Variant C (Continuous Regression) under fixed global seed (42). Saved checkpoints, loss curves, and prediction CSVs. |
| **Day 2: Afternoon** | 14:30 – 19:00 | **Evaluation, Viva Prep & Final Paper Publishing:** Executed `evaluate.py` to compute QWK, MAE, confusion matrices, error histograms, and paired bootstrap significance tests. Authored `methodology_followed.md`, `talking_points.md`, and the comprehensive IEEE-style `final_paper.md`. Successfully compiled `final_paper.docx` preserving all figures and tables. |

---

## 5. Division of Labor for the Technical Build
To ensure high velocity and accountability across the implementation lifecycle, technical responsibilities were distributed among team members:
- **Member 1 (Preprocessing & Data Pipeline):** Led `submission/code/data.py`, implementing the automated circular crop bounding-box extraction, LAB-space CLAHE contrast enhancement algorithm, and the PyTorch `APTOSDataset` / `DataLoader` pipeline with stratified 70/15/15 partitioning.
- **Member 2 (Architecture & Loss Function Engineering):** Authored `submission/code/models.py`, implementing the shared backbone integration, the CORAL ordinal regression layer ($K-1$ rank thresholds with shared projection weights), the continuous regression head, and the exact mathematical CORAL loss function.
- **Member 3 (Training Pipeline & Compute Fallback System):** Authored `submission/code/train.py` and `run_all.py`, constructing the automated hardware compute detection system, AdamW optimizer with Cosine Annealing learning rate scheduling, deterministic seed locking, and checkpointing logic.
- **Member 4 (Evaluation, Statistical Testing & Paper Conversion):** Authored `submission/code/eda.py` and `submission/code/evaluate.py`, implementing Cohen’s Quadratic Weighted Kappa (QWK), Mean Absolute Error (MAE), 1,000-iteration bootstrap significance testing, Wilcoxon signed-rank tests, figure generation, and the `python-docx` IEEE publishing pipeline.

---

## 6. Challenges Encountered and Engineering Resolutions

### 6.1 Paper-Count Corpus Adjustment (8 to 7 Papers)
- **Challenge:** Standard project assignment specifications typically assume an even four-pair split across eight papers. The assigned corpus contained exactly seven hardcoded papers.
- **Resolution:** Rather than fabricating an artificial eighth paper (which would violate academic integrity and grounding constraints), Member 4 conducted a solo deep-dive review of Paper 7 (Manoj & Bhosale [7]). In `summary_paper_7.md`, the "Cross-Paper Synthesis" section was replaced with an expanded "Relevance to Our Thesis and Clinical Decision Support" analysis of identical analytical depth, explicitly documenting this structural adjustment.

### 6.2 Compute Detection and Hardware Fallback Protocol
- **Challenge:** Deep learning training on high-resolution medical images requires significant memory bandwidth and compute. Unmanaged executions risk CUDA Out-Of-Memory (OOM) crashes or excessively slow CPU runs.
- **Resolution:** We engineered an automated compute-detection routine in `train.py` prior to model instantiation. The script queries `torch.cuda.is_available()`. If a dedicated GPU is detected, it logs the GPU model and VRAM (profiling an NVIDIA RTX GPU with ~8 GB VRAM), selects EfficientNet-B0 as the primary backbone, and sets the training budget to 8 epochs with batch size 32. If no GPU is available, the script executes an automated fallback: switching the backbone to lightweight ResNet-18, reducing epochs to 4, and adjusting batch size to 16. This adaptation is documented explicitly in the final paper's Limitations section rather than being hidden.

### 6.3 Automated Dataset Access and Label Verification
- **Challenge:** Downloading APTOS 2019 directly from Kaggle CLI requires local API authentication keys (`kaggle.json`) which were not pre-configured on the system.
- **Resolution:** We utilized a verified public Hugging Face mirror of the APTOS 2019 dataset (`bumbledeep/aptos`). In doing so, we detected an alphabetical label encoding artifact in the raw parquet file (`mild_retinopathy` encoded as 0, `no_diabetic_retinopathy` as 2). Through programmatic inspection, we corrected the mapping to strictly match the International Clinical Diabetic Retinopathy (ICDR) standard (0: No DR [1,805], 1: Mild [370], 2: Moderate [999], 3: Severe [193], 4: Proliferative [295], total 3,662). We also documented the official Kaggle CLI command in `./data/aptos2019/kaggle_command.txt`.

### 6.4 Document Format Conversion and Pandoc Absence
- **Challenge:** The course requires a formatted Microsoft Word (`.docx`) submission preserving tables, figures, and IEEE styling, but `pandoc` was not pre-installed in the local environment.
- **Resolution:** We engineered a dedicated Python conversion script using `python-docx` that directly reads `final_paper.md`, parses markdown headings, paragraphs, bullet points, callout blocks, equations, and embedded PNG figures from `./submission/results/`, generating an IEEE-formatted DOCX file with exact margins, table borders, and caption styles.

---

## 7. Reproducibility Protocol
To guarantee that all observed performance differentials (particularly the primary metric $\Delta \text{QWK}$) are strictly attributable to the mathematical loss formulation rather than stochastic noise, we enforced the following reproducibility constraints:
1. **Global Deterministic Seed:** A global seed of `42` was locked across Python’s `random`, NumPy (`np.random.seed(42)`), PyTorch (`torch.manual_seed(42)`), CUDA (`torch.cuda.manual_seed_all(42)`), and CuDNN (`torch.backends.cudnn.deterministic = True`).
2. **Identical Data Partitioning:** The stratified 70% train (2,563 images), 15% validation (549 images), and 15% test (550 images) splits were computed once under seed 42 and saved to disk. All three variants trained and evaluated on identical image subsets.
3. **Strictly Uniform Hyperparameters:** All three variants utilized the identical backbone architecture, identical pre-trained ImageNet weights, identical AdamW optimizer (`lr=3e-4`, `weight_decay=1e-2`), identical Cosine Annealing learning rate schedule, identical batch size, identical image augmentations (train only), and identical epoch counts. Only the final output head and loss function were varied.
