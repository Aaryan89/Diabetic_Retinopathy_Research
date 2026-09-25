# Literature Review: Papers 1 and 2

**Reviewer Assignment:** Member 1  
**Scope:** Deep Transfer Learning Benchmarks and Dual-Branch Ensembles for Diabetic Retinopathy Detection  
**Target Citation Style:** IEEE (Numeric, bracketed, corresponding to standard conference transactions format)  
**Calibrated Section Depth:** ~500–700 words total, matching the literature review depth observed in `sample_report.pdf`.

---

## 1. Paper 1: Deep Transfer Learning Models for Medical Diabetic Retinopathy Detection

### 1.1 Full Citation (IEEE Style)
[1] N. E. M. Khalifa, M. Loey, M. H. N. Taha, and H. N. E. T. Mohamed, "Deep transfer learning models for medical diabetic retinopathy detection," *Acta Informatica Medica*, vol. 27, no. 5, pp. 327–332, Dec. 2019, doi: 10.5455/aim.2019.27.327-332.

### 1.2 Problem Addressed
Diabetic Retinopathy (DR) represents the primary microvascular ocular complication arising from chronic diabetes mellitus and a foremost cause of irreversible vision loss among working-age populations worldwide. Timely clinical diagnosis during the early non-proliferative stages is paramount to allow laser photocoagulation or pharmacotherapy before progression to irreversible proliferative diabetic retinopathy. However, manual ophthalmoscopy screening is labor-intensive, costly, and subject to inter-observer variability. The authors investigate whether lightweight pre-trained deep convolutional neural networks (CNNs) can reliably automate five-stage severity detection using newly released fundus photography.

### 1.3 Method and Architecture Summary
The authors evaluate six pre-trained transfer learning architectures: AlexNet (8 weighted layers), ResNet-18 (18 layers), SqueezeNet (18 layers), GoogleNet (22 layers), VGG-16 (16 layers), and VGG-19 (19 layers). These networks were deliberately chosen over heavier models (e.g., DenseNet-201 or Inception-ResNet) to minimize parameter counts, training latency, and edge compute complexity. The output dense layer of each backbone was replaced with a 5-unit fully connected layer equipped with a standard categorical softmax activation function. To counter overfitting, the authors applied offline geometric and color data augmentation—including rotation, scaling, and horizontal reflection—multiplying the dataset size fourfold prior to training. Optimization was carried out using stochastic gradient descent under categorical cross-entropy loss.

### 1.4 Dataset(s) and Metrics Used (Actual Reported Numbers)
- **Dataset:** Asia Pacific Tele-Ophthalmology Society (APTOS) 2019 Blindness Detection dataset, comprising 3,662 retinal fundus images across five severity grades:
  - Class 0 (No DR): 1,805 images
  - Class 1 (Mild NPDR): 370 images
  - Class 2 (Moderate NPDR): 999 images
  - Class 3 (Severe NPDR): 193 images
  - Class 4 (Proliferative DR): 295 images
- **Reported Performance Numbers:**
  - *Dataset expansion via augmentation:* 14,648 images.
  - *Overall Classification Accuracy:*
    - AlexNet: **97.9%** (highest overall accuracy)
    - VGG-16: **97.8%**
    - VGG-19: **97.4%**
    - ResNet-18: **96.8%**
    - GoogleNet: **96.3%**
    - SqueezeNet: **90.3%**
  - *Macro Evaluation Metrics (AlexNet):* Precision = **96.23%**, Recall = **95.42%**, F1-Score = **95.82%**.
  - *Per-Class Accuracy (AlexNet):* Class 0 = 99.7%, Class 1 = 98.0%, Class 2 = 96.6%, Class 3 = 91.3%, Class 4 = 95.8%.
  - *Per-Class Accuracy (SqueezeNet):* Class 0 = 97.8%, Class 1 = 80.0%, Class 2 = 87.5%, Class 3 = 67.8%, Class 4 = 80.9%.

### 1.5 Limitations and Gaps
- **Explicitly Acknowledged by Authors:** The investigation was intentionally restricted to lightweight, shallow network topologies; deeper modern backbones and multi-modal patient demographic inputs (e.g., HbA1c levels, systemic blood pressure) were excluded. Furthermore, the paper explicitly notes that Class 3 (Severe NPDR) suffered the lowest classification accuracy across all architectures (e.g., falling to 67.8% in SqueezeNet and 88.8% in VGG-19).
- **Identified Gap (Our Analysis):** The fourfold offline data augmentation was executed across the aggregate dataset *prior* to data partitioning. This methodology introduces significant data leakage across training and evaluation splits, as geometric transformations of identical patient eyes contaminated both partitions, thereby explaining the unrealistically elevated 97.9% multi-class accuracy for AlexNet. More critically, the classification was formulated as a nominal multi-class problem under standard softmax categorical cross-entropy. The model penalizes an off-by-one boundary mistake (e.g., Grade 1 vs. Grade 2) identically to a catastrophic off-by-four misdiagnosis (e.g., Grade 0 misclassified as Grade 4), ignoring clinical ordinal geometry entirely.

### 1.6 Relevance to Our Ordinal Regression Thesis
Khalifa et al. treat the 5 DR severity grades as independent unordered categories, which conceals dangerous clinical triage errors behind an unweighted accuracy metric. Because biological DR progression is strictly continuous and ordered, our project's shift to ordinal regression (CORAL/CORN) directly resolves this gap by penalizing large inter-grade misclassifications proportionally to their ordinal distance, ensuring severe NPDR cases are not misrouted as healthy.

---

## 2. Paper 2: Dual Branch Deep Learning Network for Detection and Stage Grading of Diabetic Retinopathy

### 2.1 Full Citation (IEEE Style)
[2] H. Shakibania, S. Raoufi, B. Pourafkham, H. Khotanlou, and M. Mansoorizadeh, "Dual branch deep learning network for detection and stage grading of diabetic retinopathy," arXiv preprint arXiv:2308.09945, Aug. 2023.

### 2.2 Problem Addressed
Accurate detection and five-tier stage grading of DR from single fundus images remains hindered by severe class distribution skewness in clinical datasets and high inter-observer grading variability among expert ophthalmologists. In particular, existing automated architectures falter severely when distinguishing between contiguous advanced severity tiers—specifically Severe NPDR (Grade 3) and Proliferative DR (Grade 4)—where subtle vascular lesions dictate urgent surgical referral versus routine monitoring.

### 2.3 Method and Architecture Summary
The authors introduce a dual-branch convolutional architecture integrating two pre-trained feature extractors: ResNet-50 and EfficientNet-B0. The parallel streams extract complementary hierarchical visual representations: ResNet-50 captures high-level structural semantics via residual skip connections, while EfficientNet-B0 captures fine-grained multi-scale lesions via compound depth-width-resolution scaling. The resulting feature tensors are globally pooled, concatenated into a unified representation, and classified via multi-layer perceptron heads. To counter class imbalance, the network is trained using Complement Cross Entropy (CCE) loss, which specifically neutralizes majority-class probability inflation and emphasizes minority-class gradients. The authors evaluate training on APTOS 2019 alone and on a selectively merged multi-center dataset incorporating supplementary images from Messidor-2 and IDRiD.

### 2.4 Dataset(s) and Metrics Used (Actual Reported Numbers)
- **Datasets:** APTOS 2019 (3,662 images: 70% train [2,567], 10% validation [364], 20% test [731]), enriched with 1,744 images from Messidor-2 and 516 images from IDRiD.
- **Reported Performance Numbers:**
  - *APTOS 2019 Standalone Evaluation (with augmentations, Table 2):*
    - Binary Classification: Accuracy = **98.0%**, QWK = **95.8%**, Sensitivity = **99.2%**, Specificity = **96.9%**.
    - Multi-Class Grading: Accuracy = **88.1%**, QWK = **91.8%**, Sensitivity = **88.1%**, Specificity = **97.4%**.
  - *Selectively Merged Dataset Evaluation (Table 4):*
    - Multi-Class Overall: Accuracy = **89.60%**, Quadratic Weighted Kappa (QWK) = **93.00%**, Precision = **89.23%**, Sensitivity = **89.60%**, Specificity = **97.72%**, F1-Score = **89.15%**.
  - *Per-Class Sensitivity Breakdown (Table 4 Multi-Class):*
    - Grade 0 (No DR): Sensitivity = **98.24%** (Precision: 99.44%)
    - Grade 1 (Mild): Sensitivity = **97.30%** (Precision: 77.42%)
    - Grade 2 (Moderate): Sensitivity = **88.95%** (Precision: 86.34%)
    - Grade 3 (Severe): Sensitivity = **36.84%** (Precision: 58.33%, F1: 45.16%)
    - Grade 4 (PDR): Sensitivity = **62.71%** (Precision: 71.15%, F1: 66.66%)

### 2.5 Limitations and Gaps
- **Explicitly Acknowledged by Authors:** The paper acknowledges that despite the dual-branch fusion, the model struggled significantly to discriminate between severe non-proliferative DR (Grade 3) and proliferative DR (Grade 4), yielding an alarmingly low sensitivity of 36.84% for Grade 3. The authors attribute this to lack of depth-resolved optical coherence tomography (OCT) cross-sections and inadequate sample counts in the severe cohorts.
- **Identified Gap (Our Analysis):** While the authors rightly track Quadratic Weighted Kappa (QWK) as their target metric during evaluation, their training loss (CCE) remains fundamentally nominal. Complement Cross Entropy penalizes all incorrect classes without encoding the natural ordinal penalty structure that QWK assesses. Consequently, the network cannot learn that predicting Grade 1 for a Grade 3 lesion is clinically catastrophic compared to predicting Grade 2. Furthermore, the dual-branch framework incurs double the parameter footprint and inference latency, impeding lightweight edge deployment without solving the underlying ordinal boundary collapse.

### 2.6 Relevance to Our Ordinal Regression Thesis
Shakibania et al.'s collapse in Grade 3 sensitivity (36.84%) underscores the critical shortcoming of optimizing nominal classification objectives on graded medical stages. Rather than doubling architectural complexity via parallel feature backbones, our ordinal regression paradigm directly reformulates the loss function into rank-ordered binary thresholds (CORAL/CORN), enforcing ordinal consistency without adding inference overhead.

---

## 3. Cross-Paper Synthesis
Both Khalifa et al. [1] and Shakibania et al. [2] benchmark transfer learning pipelines on the APTOS 2019 dataset, yet both highlight the persistent failure of standard classification setups to reliably detect Grade 3 Severe NPDR (dropping to 67.8% in [1] and 36.84% in [2]). Whereas Khalifa et al. demonstrate that lightweight architectures (AlexNet/ResNet-18) achieve high apparent accuracy under data expansion, their nominal cross-entropy framing obscures severe multi-stage clinical errors. Shakibania et al. attempt to overcome this by introducing dual-branch backbones and Complement Cross Entropy, yet their objective still fails to model disease ordinality, resulting in severe boundary degradation between adjacent clinical referral classes. Together, these two studies demonstrate that simply increasing model capacity or applying heuristic minority weighting cannot substitute for an ordinal-aware loss formulation that explicitly penalizes predictions in proportion to their distance from the true clinical grade.
