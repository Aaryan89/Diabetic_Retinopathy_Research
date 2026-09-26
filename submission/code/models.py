"""
Model Architectures for Diabetic Retinopathy Detection:
Identical Backbone (EfficientNet-B0 / ResNet18 fallback) across all variants.

Variant A: Baseline Multiclass Softmax (5 classes) + Cross-Entropy Loss
Variant B: Ordinal Regression via CORAL (Consistent Rank Logits, 4 rank thresholds)
Variant C: Continuous Regression (Single continuous output) + Smooth L1 Loss
Variant CORN: Conditional Ordinal Regression for Neural Networks (CORN) with
              Class-Balanced Weights, Soft-QWK Loss, and Test-Time Augmentation (TTA).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CoralLayer(nn.Module):
    """
    CORAL (Consistent Rank Logits) output layer for ordinal regression.
    Decomposes K-class ordinal target into K-1 binary classification tasks
    sharing a single linear projection weight vector with task-specific bias thresholds.
    """
    def __init__(self, in_features, num_classes=5):
        super(CoralLayer, self).__init__()
        self.num_classes = num_classes
        # Single weight vector shared across rank thresholds
        self.linear = nn.Linear(in_features, 1, bias=False)
        # K-1 independent bias parameters
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x):
        # logits shape: (batch_size, num_classes - 1)
        logits = self.linear(x) + self.bias
        return logits

class CornLayer(nn.Module):
    """
    CORN (Conditional Ordinal Regression for Neural Networks) output layer.
    Outputs K-1 conditional logits: g_k(x) = logit(P(y > k | y > k-1)).
    Unlike CORAL, each rank threshold is conditioned on the previous one being satisfied,
    preventing non-informative gradients from dominant lower classes (Grade 0) on
    sparse upper-grade thresholds (Grade 3/4).
    """
    def __init__(self, in_features, num_classes=5):
        super(CornLayer, self).__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(in_features, num_classes - 1)

    def forward(self, x):
        # logits shape: (batch_size, num_classes - 1)
        return self.linear(x)

def get_backbone(backbone_name='efficientnet_b0', pretrained=True, freeze_early_blocks=False):
    """
    Instantiates standard pretrained vision backbone and returns (feature_extractor, in_features).
    Optionally freezes early low-level feature extraction stages to prevent overfitting on small datasets.
    """
    if backbone_name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        if freeze_early_blocks:
            # Freeze stem (features[0]) and first 3 MBConv stages (features[1..3])
            # Keeps generic low-level retinal edge/texture filters stable
            for stage in model.features[:4]:
                for param in stage.parameters():
                    param.requires_grad = False
        in_features = model.classifier[1].in_features
        # Replace classifier with identity to expose feature vector
        model.classifier = nn.Identity()
        return model, in_features
    elif backbone_name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        if freeze_early_blocks:
            # Freeze conv1, bn1, layer1
            for module in [model.conv1, model.bn1, model.layer1]:
                for param in module.parameters():
                    param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Identity()
        return model, in_features
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

class DRModel(nn.Module):
    """
    Unified Diabetic Retinopathy architecture supporting all experimental variants:
    Variant A: Softmax Baseline
    Variant B: CORAL Ordinal Regression
    Variant C: Continuous Regression
    Variant CORN: Conditional Ordinal Regression with CORN Head
    """
    def __init__(self, variant='variant_A', backbone_name='efficientnet_b0', pretrained=True,
                 freeze_early_blocks=False, num_classes=5):
        super(DRModel, self).__init__()
        self.variant = variant
        self.num_classes = num_classes
        self.backbone, in_features = get_backbone(
            backbone_name, pretrained=pretrained, freeze_early_blocks=freeze_early_blocks
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.3)

        if variant == 'variant_A':
            # Variant A: Standard Multiclass Softmax Classification
            self.head = nn.Linear(in_features, num_classes)
        elif variant == 'variant_B':
            # Variant B: CORAL Ordinal Regression (K-1 binary threshold logits)
            self.head = CoralLayer(in_features, num_classes=num_classes)
        elif variant == 'variant_C':
            # Variant C: Continuous Scalar Regression
            self.head = nn.Linear(in_features, 1)
        elif variant in ['variant_CORN', 'variant_corn']:
            # Variant CORN: Conditional Ordinal Regression Head
            self.head = CornLayer(in_features, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown variant: {variant}")

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        out = self.head(features)
        return out

    def predict(self, x, use_tta=False):
        """
        Inference prediction returning (predicted_class_label, raw_outputs).
        Supports Test-Time Augmentation (TTA) via horizontal and vertical flips.
        """
        self.eval()
        with torch.no_grad():
            if use_tta:
                # 4-way Test-Time Augmentation: Original, H-Flip, V-Flip, HV-Flip
                transforms_list = [
                    lambda t: t,
                    lambda t: torch.flip(t, dims=[3]),
                    lambda t: torch.flip(t, dims=[2]),
                    lambda t: torch.flip(t, dims=[2, 3])
                ]
                raw_outs = [self.forward(fn(x)) for fn in transforms_list]
                raw_out = torch.stack(raw_outs, dim=0).mean(dim=0)
            else:
                raw_out = self.forward(x)

            if self.variant == 'variant_A':
                # Softmax probabilities -> argmax
                probs = torch.softmax(raw_out, dim=1)
                preds = torch.argmax(probs, dim=1)
                return preds, raw_out
            elif self.variant == 'variant_B':
                # Sigmoid probabilities over K-1 binary thresholds -> sum(prob > 0.5)
                sigmoids = torch.sigmoid(raw_out)
                preds = torch.sum(sigmoids > 0.5, dim=1)
                return preds, raw_out
            elif self.variant == 'variant_C':
                # Continuous scalar -> clamp to [0, num_classes-1] -> round to nearest integer
                clamped = torch.clamp(raw_out.squeeze(-1), 0.0, float(self.num_classes - 1))
                preds = torch.round(clamped).long()
                return preds, raw_out
            elif self.variant in ['variant_CORN', 'variant_corn']:
                # CORN: cumulative conditional probabilities -> sum(cum_prob > 0.5)
                cond_probs = torch.sigmoid(raw_out)
                cum_probs = torch.cumprod(cond_probs, dim=1)
                preds = torch.sum(cum_probs > 0.5, dim=1)
                return preds, raw_out

def coral_loss(logits, targets, num_classes=5):
    """
    Computes CORAL loss (sum of binary cross-entropies for ordinal rank indicators).
    """
    batch_size = targets.size(0)
    device = targets.device
    levels = torch.zeros(batch_size, num_classes - 1, device=device)
    for i in range(num_classes - 1):
        levels[:, i] = (targets > i).float()

    loss = F.binary_cross_entropy_with_logits(logits, levels, reduction='mean')
    return loss

def corn_loss(logits, targets, num_classes=5, class_weights=None):
    """
    Conditional Ordinal Regression (CORN) loss function.
    For an instance with label y:
      - For task k < y: target is 1 (condition y > k-1 was met and y > k holds)
      - For task k == y (if y < K-1): target is 0 (condition y > k-1 was met, but y <= k)
      - For task k > y: condition y > k-1 was NOT met; task is masked out.
    Class weights (Cui et al. effective number weights) apply per-sample based on y.
    """
    batch_size = targets.size(0)
    device = targets.device
    num_tasks = num_classes - 1

    # Task indices: shape (1, num_tasks)
    task_idx = torch.arange(num_tasks, device=device).unsqueeze(0).expand(batch_size, -1)
    # Target expanded: shape (batch_size, num_tasks)
    y_exp = targets.unsqueeze(1).expand(-1, num_tasks)

    # Active mask: sample i participates in task k if and only if k <= y_i
    mask = (task_idx <= y_exp).float()
    # Binary targets: 1 if y_i > k else 0
    binary_targets = (task_idx < y_exp).float()

    # Binary cross-entropy with logits across tasks
    bce = F.binary_cross_entropy_with_logits(logits, binary_targets, reduction='none')
    # Mean loss across active tasks for each sample
    sample_loss = (bce * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

    if class_weights is not None:
        cw = class_weights.to(device)
        sample_weights = cw[targets]
        sample_loss = sample_loss * sample_weights

    return sample_loss.mean()

def corn_predict_probs(logits):
    """
    Computes class probability distribution P(y = k) for k in {0, ..., K-1}
    from CORN conditional logits g_k:
      q_k = sigmoid(g_k) = P(y > k | y > k-1)
      P(y > k) = prod_{j=0}^k q_j
      P(y = 0) = 1 - P(y > 0)
      P(y = k) = P(y > k-1) - P(y > k)
      P(y = K-1) = P(y > K-2)
    """
    cond_probs = torch.sigmoid(logits) # (B, K-1)
    cum_probs = torch.cumprod(cond_probs, dim=1) # (B, K-1)

    p0 = 1.0 - cum_probs[:, 0:1] # (B, 1)
    middle = cum_probs[:, :-1] - cum_probs[:, 1:] # (B, K-2)
    p_last = cum_probs[:, -1:] # (B, 1)

    class_probs = torch.cat([p0, middle, p_last], dim=1) # (B, K)
    # Numerical stability
    class_probs = torch.clamp(class_probs, min=1e-8, max=1.0)
    class_probs = class_probs / class_probs.sum(dim=1, keepdim=True)
    return class_probs, cum_probs

class SoftQWKLoss(nn.Module):
    """
    Differentiable Soft Quadratic Weighted Kappa (QWK) Loss.
    Directly aligns backpropagation with the primary clinical evaluation metric.
    Minimizes 1 - QWK = (W * O) / (W * E + eps).
    """
    def __init__(self, num_classes=5, eps=1e-7):
        super(SoftQWKLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
        # Quadratic penalty matrix W_ij = (i - j)^2 / (K - 1)^2
        w = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                w[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
        self.register_buffer('weight_mat', w)

    def forward(self, pred_probs, targets):
        """
        pred_probs: (B, num_classes) soft probabilities summing to 1
        targets: (B,) integer ground truth labels
        """
        device = pred_probs.device
        batch_size = pred_probs.size(0)

        # One-hot encoded true targets: (B, num_classes)
        y_onehot = F.one_hot(targets, num_classes=self.num_classes).float().to(device)

        # Observed error matrix O = Y^T P / B
        O = torch.matmul(y_onehot.t(), pred_probs) / float(batch_size)

        # Marginal distributions
        hist_true = y_onehot.sum(dim=0, keepdim=True) / float(batch_size)
        hist_pred = pred_probs.sum(dim=0, keepdim=True) / float(batch_size)
        # Expected error matrix under independence E = hist_true^T * hist_pred
        E = torch.matmul(hist_true.t(), hist_pred)

        W = self.weight_mat.to(device)
        num = (W * O).sum()
        den = (W * E).sum()

        qwk = 1.0 - (num / (den + self.eps))
        loss = 1.0 - qwk
        return loss
