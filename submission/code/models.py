"""
Model Architectures for Diabetic Retinopathy Detection:
Identical Backbone (EfficientNet-B0 / ResNet18 fallback) across all 3 variants.

Variant A: Baseline Multiclass Softmax (5 classes) + Cross-Entropy Loss
Variant B: Ordinal Regression via CORAL (Consistent Rank Logits, 4 rank thresholds)
Variant C: Continuous Regression (Single continuous output) + Smooth L1 Loss
"""

import torch
import torch.nn as nn
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

def get_backbone(backbone_name='efficientnet_b0', pretrained=True):
    """
    Instantiates standard pretrained vision backbone and returns (feature_extractor, in_features).
    """
    if backbone_name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        # Replace classifier with identity to expose feature vector
        model.classifier = nn.Identity()
        return model, in_features
    elif backbone_name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
        return model, in_features
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

class DRModel(nn.Module):
    """
    Unified Diabetic Retinopathy architecture supporting all three experimental variants.
    """
    def __init__(self, variant='variant_A', backbone_name='efficientnet_b0', pretrained=True, num_classes=5):
        super(DRModel, self).__init__()
        self.variant = variant
        self.num_classes = num_classes
        self.backbone, in_features = get_backbone(backbone_name, pretrained=pretrained)

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
        else:
            raise ValueError(f"Unknown variant: {variant}")

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        out = self.head(features)
        return out

    def predict(self, x):
        """
        Inference prediction returning (predicted_class_label, raw_outputs).
        """
        self.eval()
        with torch.no_grad():
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

def coral_loss(logits, targets, num_classes=5):
    """
    Computes CORAL loss (sum of binary cross-entropies for ordinal rank indicators).
    """
    # targets shape: (batch_size,)
    # Convert targets to binary rank indicators: shape (batch_size, num_classes - 1)
    batch_size = targets.size(0)
    device = targets.device
    levels = torch.zeros(batch_size, num_classes - 1, device=device)
    for i in range(num_classes - 1):
        levels[:, i] = (targets > i).float()

    loss = nn.functional.binary_cross_entropy_with_logits(logits, levels, reduction='mean')
    return loss
