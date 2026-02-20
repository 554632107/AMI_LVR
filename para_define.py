import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
1. Label Smoothing: Replaces original one-hot labels with (1-ε) * one_hot + ε/K distribution via smoothing parameter (K is number of classes)
2. Regularization Effect: Prevents model from being overconfident on training data, improves generalization
3. Numerical Stability: Uses log_softmax to avoid numerical overflow
4. Type Safety: Forces target conversion to long type to ensure tensor computation compatibility
Suitable for classification tasks requiring overfitting mitigation, smoothing parameter (range 0-1) controls smoothing intensity, larger values provide stronger regularization
"""
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        target = target.long()  # Add type conversion
        log_prob = F.log_softmax(pred, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

