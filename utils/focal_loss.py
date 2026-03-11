import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        """
        alpha: weight for FAKE class (1), REAL gets 1-alpha
        gamma: focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        logits: raw output of the model (B,)
        targets: binary labels (0=REAL, 1=FAKE)
        """
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)

        # Class-wise alpha factor
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        loss = alpha_factor * (1 - pt) ** self.gamma * bce
        return loss.mean()
