import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights
class FreqModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        # Replace classifier for binary classification
        self.backbone.classifier[1] = nn.Linear(
            self.backbone.classifier[1].in_features, 2
        )

    def forward(self, x):
        return self.backbone(x)

