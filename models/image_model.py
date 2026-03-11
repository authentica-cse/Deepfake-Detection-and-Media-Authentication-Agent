import torch.nn as nn
from torchvision import models


class ImageModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

