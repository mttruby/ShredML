import torch.nn as nn
import torch
from torchvision.models import resnet18, ResNet18_Weights 


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.out_dim = 512

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x