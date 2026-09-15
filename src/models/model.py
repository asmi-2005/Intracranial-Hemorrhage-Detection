import torch.nn as nn
from torchvision.models import efficientnet_v2_s


class HemorrhageClassifier(nn.Module):

    def __init__(self, num_classes=6):
        super().__init__()

        self.model = efficientnet_v2_s(weights="DEFAULT")

        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)