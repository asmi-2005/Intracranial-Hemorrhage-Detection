import torch
import torchvision

from torchvision.models import efficientnet_v2_s

print("PyTorch Version:", torch.__version__)
print("Torchvision Version:", torchvision.__version__)

model = efficientnet_v2_s(weights="DEFAULT")

print(model)