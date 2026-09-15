import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

from src.models.model import HemorrhageClassifier


# ==============================
# CONFIGURATION
# ==============================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "outputs/models/efficientnetv2_best.pth"

IMAGE_PATH = r"C:\Users\ASMI\OneDrive\Desktop\Intracranial_Hemorrhage_Detection\dataset\Patients_CT\057\brain\22.jpg"

OUTPUT_PATH = "outputs/plots/gradcam_result.png"

CLASS_NAMES = [
    "No_Hemorrhage",
    "Epidural",
    "Intraparenchymal",
    "Subdural",
    "Intraventricular",
    "Subarachnoid"
]


# ==============================
# LOAD MODEL
# ==============================

model = HemorrhageClassifier(num_classes=6)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

# Handle different checkpoint formats
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()


# ==============================
# PREPROCESSING
# ==============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


image = Image.open(IMAGE_PATH).convert("RGB")

input_tensor = transform(image).unsqueeze(0).to(DEVICE)


# ==============================
# GRAD-CAM
# ==============================

gradients = []
activations = []


def forward_hook(module, input, output):
    activations.append(output)


def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])


# EfficientNetV2-S final convolutional feature layer
target_layer = model.model.features[-1]

forward_handle = target_layer.register_forward_hook(forward_hook)
backward_handle = target_layer.register_full_backward_hook(backward_hook)


# ==============================
# FORWARD PASS
# ==============================

output = model(input_tensor)

probabilities = F.softmax(output, dim=1)

predicted_class = torch.argmax(probabilities, dim=1).item()

confidence = probabilities[0][predicted_class].item() * 100


print("\n==============================")
print("GRAD-CAM RESULT")
print("==============================")

print("Predicted Class :", CLASS_NAMES[predicted_class])
print("Confidence      :", f"{confidence:.2f}%")

print("\nClass Probabilities:")

for i, name in enumerate(CLASS_NAMES):
    print(
        f"{name:20s}: "
        f"{probabilities[0][i].item() * 100:.2f}%"
    )


# ==============================
# BACKWARD PASS
# ==============================

model.zero_grad()

score = output[0, predicted_class]

score.backward()


# ==============================
# CREATE CAM
# ==============================

activation = activations[0]
gradient = gradients[0]

weights = gradient.mean(
    dim=(2, 3),
    keepdim=True
)

cam = (weights * activation).sum(dim=1)

cam = F.relu(cam)

cam = cam.squeeze().detach().cpu().numpy()

cam = cv2.resize(
    cam,
    (224, 224)
)

# Normalize CAM
cam = cam - cam.min()

if cam.max() != 0:
    cam = cam / cam.max()


# ==============================
# CREATE HEATMAP
# ==============================

heatmap = np.uint8(255 * cam)

heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)


# ==============================
# ORIGINAL IMAGE
# ==============================

original = np.array(
    image.resize((224, 224))
)

original = cv2.cvtColor(
    original,
    cv2.COLOR_RGB2BGR
)


# ==============================
# OVERLAY
# ==============================

overlay = cv2.addWeighted(
    original,
    0.6,
    heatmap,
    0.4,
    0
)


# ==============================
# SAVE RESULT
# ==============================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

# Add labels
cv2.putText(
    overlay,
    f"{CLASS_NAMES[predicted_class]} ({confidence:.2f}%)",
    (10, 25),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
)

cv2.imwrite(
    OUTPUT_PATH,
    overlay
)


# ==============================
# CLEANUP
# ==============================

forward_handle.remove()
backward_handle.remove()


print("\nGrad-CAM saved successfully!")
print("Output:", OUTPUT_PATH)