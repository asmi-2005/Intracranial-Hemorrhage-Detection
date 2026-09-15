import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
from PIL import Image
from pathlib import Path

# ==========================================================
# CHANGE THIS TO ANY CT IMAGE FROM YOUR DATASET
# ==========================================================

image_path = Path(r"C:\Users\ASMI\OneDrive\Desktop\Intracranial_Hemorrhage_Detection\dataset\Patients_CT\057\brain\22.jpg")      # <-- change this

# ==========================================================
# Load Image
# ==========================================================

image = Image.open(image_path).convert("RGB")

# ==========================================================
# Transformations
# ==========================================================

resize = transforms.Resize((224, 224))

augmentation = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(30),
    transforms.RandomAffine(
        degrees=20,
        translate=(0.1,0.1),
        scale=(0.9,1.1)
    )
])

normalization = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ==========================================================
# Apply Transformations
# ==========================================================

original = image

resized = resize(image)

augmented = augmentation(image)

normalized = normalization(image)

# ==========================================================
# De-normalize for Visualization
# ==========================================================

mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

normalized = normalized * std + mean
normalized = normalized.clamp(0,1)

normalized = normalized.permute(1,2,0).numpy()

# ==========================================================
# Plot
# ==========================================================

fig, ax = plt.subplots(1,4, figsize=(16,4))

ax[0].imshow(original)
ax[0].set_title("Original CT")

ax[1].imshow(resized)
ax[1].set_title("Resized (224×224)")

ax[2].imshow(augmented)
ax[2].set_title("Augmented")

ax[3].imshow(normalized)
ax[3].set_title("Normalized")

for a in ax:
    a.axis("off")

plt.tight_layout()

save_path = Path("outputs/plots/preprocessing_pipeline.png")

plt.savefig(save_path, dpi=300)

plt.show()

print("\nSaved at:", save_path)