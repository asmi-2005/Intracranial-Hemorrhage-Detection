from src.preprocessing.image_dataset import *
from src.utils.config import *

# Load training dataset
dataset = HemorrhageDataset(
    csv_file=PROCESSED_DATA_DIR / "train.csv",
    transform=train_transform
)

print("Dataset Size :", len(dataset))

image, label = dataset[0]

print("Image Shape :", image.shape)
print("Label :", label)
print("Tensor Type :", type(image))