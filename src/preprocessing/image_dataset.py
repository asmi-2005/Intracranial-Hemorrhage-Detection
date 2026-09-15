import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.config import *

LABEL_MAP = {
    "No_Hemorrhage": 0,
    "Epidural": 1,
    "Intraparenchymal": 2,
    "Subdural": 3,
    "Intraventricular": 4,
    "Subarachnoid": 5
}
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class HemorrhageDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.df = pd.read_csv(csv_file)

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image_path = self.df.iloc[idx]["Image_Path"]

        label = self.df.iloc[idx]["Label"]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            label = LABEL_MAP[label]
            return image, label