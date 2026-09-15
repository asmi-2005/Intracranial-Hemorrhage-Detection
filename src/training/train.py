
# Placeholder complete train.py template.
# NOTE: This file contains the corrected structure discussed in chat.
# Replace or extend helper functions as needed.

import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.preprocessing.image_dataset import (
    HemorrhageDataset,
    train_transform,
    test_transform
)
from src.models.model import HemorrhageClassifier
from src.utils.config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = HemorrhageDataset(
    csv_file=PROCESSED_DATA_DIR / "train.csv",
    transform=train_transform
)
val_dataset = HemorrhageDataset(
    csv_file=PROCESSED_DATA_DIR / "validation.csv",
    transform=test_transform
)

train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,
                          num_workers=0,pin_memory=False)
val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,
                        num_workers=0,pin_memory=False)

model = HemorrhageClassifier().to(device)

class_weights = torch.tensor(
    [0.17,2.20,5.80,7.20,15.0,32.0],
    dtype=torch.float
).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)

EPOCHS = 15
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,T_max=EPOCHS
)

best_accuracy = float("-inf")
best_model_wts = copy.deepcopy(model.state_dict())

train_losses=[]
val_losses=[]
train_accuracies=[]
val_accuracies=[]

patience=5
counter=0

def train_one_epoch(model,dataloader,criterion,optimizer,device):
    model.train()
    running_loss=0.0
    correct=0
    total=0
    for images,labels in dataloader:
        images=images.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        _,pred=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(pred==labels).sum().item()

    return running_loss/len(dataloader),100*correct/total

def validate(model,dataloader,criterion,device):
    model.eval()
    running_loss=0.0
    correct=0
    total=0
    with torch.no_grad():
        for images,labels in dataloader:
            images=images.to(device)
            labels=labels.to(device)

            outputs=model(images)
            loss=criterion(outputs,labels)

            running_loss+=loss.item()
            _,pred=torch.max(outputs,1)
            total+=labels.size(0)
            correct+=(pred==labels).sum().item()

    return running_loss/len(dataloader),100*correct/total

print("Starting training...")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss,train_acc=train_one_epoch(
        model,train_loader,criterion,optimizer,device
    )
    val_loss,val_acc=validate(
        model,val_loader,criterion,device
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.2f}%")

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), MODELS_DIR/"efficientnetv2_best.pth")
        counter = 0
        print("Best model saved.")
    else:
        counter += 1
        print(f"No improvement ({counter}/{patience})")

    scheduler.step()

    if counter >= patience:
        print("Early stopping triggered.")
        break

model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), MODELS_DIR/"efficientnetv2_final.pth")

plt.figure(figsize=(10,5))
plt.plot(train_losses,label="Train")
plt.plot(val_losses,label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR/"loss_curve.png")
plt.close()

plt.figure(figsize=(10,5))
plt.plot(train_accuracies,label="Train")
plt.plot(val_accuracies,label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR/"accuracy_curve.png")
plt.close()

print(f"Training completed. Best Validation Accuracy: {best_accuracy:.2f}%")
