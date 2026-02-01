# scripts/train_image.py
import torch
from torch.utils.data import DataLoader
from scripts.image_dataset import FaceDataset
from models.image_model import ImageDeepfakeModel
import torch.nn as nn
import torch.optim as optim

device = "cpu"  # change to "cuda" if you have GPU

# Load datasets
train_ds = FaceDataset("dataset/train")
val_ds = FaceDataset("dataset/val")

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

# Model, loss, optimizer
model = ImageDeepfakeModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

best_val_acc = 0.0  # Track best validation accuracy

for epoch in range(10):  # Increase epochs if needed
    # --- Training ---
    model.train()
    total, correct = 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        preds = out.argmax(1)
        total += y.size(0)
        correct += (preds == y).sum().item()

    train_acc = correct / total * 100

    # --- Validation ---
    model.eval()
    total_val, correct_val = 0, 0

    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            out_val = model(x_val)
            preds_val = out_val.argmax(1)
            total_val += y_val.size(0)
            correct_val += (preds_val == y_val).sum().item()

    val_acc = correct_val / total_val * 100

    print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/best_image_model.pth")
        print(f"--> New best model saved with Val Acc: {best_val_acc:.2f}%")

print("Training complete.")

