import torch
from torch.utils.data import DataLoader
from scripts.image_dataset import FaceDataset
from models.image_model import ImageDeepfakeModel
import torch.nn as nn
import torch.optim as optim

device = "cpu"

train_ds = FaceDataset("dataset/train")
val_ds = FaceDataset("dataset/val")

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

model = ImageDeepfakeModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):
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

    acc = correct / total * 100
    print(f"Epoch {epoch+1} Train Acc: {acc:.2f}%")

torch.save(model.state_dict(), "models/image_model.pth")
print("Model saved.")

