import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from scripts.image_dataset import FaceDataset, train_transform, val_test_transform
from models.image_model import ImageModel


DATASET_DIR = "dataset_large"
BATCH_SIZE = 32
EPOCHS = 12
PATIENCE = 4
LR = 1e-4


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    train_ds = FaceDataset(f"{DATASET_DIR}/train", train_transform)
    val_ds   = FaceDataset(f"{DATASET_DIR}/val", val_test_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ImageModel().to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        scheduler.step()

        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "models/image_model.pth")
            print("Model saved")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print("Early stopping")
            break


if __name__ == "__main__":
    main()

