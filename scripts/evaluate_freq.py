import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from scripts.freq_dataset import FreqDataset
from models.freq_model import FreqModel

DATASET_DIR = "dataset_large"
BATCH_SIZE = 32


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    test_ds = FreqDataset(DATASET_DIR, split="val")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = FreqModel().to(device)
    model.load_state_dict(torch.load("models/freq_model.pth", map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            probs = F.softmax(out, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["REAL", "FAKE"]))


if __name__ == "__main__":
    main()

