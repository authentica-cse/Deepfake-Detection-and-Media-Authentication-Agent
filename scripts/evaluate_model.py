import os
import sys
import torch
import cv2
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.image_model import ImageModel

DEVICE = "cpu"
MODEL_PATH = "models/image_model.pth"
DATASET_ROOT = "dataset_large/test"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

model = ImageModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

y_true, y_pred = [], []

CLASS_MAP = {"real":0, "fake":1}

for cls in ["real","fake"]:
    folder = os.path.join(DATASET_ROOT, cls)

    for file in os.listdir(folder):
        if not file.endswith((".jpg",".png",".jpeg")):
            continue

        path = os.path.join(folder, file)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img)
        x = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(x)
            pred = torch.argmax(output, dim=1).item()

        y_true.append(CLASS_MAP[cls])
        y_pred.append(pred)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true,y_pred))

print("\nClassification Report:")
print(classification_report(y_true,y_pred,target_names=["REAL","FAKE"]))

