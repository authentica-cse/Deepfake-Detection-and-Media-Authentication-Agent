# scripts/test_images.py
import os
import cv2
import torch
from PIL import Image
from torchvision import transforms

from models.image_model import ImageModel
from utils.face_utils import detect_face

# ------------------ Setup ------------------

device = torch.device("cpu")

model = ImageModel()
model.load_state_dict(torch.load("models/image_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dir = "dataset_subset/test"
classes = ["real", "fake"]

total, correct = 0, 0

# ------------------ Testing Loop ------------------

for label_idx, cls in enumerate(classes):
    folder = os.path.join(test_dir, cls)

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        # 1️⃣ Read image
        image = cv2.imread(img_path)
        if image is None:
            continue

        # 2️⃣ Detect face (PASS IMAGE, NOT PATH)
        face = detect_face(image)
        if face is None:
            continue

        # 3️⃣ OpenCV → PIL
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = Image.fromarray(face)

        # 4️⃣ Transform + batch
        face = transform(face).unsqueeze(0).to(device)

        # 5️⃣ Predict
        with torch.no_grad():
            out = model(face)
            probs = torch.softmax(out, dim=1)
            conf, pred = torch.max(probs, 1)

        pred_label = "FAKE" if pred.item() == 1 else "REAL"
        print(f"{img_name} -> Prediction: {pred_label}, Confidence: {conf.item()*100:.2f}%")

        total += 1
        if pred.item() == label_idx:
            correct += 1

# ------------------ Accuracy ------------------

accuracy = correct / total * 100 if total > 0 else 0
print(f"\nTest Accuracy: {accuracy:.2f}% ({correct}/{total})")

