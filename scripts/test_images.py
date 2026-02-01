# scripts/test_images.py
import os
import torch
from models.image_model import ImageDeepfakeModel
from utils.face_utils import extract_face
from torchvision import transforms
from PIL import Image

# Load model
model = ImageDeepfakeModel()
model.load_state_dict(torch.load("models/image_model.pth", map_location="cpu"))
model.eval()

# Transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Test dataset path
test_dir = "dataset/test"
classes = ["real", "fake"]

total, correct = 0, 0

for label_idx, cls in enumerate(classes):
    folder = os.path.join(test_dir, cls)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        face = extract_face(img_path)

        if face is None:
            image = Image.open(img_path).resize((224, 224))
        else:
            image = Image.fromarray(face)

        x = transform(image).unsqueeze(0)

        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1)
            conf, pred = torch.max(probs, 1)

        pred_label = "FAKE" if pred.item() == 1 else "REAL"
        print(f"{img_name} -> Prediction: {pred_label}, Confidence: {conf.item()*100:.2f}%")

        total += 1
        if pred.item() == label_idx:
            correct += 1

accuracy = correct / total * 100
print(f"\nTest Accuracy: {accuracy:.2f}% ({correct}/{total})")

