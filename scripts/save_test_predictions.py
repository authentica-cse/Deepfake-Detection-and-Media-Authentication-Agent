import csv
import torch
from models.image_model import ImageDeepfakeModel
from utils.face_utils import extract_face
from torchvision import transforms
from PIL import Image
import os

model = ImageDeepfakeModel()
model.load_state_dict(torch.load("models/image_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_dir = "dataset/test"
results = []

for cls in ["real", "fake"]:
    folder = os.path.join(test_dir, cls)
    for img_file in os.listdir(folder):
        img_path = os.path.join(folder, img_file)
        face = extract_face(img_path)
        if face is None:
            image = Image.open(img_path).resize((224,224))
        else:
            image = Image.fromarray(face)
        x = transform(image).unsqueeze(0)
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1)
            conf, pred = torch.max(probs, 1)
        label = "FAKE" if pred.item() == 1 else "REAL"
        results.append([img_file, cls, label, conf.item()*100])

# Save CSV
with open("test_predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "True Label", "Predicted Label", "Confidence"])
    writer.writerows(results)

print("Predictions saved to test_predictions.csv")

