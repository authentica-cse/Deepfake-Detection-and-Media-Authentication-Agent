import torch
from models.image_model import ImageDeepfakeModel
from utils.face_utils import extract_face
from torchvision import transforms
from PIL import Image
import sys

model = ImageDeepfakeModel()
model.load_state_dict(torch.load("models/image_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img_path = sys.argv[1]
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

label = "FAKE" if pred.item() == 1 else "REAL"
print(f"Prediction: {label}")
print(f"Confidence: {conf.item()*100:.2f}%")

