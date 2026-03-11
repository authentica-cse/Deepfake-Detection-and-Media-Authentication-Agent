import torch
import sys
from PIL import Image
from torchvision import transforms
from models.image_model import ImageModel
from utils.face_utils import extract_face

MODEL_PATH = "models/image_model.pth"
DEVICE = "cpu"

model = ImageModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

img_path = sys.argv[1]
face = extract_face(img_path)

if face is not None:
    image = Image.fromarray(face)
else:
    image = Image.open(img_path).convert("RGB")

x = transform(image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0]

fake_p = probs[1].item()

if fake_p > 0.65:
    label = "FAKE"
elif fake_p < 0.35:
    label = "REAL"
else:
    label = "UNCERTAIN"

print(f"Prediction: {label}")
print(f"Fake Probability: {fake_p*100:.2f}%")

