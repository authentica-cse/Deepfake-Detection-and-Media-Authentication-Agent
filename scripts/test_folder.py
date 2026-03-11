import os
import torch
import cv2
from PIL import Image
from torchvision import transforms
from models.image_model import ImageModel
from utils.face_utils import detect_face

DEVICE = "cpu"
MODEL_PATH = "models/image_model.pth"
IMAGE_DIR = "dataset_subset/val/fake"   # change to fake to test fake folder

model = ImageModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

correct = 0
total = 0

for file in os.listdir(IMAGE_DIR):
    if not file.lower().endswith((".jpg",".png",".jpeg")):
        continue

    path = os.path.join(IMAGE_DIR, file)
    img = cv2.imread(path)

    face = detect_face(img)
    if face is None:
        continue

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(face)

    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)
        pred = torch.argmax(probs,1).item()

    total += 1

    # For REAL folder → correct prediction is 0
    if pred == 1:
        correct += 1

print(f"Accuracy on REAL folder: {correct}/{total} = {(correct/total)*100:.2f}%")

