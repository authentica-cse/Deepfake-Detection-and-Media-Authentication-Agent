import torch
import torch.nn.functional as F
import timm
import cv2
import numpy as np

# Load pretrained EfficientNet (ImageNet)
model = timm.create_model("efficientnet_b0", pretrained=True)
model.eval()

def preprocess_face(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = cv2.resize(face_img, (224, 224))
    face_img = face_img / 255.0

    face_img = np.transpose(face_img, (2, 0, 1))
    face_img = torch.tensor(face_img, dtype=torch.float32)
    face_img = face_img.unsqueeze(0)

    return face_img


def predict_face(face_img):
    input_tensor = preprocess_face(face_img)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)

    confidence, pred = torch.max(probs, dim=1)

    label = "REAL" if pred.item() == 0 else "FAKE"
    return label, confidence.item()

