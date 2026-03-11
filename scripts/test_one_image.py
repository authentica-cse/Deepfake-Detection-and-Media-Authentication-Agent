import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from models.image_model import ImageModel
from models.freq_model import FreqModel

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---------------- TRANSFORMS ----------------

img_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

freq_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# ---------------- FFT ----------------

def extract_fft(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)


# ---------------- LOAD MODELS ----------------

img_model = ImageModel().to(DEVICE)
freq_model = FreqModel().to(DEVICE)

img_model.load_state_dict(torch.load("models/image_model.pth", map_location=DEVICE))
freq_model.load_state_dict(torch.load("models/freq_model.pth", map_location=DEVICE))

img_model.eval()
freq_model.eval()


# ---------------- PREDICT FUNCTION ----------------

def predict_single_image(image_path, return_probs=False):

    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_pil = Image.fromarray(img_rgb)
    x_img = img_transform(img_pil).unsqueeze(0).to(DEVICE)

    fft_img = extract_fft(img_bgr)
    fft_pil = Image.fromarray(fft_img).convert("L")
    x_freq = freq_transform(fft_pil).unsqueeze(0).to(DEVICE)

    if x_freq.shape[1] == 1:
        x_freq = x_freq.repeat(1,3,1,1)

    with torch.no_grad():
        probs_img = torch.softmax(img_model(x_img), dim=1)[0]
        probs_freq = torch.softmax(freq_model(x_freq), dim=1)[0]

    real_p = max(probs_img[0].item(), probs_freq[0].item())
    fake_p = max(probs_img[1].item(), probs_freq[1].item())

    if fake_p > 0.75:
        label = "FAKE"
    elif real_p > 0.75:
        label = "REAL"
    else:
        label = "UNCERTAIN"

    if return_probs:
        return {
            "REAL": real_p,
            "FAKE": fake_p,
            "LABEL": label
        }

    print(f"\nREAL: {real_p*100:.2f}% | FAKE: {fake_p*100:.2f}%")
    print("Label:", label)
if __name__ == "__main__":
    image_path = "dataset_subset/image/fake_549.jpg"
    predict_single_image(image_path)
