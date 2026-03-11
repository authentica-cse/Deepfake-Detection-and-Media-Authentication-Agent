import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class FreqDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = root_dir
        self.split = split

        self.real_dir = os.path.join(root_dir, split, "real")
        self.fake_dir = os.path.join(root_dir, split, "fake")

        self.real_images = [(os.path.join(self.real_dir, f), 0)
                            for f in os.listdir(self.real_dir)]
        self.fake_images = [(os.path.join(self.fake_dir, f), 1)
                            for f in os.listdir(self.fake_dir)]

        self.samples = self.real_images + self.fake_images

        print(f"Freq Dataset loaded: {len(self.samples)} images")

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def fft_transform(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)

        magnitude = cv2.normalize(
            magnitude, None, 0, 255, cv2.NORM_MINMAX
        )

        return magnitude.astype(np.uint8)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = cv2.imread(img_path)

        if img is None:
            img = np.zeros((224,224), dtype=np.uint8)
        else:
            img = self.fft_transform(img)

        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)

        return img, label

