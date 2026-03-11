import os
import cv2
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.face_utils import detect_face


# ------------------ Transforms ------------------

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4,0.4,0.4,0.2),
    transforms.RandomGrayscale(p=0.15),
    transforms.GaussianBlur(3),
    transforms.RandomAffine(5, translate=(0.05,0.05)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

val_test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# ------------------ Patch Function ------------------

def extract_random_patch(img, patch_size=96):
    h, w, _ = img.shape

    if h < patch_size or w < patch_size:
        return img

    x = random.randint(0, w - patch_size)
    y = random.randint(0, h - patch_size)

    return img[y:y+patch_size, x:x+patch_size]


# ------------------ Face Dataset (legacy utility) ------------------

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        CLASS_MAP = {"real":0, "fake":1}

        for cls, label in CLASS_MAP.items():
            class_path = os.path.join(root_dir, cls)
            if not os.path.exists(class_path):
                continue

            for file in os.listdir(class_path):
                if file.lower().endswith((".jpg",".jpeg",".png")):
                    self.samples.append((os.path.join(class_path, file), label))

        random.shuffle(self.samples)
        print(f"Dataset loaded: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = cv2.imread(img_path)

        if image is None:
            image = Image.new("RGB", (224,224))
        else:
            face = detect_face(image)
            if face is not None:
                image = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label


# ------------------ Image Dataset (Phase 2 PATCH LEARNING) ------------------

class ImageDataset(Dataset):
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
        random.shuffle(self.samples)

        self.transform = train_transform if split == "train" else val_test_transform

        print(f"Dataset loaded: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = cv2.imread(img_path)

        if image is None:
            image = Image.new("RGB", (224,224))
        else:
            face = detect_face(image)

            # 50% → face crop
            if face is not None and random.random() < 0.5:
                image = face

            # 50% → random patch
            if random.random() < 0.5:
                image = extract_random_patch(image)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        image = self.transform(image)
        return image, label

