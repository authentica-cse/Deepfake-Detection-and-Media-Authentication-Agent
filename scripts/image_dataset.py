import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils.face_utils import extract_face

class FaceDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        for label, cls in enumerate(["real", "fake"]):
            folder = os.path.join(root_dir, cls)
            for img in os.listdir(folder):
                self.samples.append((os.path.join(folder, img), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        face = extract_face(path)

        if face is None:
            image = Image.open(path).resize((224, 224))
        else:
            image = Image.fromarray(face)

        image = self.transform(image)
        return image, label

