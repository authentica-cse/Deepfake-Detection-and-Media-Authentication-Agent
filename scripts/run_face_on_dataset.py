import os
import cv2
import matplotlib.pyplot as plt
from utils.face_utils import detect_crop_resize

BASE_DIR = "dataset/image"
CASCADE_PATH = "utils/haarcascade_frontalface_default.xml"

def show_faces(folder, max_images=5):
    folder_path = os.path.join(BASE_DIR, folder)

    shown = 0
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        face = detect_crop_resize(img_path, CASCADE_PATH)
        if face is None:
            continue

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        plt.imshow(face_rgb)
        plt.title(folder)
        plt.axis("off")
        plt.show()

        shown += 1
        if shown >= max_images:
            break

if __name__ == "__main__":
    show_faces("real")
    show_faces("fake")

