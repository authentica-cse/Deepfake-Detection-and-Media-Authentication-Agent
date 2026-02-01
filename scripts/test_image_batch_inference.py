import os
from utils.face_utils import detect_crop_resize
from models.image_model import predict_face

BASE_DIR = "dataset/image"
CASCADE_PATH = "utils/haarcascade_frontalface_default.xml"


def run_batch(folder, true_label):
    folder_path = os.path.join(BASE_DIR, folder)
    images = os.listdir(folder_path)

    print(f"\nRunning inference on {len(images)} {true_label} images")

    for idx, img_name in enumerate(images, start=1):
        img_path = os.path.join(folder_path, img_name)

        face = detect_crop_resize(img_path, CASCADE_PATH)

        if face is None:
            print(f"[{idx}] {img_name} → No face detected")
            continue

        pred, conf = predict_face(face)

        print(
            f"[{idx}] {img_name} | True: {true_label} | "
            f"PredClassID: {pred} | Conf: {conf:.2f}"
        )


run_batch("real", "REAL")
run_batch("fake", "FAKE")

