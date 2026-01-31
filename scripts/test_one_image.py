import cv2
import matplotlib.pyplot as plt
from utils.face_utils import detect_crop_resize

IMAGE_PATH = "dataset/image/real/real_0.jpg"  
CASCADE_PATH = "utils/haarcascade_frontalface_default.xml"

face = detect_crop_resize(IMAGE_PATH, CASCADE_PATH)
if face is not None:
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    plt.imshow(face_rgb)
    plt.title("Real Image Face")
    plt.axis("off")
    plt.show()
else:
    print("Face not detected.")

