import cv2
import os

# Load Haar cascade
CASCADE_PATH = os.path.join(
    os.path.dirname(__file__),
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def detect_face(image, margin=0.25, min_size=40):
    """
    Detects the largest face in an image/frame and returns cropped face.
    Returns None if no face is detected.
    """

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_size, min_size)
    )

    if len(faces) == 0:
        return None

    # Pick the largest face (best for videos)
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    # Add margin around face
    h_img, w_img, _ = image.shape
    margin_x = int(w * margin)
    margin_y = int(h * margin)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w_img, x + w + margin_x)
    y2 = min(h_img, y + h + margin_y)

    face = image[y1:y2, x1:x2]

    if face.size == 0:
        return None

    return face
