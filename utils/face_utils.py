import cv2

def detect_crop_resize(image_path, cascade_path):
    face_cascade = cv2.CascadeClassifier(cascade_path)

    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]

    face_resized = cv2.resize(face, (224, 224))
    return face_resized

