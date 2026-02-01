def detect_media_type(mime_type: str):
    """
    Detect whether uploaded file is image, video, or audio
    """
    if "image" in mime_type:
        return "image"
    elif "video" in mime_type:
        return "video"
    elif "audio" in mime_type:
        return "audio"
    else:
        return "unknown"


def routeToModel(media_type, file_bytes=None):
    """
    Route uploaded media to appropriate model
    """
    if media_type == "image":
        try:
            import cv2
            import numpy as np
            from models.image_model import predict_face

            # Convert uploaded bytes to OpenCV image
            image_array = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            label, confidence = predict_face(img)
            return label, confidence

        except Exception as e:
            return f"Image model error: {str(e)}", 0.0

    elif media_type in ["audio", "video"]:
        # Still dummy for Day 4
        return "REAL / DEEPFAKE (audio/video stub)", 0.80

    else:
        return "Unsupported file type", 0.0
