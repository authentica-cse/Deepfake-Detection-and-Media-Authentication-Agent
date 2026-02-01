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


def route_to_model(media_type: str):
    """
    Dummy backend routing logic
    (ML models will be connected later)
    """
    if media_type == "unknown":
        return "Unsupported file", 0.0

    # Placeholder prediction
    return "REAL / DEEPFAKE", 0.85
