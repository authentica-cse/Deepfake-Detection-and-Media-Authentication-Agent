def detect_file_type(filename):
    filename = filename.lower()

    if filename.endswith((".jpg", ".jpeg", ".png")):
        return "image"
    elif filename.endswith((".mp4", ".avi")):
        return "video"
    elif filename.endswith((".wav", ".mp3")):
        return "audio"
    else:
        return "unknown"