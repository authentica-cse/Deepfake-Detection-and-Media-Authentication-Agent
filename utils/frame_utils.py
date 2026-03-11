import cv2

def extract_frames(video_path, max_frames=30, frame_skip=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    saved = 0

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_skip == 0:
            frames.append(frame)
            saved += 1
        count += 1

    cap.release()
    return frames

