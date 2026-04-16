import os
from utils.detector import detect_file_type
from models.dummy_model import predict_image, predict_video, predict_audio

def main():
    file_path = input("Enter file path: ")

    if not os.path.exists(file_path):
        print("File not found!")
        return

    file_type = detect_file_type(file_path)

    print(f"Detected file type: {file_type}")

    if file_type == "image":
        result, confidence = predict_image(file_path)

    elif file_type == "video":
        result, confidence = predict_video(file_path)

    elif file_type == "audio":
        result, confidence = predict_audio(file_path)

    else:
        print("Unsupported file type")
        return

    print("\n===== RESULT =====")
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence}")

if __name__ == "__main__":
    main()