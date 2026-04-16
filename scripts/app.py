import streamlit as st
import os
from utils.detector import detect_file_type
from models.dummy_model import predict_image, predict_video, predict_audio

st.title("Deepfake Detection App")

uploaded_file = st.file_uploader("Upload an image/video/audio file")

if uploaded_file is not None:
    file_path = os.path.join("temp_" + uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_type = detect_file_type(file_path)

    st.write(f"Detected file type: {file_type}")

    if file_type == "image":
        result = predict_image(file_path)
    elif file_type == "video":
        result = predict_video(file_path)
    elif file_type == "audio":
        result = predict_audio(file_path)
    else:
        result = "Unsupported file type"

    st.success(f"Prediction: {result}")