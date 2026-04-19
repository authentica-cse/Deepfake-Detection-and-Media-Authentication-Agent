import streamlit as st
import os

from models.fortress_scanner import predict_audio, predict_image, predict_video
from utils.detector import detect_file_type

st.set_page_config(page_title="Deepfake Detector", layout="centered")

st.title("AI Media Authentication System")

# Upload file
user_file = st.file_uploader("Upload an image / video / audio file")

if user_file is not None:
    # Save file temporarily
    media_path = os.path.join("temp_" + user_file.name)

    with open(media_path, "wb") as f:
        f.write(user_file.getbuffer())

    # Detect type
    file_type = detect_file_type(media_path)

    st.write(f"Detected file type: {file_type}")

    # Preview
    if file_type == "image":
        st.image(user_file, caption="Uploaded Image", use_column_width=True)

    elif file_type == "video":
        st.video(user_file)

    elif file_type == "audio":
        st.audio(user_file)

    else:
        st.error("Unsupported file type")
        st.stop()

    # Prediction
    with st.spinner("Analyzing..."):

        if file_type == "image":
            result = predict_image(media_path)

        elif file_type == "video":
            result = predict_video(media_path)

        elif file_type == "audio":
            result = predict_audio(media_path)

    # Display result nicely
    if "fake" in result.lower():
        st.error(f"Prediction: {result}")
    else:
        st.success(f"Prediction: {result}")
    
