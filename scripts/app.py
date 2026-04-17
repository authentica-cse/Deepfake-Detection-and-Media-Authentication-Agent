import streamlit as st
st.set_page_config(page_title="Deepfake Detector", layout="centered")
import os
from utils.detector import detect_file_type
from models.fortress_scanner import predict_audio
st.title("Ai Media Authentication Checklist")

user_file = st.file_uploader("Upload an image/video/audio file")

if user_file is not None:
    media_path = os.path.join("temp_" + user_file.name)

    with open(media_path, "wb") as f:
        f.write(user_file.getbuffer())

    file_type = detect_file_type(media_path)

    st.write(f"Detected file type: {file_type}")

    # Preview
    if file_type == "image":
        st.image(user_file, caption="Uploaded Image", use_column_width=True)
    elif file_type == "video":
        st.video(user_file)
    elif file_type == "audio":
        st.audio(user_file)

    # Error handling
    if file_type == "unknown":
        st.error("Unsupported file format")
        st.stop()

    # Prediction
    with st.spinner("Analyzing..."):
        if file_type == "image":
            result = predict_image(file_path)
        elif file_type == "video":
            result = predict_video(file_path)
        elif file_type == "audio":
            result = predict_audio(file_path)
        else:
            result = "Unsupported file type"

    st.markdown(f"### 🎯 Prediction: `{result}`")

   
