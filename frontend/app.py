import streamlit as st
from backend.media_router import detect_media_type, route_to_model

st.set_page_config(
    page_title="Authentica - Deepfake Detector",
    layout="centered"
)

st.title("Authentica")
st.subheader("Deepfake Detection System")
st.write("Upload an image, audio, or video file to test the UI.")

uploaded_file = st.file_uploader(
    "Choose a file",
    type=["jpg", "png", "mp4", "wav", "mp3"]
)

if uploaded_file is not None:
    st.write(f"File name: {uploaded_file.name}")
    st.write(f"MIME type: {uploaded_file.type}")
    file_bytes = uploaded_file.getvalue()

    media_type = detect_media_type(uploaded_file.type)
    st.info(f"Detected media type: {media_type.upper()}")

    # Preview
    if media_type == "image":
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    elif media_type == "video":
        st.video(uploaded_file)
    elif media_type == "audio":
        st.audio(uploaded_file)
    else:
        st.error("Unsupported file type")

    if st.button("Detect Deepfake"):
        label, confidence = route_to_model(media_type, file_bytes)

        st.success(f"Prediction: {label}")
        st.write(f"Confidence: {confidence * 100:.2f}%")



