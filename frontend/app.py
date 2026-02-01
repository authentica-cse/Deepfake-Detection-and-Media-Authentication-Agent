import streamlit as st

st.set_page_config(page_title="Authentica - Deepfake Detector", layout="centered")
st.title("Authentica")
st.subheader("Basic UI Demo")
st.write("Upload an image, audio, or video file to test the UI.")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "mp4", "wav", "mp3"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    st.write(f"Uploaded file: {uploaded_file.name} (type: {file_type})")
    
    # Preview uploaded file
    if "image" in file_type:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    elif "video" in file_type:
        st.video(uploaded_file)
    elif "audio" in file_type:
        st.audio(uploaded_file)
    
    # Detect button (placeholder for now)
    if st.button("Detect Deepfake"):
        st.write("Processing...")
        st.success("Prediction: Real / Deepfake (placeholder)")

