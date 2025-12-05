import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import time
import shutil

# -------------------------
# App UI
# -------------------------
st.set_page_config(page_title="YOLOv8 Object Detection", layout="centered")
st.title("🟡 YOLOv8 Object Detection")

st.markdown(
    """
Upload an image and the app will run YOLOv8 (yolov8n) detector and show the output image with bounding boxes.
"""
)

# -------------------------
# Load model (cached)
# -------------------------
@st.cache_resource
def load_model(model_name="yolov8n.pt"):
    return YOLO(model_name)

with st.spinner("Loading YOLOv8 model..."):
    model = load_model()

# -------------------------
# File uploader
# -------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # show uploaded image
    input_image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Input Image")
    st.image(input_image, width="stretch")  # UPDATED

    # save input temporarily
    temp_input_dir = "temp_inputs"
    os.makedirs(temp_input_dir, exist_ok=True)
    input_path = os.path.join(temp_input_dir, f"input_{int(time.time())}.png")
    input_image.save(input_path)

    # clear previous runs/detect/predict
    out_root = "runs/detect"
    if os.path.exists(out_root):
        try:
            shutil.rmtree(out_root)
        except Exception:
            pass

    # run detection
    with st.spinner("Running detection (this may take a few seconds)..."):
        model(input_path, save=True)

    # locate output
    out_dir = os.path.join("runs", "detect", "predict")
    output_image_path = None
    if os.path.exists(out_dir):
        files_list = [
            os.path.join(out_dir, f)
            for f in os.listdir(out_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if files_list:
            output_image_path = max(files_list, key=os.path.getmtime)

    # show results
    if output_image_path:
        st.subheader("Detected Output")
        st.image(output_image_path, width="stretch")  # UPDATED

        # download button
        with open(output_image_path, "rb") as f:
            st.download_button(
                label="Download result",
                data=f,
                file_name=os.path.basename(output_image_path),
                mime="image/png",
            )
    else:
        st.error("Could not find the detection output image.")
