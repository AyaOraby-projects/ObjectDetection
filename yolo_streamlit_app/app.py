import streamlit as st
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(page_title="YOLOv8 Detection", layout="centered")
st.title("🟡 YOLOv8 Object Detection")

# Simple model loader
@st.cache_resource
def load_simple_model():
    try:
        # Disable OpenCV GUI
        import os
        os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
        
        # Import and load model
        from ultralytics import YOLO
        return YOLO('yolov8n.pt')
    except:
        return None

# Load model
model = load_simple_model()

if model is None:
    st.error("Could not load model. Try installing: pip install ultralytics pillow")
    st.stop()

# Upload image
uploaded = st.file_uploader("Upload image", type=['jpg', 'png', 'jpeg'])

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption="Original", use_container_width=True)
    
    with st.spinner("Detecting..."):
        # Convert to numpy
        img_np = np.array(img)
        
        # Run prediction
        results = model(img_np, device='cpu')
        
        if results:
            # Get first result
            result = results[0]
            
            # Show number of detections
            if result.boxes:
                count = len(result.boxes)
                st.success(f"Found {count} objects!")
                
                # List detections
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = result.names[cls]
                    st.write(f"- {name}: {conf:.2%} confidence")
                
                # Show image with boxes
                annotated = result.plot()[:, :, ::-1]  # BGR to RGB
                st.image(annotated, caption="Detections", use_container_width=True)
            else:
                st.warning("No objects detected")
