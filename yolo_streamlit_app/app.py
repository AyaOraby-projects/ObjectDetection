import streamlit as st
import os
import time
import sys
from PIL import Image
import numpy as np

# Try to import cv2 with fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    st.warning(f"OpenCV import warning: {e}")
    CV2_AVAILABLE = False
    # Try to install missing package
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless==4.9.0.80"])
        import cv2
        CV2_AVAILABLE = True
        st.success("Successfully installed OpenCV!")
    except:
        pass

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
# Load model (cached) - with better error handling
# -------------------------
@st.cache_resource
def load_model():
    try:
        # Import here to catch errors
        from ultralytics import YOLO
        import torch
        
        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st.sidebar.info(f"Using device: {device.upper()}")
        
        # Load model
        model = YOLO("yolov8n.pt")
        model.to(device)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Make sure requirements.txt includes: ultralytics, opencv-python-headless")
        return None

model = None
with st.spinner("Loading YOLOv8 model..."):
    model = load_model()

if model is None:
    st.error("Model failed to load. Check the logs for details.")
    st.stop()

# -------------------------
# File uploader
# -------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    # Read and show uploaded image
    input_image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Image")
        st.image(input_image, use_container_width=True)
    
    # Convert PIL to numpy array for processing
    img_array = np.array(input_image)
    
    # Create temp directory
    temp_input_dir = "temp_inputs"
    os.makedirs(temp_input_dir, exist_ok=True)
    
    # Save input temporarily
    timestamp = int(time.time())
    input_path = os.path.join(temp_input_dir, f"input_{timestamp}.jpg")
    input_image.save(input_path, format="JPEG", quality=95)

    # Clear previous runs if they exist
    out_root = "runs/detect"
    if os.path.exists(out_root):
        try:
            import shutil
            shutil.rmtree(out_root)
        except:
            pass

    # Run detection
    with st.spinner("Running YOLOv8 detection..."):
        try:
            # Run prediction with optimized settings for Streamlit Cloud
            results = model.predict(
                source=img_array,  # Use numpy array directly
                save=False,  # Don't save to disk by default
                save_txt=False,
                save_conf=True,
                conf=0.25,  # Confidence threshold
                iou=0.45,   # NMS IoU threshold
                max_det=300, # Maximum detections per image
                device='cpu'  # Force CPU to avoid CUDA issues in cloud
            )
            
            # Process results
            if results and len(results) > 0:
                result = results[0]
                
                # Get annotated image
                annotated_array = result.plot()  # Returns BGR numpy array
                
                # Convert BGR to RGB for PIL
                if CV2_AVAILABLE:
                    annotated_array_rgb = cv2.cvtColor(annotated_array, cv2.COLOR_BGR2RGB)
                else:
                    # Fallback if cv2 not available
                    annotated_array_rgb = annotated_array[:, :, ::-1]  # Simple BGR to RGB
                
                # Convert to PIL Image
                output_image = Image.fromarray(annotated_array_rgb)
                
                # Save output
                output_path = os.path.join(temp_input_dir, f"output_{timestamp}.jpg")
                output_image.save(output_path, format="JPEG", quality=95)
                
                # Display output
                with col2:
                    st.subheader("Detected Output")
                    st.image(output_image, use_container_width=True)
                
                # Show detection statistics
                st.success(f"✅ Detection complete!")
                
                # Count detections
                if hasattr(result, 'boxes') and result.boxes is not None:
                    num_detections = len(result.boxes)
                    st.info(f"**Detected objects:** {num_detections}")
                    
                    # Show class distribution
                    if num_detections > 0:
                        with st.expander("📊 View Detailed Results", expanded=True):
                            class_names = result.names
                            detected_classes = {}
                            
                            for box in result.boxes:
                                class_id = int(box.cls[0])
                                class_name = class_names[class_id]
                                confidence = float(box.conf[0])
                                
                                if class_name not in detected_classes:
                                    detected_classes[class_name] = {
                                        'count': 0,
                                        'confidences': []
                                    }
                                
                                detected_classes[class_name]['count'] += 1
                                detected_classes[class_name]['confidences'].append(confidence)
                            
                            # Display as table
                            if detected_classes:
                                st.markdown("### Detection Summary")
                                for class_name, data in detected_classes.items():
                                    avg_conf = sum(data['confidences']) / len(data['confidences'])
                                    st.write(f"**{class_name}**: {data['count']} objects (avg confidence: {avg_conf:.1%})")
                
                # Download button
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Result",
                        data=f,
                        file_name=f"yolov8_detection_{timestamp}.jpg",
                        mime="image/jpeg",
                        key=f"download_{timestamp}"
                    )
            else:
                st.warning("No objects detected. Try adjusting the confidence threshold.")
                
        except Exception as e:
            st.error(f"Detection error: {str(e)}")
            st.info("If this is a CUDA error, try running on CPU by modifying the device parameter.")
    
    # Clean up temporary files
    try:
        if os.path.exists(input_path):
            os.remove(input_path)
    except:
        pass

# Sidebar information
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    This app uses YOLOv8n (nano version) for object detection.
    
    **Model details:**
    - 80 COCO classes
    - Real-time inference
    - Optimized for cloud deployment
    
    **Tips:**
    - Upload clear images for best results
    - Supported formats: JPG, PNG, BMP, WebP
    - Detection works best with multiple objects
    """)
    
    # Confidence threshold slider
    st.markdown("### Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
        help="Higher values = fewer but more confident detections"
    )
