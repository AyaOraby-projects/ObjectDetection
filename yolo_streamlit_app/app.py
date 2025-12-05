import streamlit as st
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
# Load model (cached) - with error handling
# -------------------------
@st.cache_resource
def load_model():
    try:
        # Import here to catch errors
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
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
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Read and show uploaded image
    from PIL import Image
    
    input_image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Input Image")
    st.image(input_image, use_container_width=True)

    # Create temp directory
    temp_input_dir = "temp_inputs"
    os.makedirs(temp_input_dir, exist_ok=True)
    
    # Save input temporarily
    timestamp = int(time.time())
    input_path = os.path.join(temp_input_dir, f"input_{timestamp}.jpg")
    input_image.save(input_path, format="JPEG", quality=95)

    # Clear previous runs
    out_root = "runs/detect"
    if os.path.exists(out_root):
        try:
            shutil.rmtree(out_root)
        except:
            pass

    # Run detection
    with st.spinner("Running detection..."):
        try:
            # Run prediction
            results = model.predict(input_path, save=True, save_txt=False, save_conf=True)
        except Exception as e:
            st.error(f"Detection error: {str(e)}")
            st.stop()

    # Find the output image
    output_image_path = None
    
    # Method 1: Check runs/detect directory
    if os.path.exists("runs/detect"):
        # Find all prediction directories
        for root, dirs, files in os.walk("runs/detect"):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Check if this looks like our input file
                    if f"input_{timestamp}" in file or "input_" in file:
                        output_image_path = os.path.join(root, file)
                        break
            if output_image_path:
                break
    
    # Method 2: If not found, look for any image file
    if output_image_path is None and os.path.exists("runs/detect"):
        for root, dirs, files in os.walk("runs/detect"):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    output_image_path = os.path.join(root, file)
                    break
            if output_image_path:
                break
    
    # Method 3: Use results directly (fallback)
    if output_image_path is None:
        try:
            # Process results directly
            import cv2
            import numpy as np
            
            # Get the annotated image from results
            result = results[0]
            annotated_img = result.plot()  # Get numpy array
            
            # Convert to PIL Image
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            output_image = Image.fromarray(annotated_img_rgb)
            
            # Save it
            output_image_path = os.path.join(temp_input_dir, f"output_{timestamp}.jpg")
            output_image.save(output_image_path, format="JPEG", quality=95)
            
            st.success("Processed using direct results!")
        except Exception as e:
            st.error(f"Could not process results: {str(e)}")

    # Show results
    if output_image_path and os.path.exists(output_image_path):
        st.subheader("Detected Output")
        st.image(output_image_path, use_container_width=True)

        # Download button
        with open(output_image_path, "rb") as f:
            st.download_button(
                label="📥 Download Result",
                data=f,
                file_name=f"detected_{timestamp}.jpg",
                mime="image/jpeg",
            )
        
        # Show detection info
        try:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                num_detections = len(result.boxes)
                st.info(f"✅ Detected **{num_detections}** objects")
                
                # Show class distribution
                if num_detections > 0:
                    with st.expander("📊 View detailed results"):
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
                        
                        # Display table
                        for class_name, data in detected_classes.items():
                            avg_conf = sum(data['confidences']) / len(data['confidences'])
                            st.write(f"**{class_name}**: {data['count']} objects (avg confidence: {avg_conf:.2%})")
        except:
            pass  # Skip stats if there's an error
    else:
        st.error("Could not generate output image.")
