import streamlit as st
import os
import time
import shutil
from PIL import Image
import numpy as np
import sys

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
# Load model (cached) - with OpenCV workaround
# -------------------------
@st.cache_resource
def load_model():
    try:
        # Try to set OpenCV environment variable before importing
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
        
        # Import here to isolate OpenCV issues
        from ultralytics import YOLO
        
        with st.spinner("Downloading YOLOv8 model (first time only)..."):
            try:
                # Use yolov8n.pt - it will download automatically
                model = YOLO("yolov8n.pt")
                return model
            except Exception as e:
                st.error(f"Model loading error: {str(e)}")
                return None
                
    except ImportError as e:
        st.error(f"Import error: {str(e)}")
        st.info("""
        **Required packages:** Make sure your `requirements.txt` includes:
        ```
        ultralytics
        opencv-python
        Pillow
        numpy
        streamlit
        ```
        """)
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

# Load the model
model = load_model()

if model is None:
    st.error("""
    **Failed to load model. Possible solutions:**
    
    1. **For Streamlit Cloud:** Make sure your `requirements.txt` has:
       ```
       ultralytics
       opencv-python
       Pillow
       numpy
       streamlit
       ```
    
    2. **Wait a moment and refresh** - First-time model download can take time
    
    3. **Check deployment logs** in Streamlit Cloud dashboard
    """)
    st.stop()

# -------------------------
# File uploader
# -------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    try:
        # Show uploaded image
        input_image = Image.open(uploaded_file).convert("RGB")
        st.subheader("📤 Input Image")
        st.image(input_image, use_container_width=True)
        
        # Create temp directory
        temp_input_dir = "temp_inputs"
        os.makedirs(temp_input_dir, exist_ok=True)
        
        # Save input temporarily
        timestamp = int(time.time())
        input_path = os.path.join(temp_input_dir, f"input_{timestamp}.jpg")
        input_image.save(input_path, format="JPEG", quality=95)
        
        # Clear previous runs
        if os.path.exists("runs"):
            try:
                shutil.rmtree("runs")
            except:
                pass
        
        # Run detection
        with st.spinner("🔍 Detecting objects..."):
            try:
                # Run prediction with specific settings
                results = model.predict(
                    source=input_path,
                    save=True,
                    save_txt=False,
                    conf=0.25,
                    iou=0.45,
                    show=False,
                    project="runs/detect",
                    name="prediction",
                    exist_ok=True
                )
                
                st.success("✅ Detection complete!")
                
            except Exception as e:
                st.error(f"Detection failed: {str(e)}")
                # Try alternative method
                with st.spinner("Trying alternative method..."):
                    try:
                        results = model(input_path)
                        st.success("✅ Detection complete (alternative method)!")
                    except Exception as e2:
                        st.error(f"All detection methods failed: {str(e2)}")
                        st.stop()
        
        # Find the output image
        output_image_path = None
        
        # Look for output in runs/detect/prediction
        possible_locations = [
            os.path.join("runs", "detect", "prediction"),
            os.path.join("runs", "detect", "predict"),
            os.path.join("runs", "detect")
        ]
        
        for location in possible_locations:
            if os.path.exists(location):
                for file in os.listdir(location):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        output_image_path = os.path.join(location, file)
                        break
                if output_image_path:
                    break
        
        # If still not found, search recursively
        if output_image_path is None and os.path.exists("runs"):
            for root, dirs, files in os.walk("runs"):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        output_image_path = os.path.join(root, file)
                        break
                if output_image_path:
                    break
        
        # Show results
        if output_image_path and os.path.exists(output_image_path):
            st.subheader("🎯 Detected Output")
            st.image(output_image_path, use_container_width=True)
            
            # Download button
            with open(output_image_path, "rb") as f:
                st.download_button(
                    label="📥 Download Result",
                    data=f,
                    file_name=f"yolo_detection_{timestamp}.jpg",
                    mime="image/jpeg",
                )
            
            # Show detection statistics
            try:
                if results and len(results) > 0:
                    result = results[0]
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        num_objects = len(result.boxes)
                        
                        st.subheader("📊 Detection Statistics")
                        
                        # Create metrics in columns
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Objects Detected", num_objects)
                        
                        if num_objects > 0:
                            # Get confidence values
                            confidences = [float(box.conf[0]) for box in result.boxes]
                            avg_confidence = np.mean(confidences)
                            
                            with col2:
                                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                            
                            with col3:
                                st.metric("High Confidence", f"{max(confidences):.1%}")
                            
                            # Show detected classes
                            st.write("**Detected Classes:**")
                            class_names = result.names
                            class_counts = {}
                            
                            for box in result.boxes:
                                class_id = int(box.cls[0])
                                class_name = class_names[class_id]
                                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                            
                            # Display classes in a nice format
                            classes_text = ", ".join([f"{name} ({count})" for name, count in class_counts.items()])
                            st.info(classes_text)
                        else:
                            st.warning("No objects detected in the image.")
            except Exception as e:
                st.warning(f"Could not display statistics: {str(e)}")
                
        else:
            # Fallback: Create output from results directly
            st.warning("Output file not found. Trying to create from results...")
            try:
                if results and len(results) > 0:
                    result = results[0]
                    # Get the plotted image
                    plotted_array = result.plot()
                    
                    # Convert to PIL Image
                    plotted_image = Image.fromarray(plotted_array[..., ::-1])  # BGR to RGB
                    
                    st.subheader("🎯 Detected Output")
                    st.image(plotted_image, use_container_width=True)
                    
                    # Save and offer download
                    output_path = os.path.join(temp_input_dir, f"result_{timestamp}.jpg")
                    plotted_image.save(output_path, format="JPEG", quality=95)
                    
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Result",
                            data=f,
                            file_name=f"yolo_detection_{timestamp}.jpg",
                            mime="image/jpeg",
                        )
            except Exception as e:
                st.error(f"Could not display results: {str(e)}")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# -------------------------
# Sidebar info
# -------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **YOLOv8 Object Detection**
    
    This app uses YOLOv8 (You Only Look Once) 
    to detect objects in images.
    
    **Model:** yolov8n (nano version)
    **Classes:** 80+ COCO dataset objects
    **Confidence:** 0.25 threshold
    
    Upload any image to see object detection
    in action!
    """)
    
    # Clear cache button
    if st.button("🔄 Clear Cache", help="Clear temporary files and cache"):
        try:
            if os.path.exists("temp_inputs"):
                shutil.rmtree("temp_inputs")
            if os.path.exists("runs"):
                shutil.rmtree("runs")
            st.cache_resource.clear()
            st.success("Cache cleared!")
            st.rerun()
        except:
            st.warning("Could not clear all cache")
