import streamlit as st
import os
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tempfile

# -------------------------
# App UI Configuration
# -------------------------
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🟡",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🟡 YOLOv8 Object Detection")
st.markdown("""
Upload an image and the app will run YOLOv8 (yolov8n) detector 
and show the output image with bounding boxes.
""")
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Load Model (Cached) - Fixed Version
# -------------------------
@st.cache_resource
def load_model():
    try:
        # Set environment variable to avoid OpenCV GUI issues
        os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'
        
        # Import ultralytics
        from ultralytics import YOLO
        import torch
        
        # Load model
        st.info("📥 Downloading YOLOv8n model (if not already cached)...")
        model = YOLO('yolov8n.pt')
        
        # Test the model with a simple prediction
        test_image = Image.new('RGB', (640, 480), color='white')
        test_array = np.array(test_image)
        _ = model(test_array, verbose=False)
        
        st.success("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("💡 Trying alternative loading method...")
        
        # Alternative loading method
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            return model
        except Exception as e2:
            st.error(f"❌ Alternative loading also failed: {str(e2)}")
            return None

# -------------------------
# Sidebar Configuration
# -------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
        help="Higher values = fewer but more confident detections"
    )
    
    iou_threshold = st.slider(
        "IOU Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
        help="Non-maximum suppression threshold"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    **YOLOv8n (Nano)**
    - 80 COCO classes
    - Real-time detection
    - Optimized for speed
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    1. Upload clear images
    2. Adjust confidence for better results
    3. Try different images for varied objects
    """)

# -------------------------
# Initialize Model
# -------------------------
if 'model' not in st.session_state:
    with st.spinner("🚀 Loading YOLOv8 model..."):
        model = load_model()
        if model is not None:
            st.session_state.model = model
            st.rerun()
        else:
            st.error("Failed to load model. Please check the logs.")
            st.stop()

# Get model from session state
model = st.session_state.model

# -------------------------
# File Uploader
# -------------------------
st.markdown("---")
st.subheader("📤 Upload Image")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Supported formats: JPG, PNG, BMP, WebP"
)

# -------------------------
# Processing Function
# -------------------------
def draw_bounding_boxes(image, boxes, class_names, colors=None):
    """Draw bounding boxes on image without OpenCV"""
    draw = ImageDraw.Draw(image)
    
    # Create a simple color palette
    if colors is None:
        # Generate distinct colors for different classes
        import random
        random.seed(42)
        colors = {}
        for class_id in range(len(class_names)):
            colors[class_id] = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )
    
    # Try to use a font, fallback to default if not available
    try:
        font = ImageFont.truetype("Arial", size=14)
    except:
        font = ImageFont.load_default()
    
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        class_id = int(box.cls[0].cpu().numpy())
        confidence = float(box.conf[0].cpu().numpy())
        
        # Get class info
        class_name = class_names[class_id]
        color = colors[class_id]
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Draw label background
        label = f"{class_name} {confidence:.2f}"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        
        # Draw text
        draw.text((x1, y1), label, fill=(255, 255, 255), font=font)
    
    return image

# -------------------------
# Main Processing
# -------------------------
if uploaded_file is not None:
    # Create two columns for display
    col1, col2 = st.columns(2)
    
    with col1:
        # Load and display original image
        original_image = Image.open(uploaded_file).convert("RGB")
        st.subheader("📷 Original Image")
        st.image(original_image, use_container_width=True, caption=f"Size: {original_image.size}")
    
    with col2:
        # Convert to numpy array for YOLO
        img_array = np.array(original_image)
        
        with st.spinner("🔍 Detecting objects..."):
            try:
                # Run inference
                results = model(
                    img_array,
                    conf=confidence_threshold,
                    iou=iou_threshold,
                    verbose=False,
                    device='cpu'  # Force CPU to avoid any CUDA issues
                )
                
                # Process results
                if results and len(results) > 0:
                    result = results[0]
                    
                    # Create a copy for drawing
                    annotated_image = original_image.copy()
                    
                    # Check if we have detections
                    if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                        # Draw bounding boxes
                        annotated_image = draw_bounding_boxes(
                            annotated_image,
                            result.boxes,
                            result.names
                        )
                        
                        # Display annotated image
                        st.subheader("🎯 Detected Objects")
                        st.image(annotated_image, use_container_width=True)
                        
                        # Statistics
                        num_detections = len(result.boxes)
                        st.markdown(f"### ✅ Found {num_detections} objects")
                        
                        # Detailed results in expander
                        with st.expander("📊 View Detailed Detection Results", expanded=True):
                            # Count objects by class
                            class_counts = {}
                            confidences_by_class = {}
                            
                            for box in result.boxes:
                                class_id = int(box.cls[0].cpu().numpy())
                                class_name = result.names[class_id]
                                confidence = float(box.conf[0].cpu().numpy())
                                
                                if class_name not in class_counts:
                                    class_counts[class_name] = 0
                                    confidences_by_class[class_name] = []
                                
                                class_counts[class_name] += 1
                                confidences_by_class[class_name].append(confidence)
                            
                            # Display summary
                            for class_name, count in sorted(class_counts.items()):
                                avg_conf = np.mean(confidences_by_class[class_name])
                                st.write(f"**{class_name}**: {count} objects (avg confidence: {avg_conf:.1%})")
                        
                        # Download button for results
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            annotated_image.save(tmp_file.name, format='JPEG', quality=95)
                            tmp_file_path = tmp_file.name
                        
                        with open(tmp_file_path, 'rb') as f:
                            st.download_button(
                                label="📥 Download Annotated Image",
                                data=f,
                                file_name=f"yolov8_detection_{int(time.time())}.jpg",
                                mime="image/jpeg"
                            )
                        
                        # Clean up temp file
                        try:
                            os.unlink(tmp_file_path)
                        except:
                            pass
                        
                    else:
                        st.warning("⚠️ No objects detected. Try lowering the confidence threshold.")
                        st.image(original_image, use_container_width=True, caption="No objects detected")
                
                else:
                    st.error("❌ No results returned from model")
                    
            except Exception as e:
                st.error(f"❌ Error during detection: {str(e)}")
                st.info("💡 If this is an OpenCV error, try the following:")
                st.code("""
                # In your terminal:
                pip uninstall opencv-python opencv-python-headless -y
                pip install ultralytics pillow numpy
                """)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Powered by YOLOv8 | Built with Streamlit</p>
    <p>No OpenCV required! Uses pure Python with Pillow for visualization.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Troubleshooting Section (Hidden by default)
# -------------------------
with st.expander("🔧 Troubleshooting"):
    st.markdown("""
    ### Common Issues and Solutions:
    
    1. **Model loading fails**
       - Check internet connection for model download
       - Ensure you have write permissions
       
    2. **Slow performance**
       - Use smaller images
       - The app runs on CPU in cloud environments
       
    3. **Memory issues**
       - Upload smaller images
       - Close other browser tabs
       
    4. **No objects detected**
       - Lower the confidence threshold
       - Try images with clear, visible objects
       
    ### For Local Development:
    ```bash
    # Create virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate
    
    # Install requirements
    pip install streamlit ultralytics pillow numpy
    
    # Run the app
    streamlit run app.py
    ```
    """)
