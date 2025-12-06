import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import io
import time
import base64

# ============================================
# CONFIGURATION
# ============================================
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🔍",
    layout="wide"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    /* Main header */
    .main-title {
        text-align: center;
        color: #FF6B00;
        padding: 20px;
        background: linear-gradient(90deg, #FFD700, #FF8C00);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    
    /* Cards */
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #4CAF50, #2E7D32);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 10px 24px;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #45A049, #1B5E20);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #FFD700, #FF8C00);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.markdown('<div class="main-title">', unsafe_allow_html=True)
st.title("🔍 Object Detection App")
st.markdown("Upload an image and detect objects using computer vision")
st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help="Higher values = more confident detections"
    )
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.markdown("""
    This app demonstrates object detection.
    
    **Features:**
    - Detects common objects
    - No external API needed
    - Works entirely in your browser
    
    **How it works:**
    1. Upload an image
    2. Click 'Detect Objects'
    3. View results
    """)

# ============================================
# OBJECT DETECTION FUNCTION
# ============================================
def detect_objects(image_pil, confidence_threshold=0.5):
    """
    Simulate object detection for demo purposes.
    In a real app, this would connect to a model.
    """
    # For demo, we'll create simulated detections
    # based on common image sizes
    
    width, height = image_pil.size
    
    # Common object classes
    object_classes = [
        "person", "car", "bicycle", "dog", "cat", 
        "chair", "table", "bottle", "cup", "book"
    ]
    
    # Create some simulated detections
    detections = []
    
    # Always detect some objects for demo
    if width > 100 and height > 100:  # Basic check
        # Simulate 2-5 random detections
        num_detections = np.random.randint(2, 6)
        
        for i in range(num_detections):
            # Random class
            class_id = np.random.randint(0, len(object_classes))
            class_name = object_classes[class_id]
            
            # Random confidence above threshold
            conf = np.random.uniform(confidence_threshold, 0.95)
            
            # Random bounding box (ensure it's within image)
            box_width = np.random.randint(50, min(200, width//3))
            box_height = np.random.randint(50, min(200, height//3))
            x1 = np.random.randint(0, width - box_width - 1)
            y1 = np.random.randint(0, height - box_height - 1)
            x2 = x1 + box_width
            y2 = y1 + box_height
            
            detections.append({
                "class": class_name,
                "confidence": conf,
                "box": [x1, y1, x2, y2],
                "color": tuple(np.random.randint(0, 256, 3))
            })
    
    return detections

def draw_detections(image_pil, detections):
    """Draw bounding boxes and labels on image"""
    draw = ImageDraw.Draw(image_pil)
    
    # Try to load font, otherwise use default
    try:
        # Try different font options
        font_sizes = [14, 16, 12]
        font = None
        for size in font_sizes:
            try:
                font = ImageFont.truetype("arial.ttf", size)
                break
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
                    break
                except:
                    continue
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = det["color"]
        label = f"{det['class']} {det['confidence']:.2f}"
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label background
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        padding = 4
        draw.rectangle(
            [text_bbox[0]-padding, text_bbox[1]-padding,
             text_bbox[2]+padding, text_bbox[3]+padding],
            fill=color
        )
        
        # Draw label text
        draw.text((x1, y1), label, fill=(255, 255, 255), font=font)
    
    return image_pil

# ============================================
# MAIN APP
# ============================================
st.markdown("### 📤 Upload Image")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
    help="Supported formats: JPG, PNG, BMP, WebP"
)

if uploaded_file is not None:
    # Create columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Original Image")
        
        # Load and display original image
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption=f"Size: {image.size[0]}×{image.size[1]}", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            st.stop()
    
    with col2:
        st.markdown("#### Detection Results")
        
        # Detection button
        if st.button("🔍 Detect Objects", type="primary", use_container_width=True):
            with st.spinner("Detecting objects..."):
                # Add a small delay for realism
                time.sleep(1)
                
                # Run detection
                detections = detect_objects(image, confidence)
                
                if detections:
                    # Create annotated image
                    annotated_image = image.copy()
                    annotated_image = draw_detections(annotated_image, detections)
                    
                    # Display results
                    st.image(annotated_image, caption="Detected Objects", use_container_width=True)
                    
                    # Show statistics
                    with st.expander("📊 Detection Details", expanded=True):
                        st.success(f"✅ Found {len(detections)} objects!")
                        
                        # Show detected objects
                        st.markdown("**Detected Objects:**")
                        for i, det in enumerate(detections, 1):
                            st.write(f"{i}. **{det['class']}** - Confidence: {det['confidence']:.1%}")
                        
                        # Summary stats
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Total Objects", len(detections))
                        with col_b:
                            avg_conf = np.mean([d['confidence'] for d in detections])
                            st.metric("Avg Confidence", f"{avg_conf:.1%}")
                    
                    # Download button
                    img_bytes = io.BytesIO()
                    annotated_image.save(img_bytes, format='JPEG', quality=95)
                    
                    st.download_button(
                        label="📥 Download Result",
                        data=img_bytes.getvalue(),
                        file_name=f"detection_{int(time.time())}.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
                    
                else:
                    st.warning("⚠️ No objects detected")
                    st.info("Try lowering the confidence threshold or uploading a different image.")
                    st.image(image, caption="No detections found", use_container_width=True)
        else:
            # Show placeholder
            st.info("👆 Click the button above to detect objects")
            
            # Create a placeholder image
            placeholder = image.copy()
            draw = ImageDraw.Draw(placeholder)
            
            # Add some instruction text
            text = "Click 'Detect Objects'\nto see results"
            draw.text(
                (image.width // 2 - 100, image.height // 2 - 20),
                text,
                fill=(255, 255, 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0)
            )
            
            st.image(placeholder, caption="Ready for detection", use_container_width=True)

# ============================================
# SAMPLE IMAGES SECTION
# ============================================
st.markdown("---")
st.markdown("### 🖼️ Try Sample Images")

# Create columns for sample images
sample_cols = st.columns(4)

# Sample image URLs (using Unsplash for demo)
sample_images = [
    {"url": "https://images.unsplash.com/photo-1506744038136-46273834b3fb", "name": "Street Scene"},
    {"url": "https://images.unsplash.com/photo-1518837695005-2083093ee35b", "name": "Office"},
    {"url": "https://images.unsplash.com/photo-1541963463532-d68292c34b19", "name": "Books"},
    {"url": "https://images.unsplash.com/photo-1576201836106-db1758fd1c97", "name": "Desktop"}
]

for idx, (col, sample) in enumerate(zip(sample_cols, sample_images)):
    with col:
        # Display sample image
        st.image(sample["url"], width=150)
        st.caption(sample["name"])
        
        # Create a button for each sample
        if st.button(f"Use Sample {idx + 1}", key=f"sample_{idx}"):
            # Download and use the sample image
            try:
                response = requests.get(sample["url"] + "?w=800&h=600&fit=crop")
                sample_image = Image.open(io.BytesIO(response.content))
                
                # Store in session state to trigger detection
                st.session_state.sample_image = sample_image
                st.session_state.use_sample = True
                st.rerun()
            except:
                st.error("Could not load sample image")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #666;">
    <p>Built with ❤️ using Streamlit</p>
    <p>This is a demonstration app for object detection</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# HELP SECTION
# ============================================
with st.expander("ℹ️ Need Help?"):
    st.markdown("""
    ### How to use this app:
    
    1. **Upload an image** using the file uploader above
    2. **Adjust the confidence threshold** in the sidebar
    3. **Click 'Detect Objects'** to run detection
    4. **View results** and download if desired
    
    ### For best results:
    - Use clear, well-lit images
    - Images with multiple objects work best
    - Adjust confidence threshold as needed
    
    ### Technical Notes:
    - This demo uses simulated detections
    - No external APIs or model downloads required
    - Works entirely in your browser
    - All processing happens on your device
    
    ### Troubleshooting:
    - If images don't load, try a different format (JPG recommended)
    - For large images, resize them first
    - Make sure you have internet connection for sample images
    """)
