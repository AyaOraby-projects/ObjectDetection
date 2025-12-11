import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import time
import shutil
import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# Try to import DeepFace with proper error handling
# -------------------------
deepface_available = False
DeepFace = None

try:
    # First, check if tf_keras is available
    try:
        import tf_keras
        print("tf_keras is available")
    except ImportError:
        print("tf_keras not found, trying to import tensorflow directly")
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
    
    # Now try to import DeepFace
    from deepface import DeepFace
    deepface_available = True
    print("DeepFace imported successfully")
    
except Exception as e:
    st.warning(f"DeepFace import error: {e}")
    st.info("""
    **Installation required:** Run these commands in your terminal:
    ```bash
    pip install tf-keras
    pip install deepface
    ```
    Or use the simplified version below.
    """)
    deepface_available = False

# -------------------------
# Fallback Face Detection (if DeepFace not available)
# -------------------------
def detect_faces_opencv(image_np):
    """Detect faces using OpenCV Haar Cascade"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Load face cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            return []
        
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    except Exception as e:
        st.error(f"Face detection error: {e}")
        return []

# -------------------------
# Simple Emotion Detection using OpenCV DNN
# -------------------------
def load_emotion_model():
    """Load pre-trained emotion detection model"""
    try:
        # Emotion model from OpenCV
        model_path = "emotion_net.caffemodel"
        config_path = "deploy.prototxt.txt"
        
        # If files don't exist, we'll use a simpler approach
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            return None
        
        net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        return net
    except:
        return None

# Emotion labels for the model
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_emotion_simple(face_img):
    """Simple emotion detection using facial features"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        
        # Simple heuristic based on facial features
        # This is a simplified approach - in production, use a trained model
        
        # Get face dimensions
        h, w = gray.shape
        
        # Calculate average brightness (simulating smile detection)
        brightness = np.mean(gray)
        
        # Calculate face symmetry
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        symmetry = np.abs(np.mean(left_half) - np.mean(right_half))
        
        # Simple rules for emotion detection
        if brightness > 150 and symmetry < 30:
            return "Happy", 0.7
        elif brightness < 100:
            return "Sad", 0.6
        elif symmetry > 50:
            return "Surprise", 0.6
        else:
            return "Neutral", 0.5
            
    except:
        return "Unknown", 0.0

def detect_gender_simple(face_img):
    """Simple gender detection based on facial features"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        
        # Get face dimensions
        h, w = gray.shape
        
        # Calculate face shape features
        aspect_ratio = w / h
        jaw_width = w
        
        # Simple heuristics (these are just placeholders)
        # In production, use a trained gender detection model
        
        # Faces with wider jaw are often male
        if jaw_width > h * 0.8:
            return "Male", 0.6
        else:
            return "Female", 0.6
            
    except:
        return "Unknown", 0.0

# -------------------------
# App UI Configuration
# -------------------------
st.set_page_config(page_title="Advanced Detection App", layout="centered")
st.title("🎯 Advanced Object & Face Analysis")

st.markdown(
    """
Upload an image and the app will run:
1. **YOLOv8 Object Detection** - Detect 80+ object categories
2. **Face Detection & Analysis** - Detect faces with basic emotion and gender analysis
"""
)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Detection confidence
    confidence = st.slider(
        "Detection Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        help="Higher values mean more confident detections"
    )
    
    # Face analysis options
    enable_face_analysis = st.checkbox("Enable Face Analysis", value=True)
    
    if not deepface_available and enable_face_analysis:
        st.warning("Using simplified face analysis")

# -------------------------
# Load YOLO Model (cached)
# -------------------------
@st.cache_resource
def load_yolo_model():
    """Load YOLOv8 model for object detection"""
    return YOLO("yolov8n.pt")

# -------------------------
# Main Application
# -------------------------
with st.spinner("Loading object detection model..."):
    yolo_model = load_yolo_model()
    st.success("✅ Object detection model loaded")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📸 Input", "🎯 Objects", "😊 Faces", "🎥 Real-Time"])
    
    with tab1:
        # Show uploaded image
        input_image = Image.open(uploaded_file).convert("RGB")
        st.subheader("Input Image")
        
        # Display image info
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric("Width", f"{input_image.width}px")
        with col_info2:
            st.metric("Height", f"{input_image.height}px")
        
        st.image(input_image, width="stretch")
    
    # Convert to numpy array
    image_np = np.array(input_image)
    
    # Save input temporarily
    temp_input_dir = "temp_inputs"
    os.makedirs(temp_input_dir, exist_ok=True)
    input_path = os.path.join(temp_input_dir, f"input_{int(time.time())}.png")
    input_image.save(input_path)
    
    # Clear previous runs
    out_root = "runs/detect"
    if os.path.exists(out_root):
        try:
            shutil.rmtree(out_root)
        except Exception:
            pass
    
    # Run YOLO object detection
    with st.spinner("Detecting objects..."):
        yolo_results = yolo_model(input_path, save=True, conf=confidence)
    
    # Run face analysis if enabled
    face_results = []
    if enable_face_analysis:
        with st.spinner("Detecting faces..."):
            if deepface_available:
                # Use DeepFace for advanced analysis
                try:
                    analyses = DeepFace.analyze(
                        img_path=image_np,
                        actions=['emotion', 'gender'],
                        detector_backend='opencv',
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if not isinstance(analyses, list):
                        analyses = [analyses]
                    
                    for analysis in analyses:
                        if 'region' in analysis:
                            x, y, w, h = (
                                analysis['region']['x'],
                                analysis['region']['y'],
                                analysis['region']['w'],
                                analysis['region']['h']
                            )
                            
                            # Get emotion
                            emotion = analysis.get('dominant_emotion', 'Unknown')
                            emotion_conf = analysis.get('emotion', {})
                            if emotion_conf and emotion in emotion_conf:
                                emotion_score = emotion_conf[emotion]
                            else:
                                emotion_score = 0
                            
                            # Get gender
                            gender = analysis.get('dominant_gender', 'Unknown')
                            gender_conf = analysis.get('gender', {})
                            if gender_conf and gender in gender_conf:
                                gender_score = gender_conf[gender]
                            else:
                                gender_score = 0
                            
                            face_results.append({
                                'bbox': (x, y, w, h),
                                'emotion': emotion,
                                'emotion_score': float(emotion_score),
                                'gender': gender,
                                'gender_score': float(gender_score)
                            })
                except Exception as e:
                    st.error(f"DeepFace error: {e}. Using fallback detection.")
                    # Fallback to OpenCV
                    faces = detect_faces_opencv(image_np)
                    for (x, y, w, h) in faces:
                        face_roi = image_np[y:y+h, x:x+w]
                        emotion, emotion_score = detect_emotion_simple(face_roi)
                        gender, gender_score = detect_gender_simple(face_roi)
                        
                        face_results.append({
                            'bbox': (x, y, w, h),
                            'emotion': emotion,
                            'emotion_score': emotion_score,
                            'gender': gender,
                            'gender_score': gender_score
                        })
            else:
                # Use OpenCV fallback
                faces = detect_faces_opencv(image_np)
                for (x, y, w, h) in faces:
                    face_roi = image_np[y:y+h, x:x+w]
                    emotion, emotion_score = detect_emotion_simple(face_roi)
                    gender, gender_score = detect_gender_simple(face_roi)
                    
                    face_results.append({
                        'bbox': (x, y, w, h),
                        'emotion': emotion,
                        'emotion_score': emotion_score,
                        'gender': gender,
                        'gender_score': gender_score
                    })
    
    # Display results in tabs
    with tab2:
        # Locate YOLO output
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
        
        if output_image_path:
            st.subheader("Object Detection Results")
            st.image(output_image_path, width="stretch")
            
            # Show detection statistics
            if len(yolo_results) > 0:
                detections = yolo_results[0].boxes
                if detections is not None and len(detections) > 0:
                    st.info(f"**Detected {len(detections)} objects**")
                    
                    # Count objects by class
                    class_counts = {}
                    for i in range(len(detections)):
                        cls = int(detections.cls[i])
                        class_name = yolo_model.names[cls]
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    # Display top objects
                    if class_counts:
                        st.write("**Top detected objects:**")
                        for obj, count in list(sorted(class_counts.items(), key=lambda x: x[1], reverse=True))[:5]:
                            st.write(f"- {obj}: {count}")
            
            # Download button
            with open(output_image_path, "rb") as f:
                st.download_button(
                    label="📥 Download Object Detection",
                    data=f,
                    file_name="object_detection.png",
                    mime="image/png",
                )
        else:
            st.warning("No objects detected or output not found.")
    
    with tab3:
        if enable_face_analysis:
            st.subheader("Face Analysis Results")
            
            if face_results:
                # Create annotated image
                annotated_img = image_np.copy()
                
                for i, result in enumerate(face_results):
                    x, y, w, h = result['bbox']
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Prepare label
                    def format_score(score):
                        """Convert score to proper percentage between 0-100%"""
                        # Ensure score is between 0 and 1
                        if score > 1:  # If score is like 100 or 83.97
                            score = score / 100
                        elif score > 10:  # If score is like 839.6
                            score = score / 1000
                        score = max(0, min(1, score))  # Clamp between 0 and 1
                        return f"{score*100:.1f}%"

                    # Then use it:
                    label = f"Face {i+1}: {result['gender']} ({format_score(result['gender_score'])}), {result['emotion']} ({format_score(result['emotion_score'])})"
                    
                    # Draw label
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    
                    # Calculate text size for background
                    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                    
                    # Draw background rectangle
                    cv2.rectangle(annotated_img,
                                (x, y - text_size[1] - 10),
                                (x + text_size[0], y),
                                (0, 255, 0), -1)
                    
                    # Draw text
                    cv2.putText(annotated_img, label, (x, y - 5),
                              font, font_scale, (0, 0, 0), thickness)
                
                # Display annotated image
                st.image(annotated_img, channels="RGB", width="stretch")
                
        #         # Display face details in a table
        #         st.subheader("📊 Face Details")
                
        #         # Create data for table
        #         face_data = []
        #         for i, result in enumerate(face_results):
        #             face_data.append({
        #                 "Face #": i + 1,
        #                 "Gender": f"{result['gender']} ({result['gender_score']:.0%})",
        #                 "Emotion": f"{result['emotion']} ({result['emotion_score']:.0%})",
        #                 "Position": f"({result['bbox'][0]}, {result['bbox'][1]})",
        #                 "Size": f"{result['bbox'][2]}x{result['bbox'][3]}"
        #             })
                
        #         # Display table
        #         st.dataframe(face_data, use_container_width=True)
                
        #         # Summary statistics
        #         st.subheader("📈 Summary")
                
        #         if face_results:
        #             col1, col2, col3 = st.columns(3)
                    
        #             with col1:
        #                 st.metric("Total Faces", len(face_results))
                    
        #             with col2:
        #                 # Most common gender
        #                 genders = [r['gender'] for r in face_results]
        #                 if genders:
        #                     common_gender = max(set(genders), key=genders.count)
        #                     st.metric("Most Common Gender", common_gender)
                    
        #             with col3:
        #                 # Most common emotion
        #                 emotions = [r['emotion'] for r in face_results]
        #                 if emotions:
        #                     common_emotion = max(set(emotions), key=emotions.count)
        #                     st.metric("Most Common Emotion", common_emotion)
                
        #         # Download button
        #         _, img_encoded = cv2.imencode('.png', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        #         st.download_button(
        #             label="📥 Download Face Analysis",
        #             data=img_encoded.tobytes(),
        #             file_name="face_analysis.png",
        #             mime="image/png",
        #         )
        #     else:
        #         st.warning("No faces detected in the image.")
        #         if not deepface_available:
        #             st.info("""
        #             **Note:** Using basic face detection. For better results:
        #             1. Install DeepFace: `pip install tf-keras deepface`
        #             2. Ensure faces are clearly visible
        #             3. Use well-lit images
        #             """)
        # else:
        #     st.info("Face analysis is disabled. Enable it in the sidebar settings.")
    
    # Cleanup

    # -------------------------
# Tab 4: Real-Time Detection
# -------------------------
    with tab4:
        st.header("🎥 Real-Time Detection")
        st.markdown("Use your webcam for real-time object, face, gender, and emotion detection")
        
        # Configuration for real-time
        col_config1, col_config2, col_config3 = st.columns(3)
        
        with col_config1:
            detection_interval = st.slider("Detection Interval (ms)", 100, 1000, 300, 100)
        
        with col_config2:
            confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
        
        with col_config3:
            enable_face_recognition = st.checkbox("Enable Face Recognition", value=True)
            enable_object_detection = st.checkbox("Enable Object Detection", value=True)
        
        # Webcam and recording controls
        col_control1, col_control2, col_control3 = st.columns(3)
        
        with col_control1:
            start_button = st.button("▶️ Start Webcam", type="primary")
        
        with col_control2:
            stop_button = st.button("⏹️ Stop")
        
        with col_control3:
            capture_button = st.button("📸 Capture Frame")
        
        # Placeholders for display
        webcam_placeholder = st.empty()
        stats_placeholder = st.empty()
        capture_placeholder = st.empty()
        
        # Session state for webcam control
        if 'webcam_active' not in st.session_state:
            st.session_state.webcam_active = False
        if 'captured_frames' not in st.session_state:
            st.session_state.captured_frames = []
        if 'detection_stats' not in st.session_state:
            st.session_state.detection_stats = {
                'total_frames': 0,
                'faces_detected': 0,
                'objects_detected': 0,
                'emotions': {},
                'genders': {}
            }
        
        def update_stats(face_results, object_count):
            """Update detection statistics"""
            st.session_state.detection_stats['total_frames'] += 1
            
            if face_results:
                st.session_state.detection_stats['faces_detected'] += len(face_results)
                
                for result in face_results:
                    # Update emotion stats
                    emotion = result.get('emotion', 'Unknown')
                    st.session_state.detection_stats['emotions'][emotion] = \
                        st.session_state.detection_stats['emotions'].get(emotion, 0) + 1
                    
                    # Update gender stats
                    gender = result.get('gender', 'Unknown')
                    st.session_state.detection_stats['genders'][gender] = \
                        st.session_state.detection_stats['genders'].get(gender, 0) + 1
            
            if object_count > 0:
                st.session_state.detection_stats['objects_detected'] += object_count
        
        def draw_realtime_annotations(frame, face_results, yolo_results):
            """Draw annotations on real-time frame"""
            annotated_frame = frame.copy()
            
            # Draw face annotations
            for i, result in enumerate(face_results):
                x, y, w, h = result['bbox']
                
                # Draw face bounding box (green)
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Format scores properly
                gender_score = result.get('gender_score', 0)
                emotion_score = result.get('emotion_score', 0)
                
                # Ensure scores are between 0-1
                if gender_score > 1:
                    gender_score = gender_score / 100
                if emotion_score > 1:
                    emotion_score = emotion_score / 100
                
                # Prepare label
                gender_label = f"{result.get('gender', 'Unknown')} ({gender_score*100:.0f}%)"
                emotion_label = f"{result.get('emotion', 'Unknown')} ({emotion_score*100:.0f}%)"
                label = f"Face {i+1}: {gender_label}, {emotion_label}"
                
                # Draw label background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                
                cv2.rectangle(annotated_frame,
                            (x, max(0, y - text_size[1] - 10)),
                            (x + text_size[0], y),
                            (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(annotated_frame, label, (x, max(5, y - 5)),
                        font, font_scale, (0, 0, 0), thickness)
            
            # Draw object detections if available
            if yolo_results and hasattr(yolo_results[0], 'boxes'):
                detections = yolo_results[0].boxes
                if detections is not None:
                    for i in range(len(detections)):
                        box = detections.xyxy[i].cpu().numpy()
                        conf = detections.conf[i].cpu().numpy()
                        cls = int(detections.cls[i].cpu().numpy())
                        
                        if conf >= confidence_threshold:
                            x1, y1, x2, y2 = map(int, box)
                            
                            # Draw object bounding box (blue)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            
                            # Object label
                            label = f"{yolo_model.names[cls]} {conf:.2f}"
                            
                            # Draw label background
                            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                            cv2.rectangle(annotated_frame,
                                        (x1, max(0, y1 - text_size[1] - 10)),
                                        (x1 + text_size[0], y1),
                                        (255, 0, 0), -1)
                            
                            # Draw label text
                            cv2.putText(annotated_frame, label, (x1, max(5, y1 - 5)),
                                    font, font_scale, (0, 0, 0), thickness)
            
            return annotated_frame
        
        def process_realtime_frame(frame):
            """Process a single frame for real-time detection"""
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            face_results = []
            yolo_results = None
            
            # Face detection
            if enable_face_recognition and deepface_available:
                try:
                    analyses = DeepFace.analyze(
                        img_path=rgb_frame,
                        actions=['emotion', 'gender'],
                        detector_backend='opencv',
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if not isinstance(analyses, list):
                        analyses = [analyses]
                    
                    for analysis in analyses:
                        if 'region' in analysis:
                            x, y, w, h = (
                                analysis['region']['x'],
                                analysis['region']['y'],
                                analysis['region']['w'],
                                analysis['region']['h']
                            )
                            
                            emotion = analysis.get('dominant_emotion', 'Unknown')
                            gender = analysis.get('dominant_gender', 'Unknown')
                            
                            # Get confidence scores
                            emotion_conf = analysis.get('emotion', {}).get(emotion, 50)
                            gender_conf = analysis.get('gender', {}).get(gender, 50)
                            
                            face_results.append({
                                'bbox': (x, y, w, h),
                                'emotion': emotion,
                                'emotion_score': float(emotion_conf),
                                'gender': gender,
                                'gender_score': float(gender_conf)
                            })
                except Exception as e:
                    st.warning(f"Face detection error: {e}")
            
            # Object detection
            if enable_object_detection:
                try:
                    yolo_results = yolo_model(rgb_frame, conf=confidence_threshold, verbose=False)
                except Exception as e:
                    st.warning(f"Object detection error: {e}")
            
            # Draw annotations
            annotated_frame = draw_realtime_annotations(frame, face_results, yolo_results)
            
            # Update statistics
            object_count = len(yolo_results[0].boxes) if yolo_results and yolo_results[0].boxes else 0
            update_stats(face_results, object_count)
            
            return annotated_frame, len(face_results), object_count
        
        # Handle button clicks
        if start_button:
            st.session_state.webcam_active = True
            
            # Initialize webcam
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Cannot access webcam. Please check your camera permissions.")
                st.session_state.webcam_active = False
            else:
                st.success("Webcam started! Processing real-time video...")
                
                # Set camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # Real-time processing loop
                last_detection_time = time.time()
                
                while st.session_state.webcam_active:
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.error("Failed to capture frame")
                        break
                    
                    current_time = time.time()
                    
                    # Process frame at specified interval
                    if current_time - last_detection_time >= (detection_interval / 1000):
                        # Process the frame
                        processed_frame, face_count, object_count = process_realtime_frame(frame)
                        
                        # Convert to RGB for Streamlit display
                        display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        
                        # Display the frame
                        webcam_placeholder.image(display_frame, channels="RGB", use_column_width=True)
                        
                        # Update statistics display
                        stats_text = f"""
                        **Real-time Stats:**
                        - Faces detected: {face_count}
                        - Objects detected: {object_count}
                        - Total frames processed: {st.session_state.detection_stats['total_frames']}
                        """
                        stats_placeholder.markdown(stats_text)
                        
                        last_detection_time = current_time
                    
                    # Check for stop signal
                    if stop_button:
                        st.session_state.webcam_active = False
                        break
                    
                    # Small delay to prevent high CPU usage
                    time.sleep(0.01)
                
                # Release webcam
                cap.release()
                cv2.destroyAllWindows()
                st.success("Webcam stopped")
        
        if stop_button:
            st.session_state.webcam_active = False
            st.info("Webcam stopped")
        
        if capture_button and st.session_state.webcam_active:
            # Capture current frame (you would need to store the last frame)
            st.info("Capture functionality requires storing the last processed frame")
        
        # Display overall statistics
        if st.session_state.detection_stats['total_frames'] > 0:
            st.subheader("📊 Session Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Frames", st.session_state.detection_stats['total_frames'])
            
            with col2:
                st.metric("Total Faces", st.session_state.detection_stats['faces_detected'])
            
            with col3:
                st.metric("Total Objects", st.session_state.detection_stats['objects_detected'])
            
            # Emotion distribution
            if st.session_state.detection_stats['emotions']:
                st.write("**Emotion Distribution:**")
                emotion_data = st.session_state.detection_stats['emotions']
                st.bar_chart(emotion_data)
            
            # Gender distribution
            if st.session_state.detection_stats['genders']:
                st.write("**Gender Distribution:**")
                gender_data = st.session_state.detection_stats['genders']
                st.bar_chart(gender_data)
        
        # Download session data button
        if st.session_state.detection_stats['total_frames'] > 0:
            if st.button("💾 Download Session Report"):
                # Create a simple report
                report = f"""
                Real-Time Detection Session Report
                ==================================
                Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                
                Statistics:
                - Total frames processed: {st.session_state.detection_stats['total_frames']}
                - Total faces detected: {st.session_state.detection_stats['faces_detected']}
                - Total objects detected: {st.session_state.detection_stats['objects_detected']}
                
                Emotion Distribution:
                {st.session_state.detection_stats['emotions']}
                
                Gender Distribution:
                {st.session_state.detection_stats['genders']}
                """
                
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
       
        # Requirements note
        if not deepface_available:
            st.warning("""
            ⚠️ **Face recognition requires DeepFace**
            
            Install with:
            ```bash
            pip install deepface
            ```
            
            Currently, only object detection is available in real-time mode.
            """)
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except:
            pass

        else:
            st.info("👆 Please upload an image to get started!")
            
        # Installation instructions

        
        # Example images
        
