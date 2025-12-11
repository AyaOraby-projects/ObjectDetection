# Real-Time Detection Application Documentation

## Overview

This application performs **real-time object detection, face detection,
gender prediction, and emotion analysis** using the webcam.\
It uses **YOLO**, **OpenCV**, and optionally **DeepFace** to analyze
each frame and produce statistics and a downloadable session report.

------------------------------------------------------------------------

## Features

### ✔ Object Detection

Uses **YOLO** to detect common objects such as persons, cars, phones,
etc.

### ✔ Face Detection

Uses YOLO's `face` model to detect faces in real time.

### ✔ Emotion Recognition

If **DeepFace** is installed, the app uses its pre-trained models:\
- happy\
- sad\
- angry\
- neutral\
- surprise\
- fear\
- disgust

If DeepFace is *not installed*, it falls back to **a simple
pixel-intensity based emotion classifier**.

### ✔ Gender Prediction

If DeepFace is installed → uses its gender model.\
If not → uses a fallback brightness-based estimation.

### ✔ Real-Time Statistics

While the webcam is running, the app logs: - number of frames processed\
- total objects detected\
- total faces detected\
- emotion distribution\
- gender distribution

### ✔ Downloadable Session Report

When the session ends, a `.txt` report can be generated summarizing the
entire run.

------------------------------------------------------------------------

## Technology Stack

  Component             Library         Purpose
  --------------------- --------------- ------------------------------
  YOLO                  `ultralytics`   Object & face detection
  OpenCV                `cv2`           Webcam input & drawing
  DeepFace (optional)   `deepface`      Emotion & gender recognition
  Streamlit             `streamlit`     Web interface
  Python                3.8+            Main language

------------------------------------------------------------------------

## Models Used

### **1. YOLO Face Detection Model**

-   Inputs: webcam frames\
-   Outputs: bounding boxes of faces\
-   Fast and efficient for real-time performance

### **2. YOLO Object Detection Model**

Pre-trained YOLO model detects: - persons\
- vehicles\
- objects\
- electronics\
- household items

### **3. DeepFace (Optional Models)**

If installed, DeepFace provides: - Emotion model\
- Gender model

These models are more accurate than the fallback alternatives.

------------------------------------------------------------------------

## Installation Guide

### **Step 1 -- Clone Your Project Folder**

``` bash
mkdir realtime_detection
cd realtime_detection
```

### **Step 2 -- Install Dependencies**

``` bash
pip install streamlit ultralytics opencv-python numpy
pip install deepface     # Optional, for higher accuracy
```

### **Step 3 -- Run the Application**

Save your Python file (for example `app.py`) and run:

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## Application Workflow

### **1. User Starts Webcam Session**

-   Camera feed begins\
-   YOLO loads models\
-   Each frame is processed

### **2. Object Detection**

YOLO detects objects with: - bounding boxes\
- labels\
- confidence scores

### **3. Face Detection**

YOLO face model identifies all faces in the frame.

### **4. Emotion & Gender Prediction**

For each face: - If DeepFace is available → accurate prediction\
- Else → fallback model

### **5. Aggregation of Statistics**

Every frame adds to: - `faces_detected` - `objects_detected` - emotion
counts - gender counts

### **6. User Stops Webcam**

-   Feed stops\
-   Summary stats displayed

### **7. User Downloads Report**

A structured `.txt` file includes: - Total frames\
- Total faces\
- Total objects\
- All emotions count\
- All gender count\
- Timestamp

------------------------------------------------------------------------

## Folder Structure Example

    realtime_detection/
    │
    ├── app.py               # Streamlit application
    ├── session_report.txt   # Generated after webcam session
    ├── models/              # (Optional) YOLO or DeepFace custom models
    └── app_documentation.md # Generated documentation

------------------------------------------------------------------------

## Known Limitations

-   Fallback emotion/gender model is much less accurate than DeepFace.
-   Performance depends on CPU/GPU.
-   YOLO may detect small faces inaccurately.
-   Lighting changes may affect predictions.

------------------------------------------------------------------------

## Recommended Improvements

-   Add face tracking (assign IDs).
-   Add person re-identification.
-   Replace fallback model with a CNN.
-   Save detected frames as images.
-   Add GPU support for faster inference.

------------------------------------------------------------------------

## Conclusion

This real-time detection app is a flexible, extendable project that
combines object detection, face analysis, and emotion recognition into a
single interface. You can enhance it by integrating more advanced models
or expanding the dashboard.

Enjoy using and improving your project 🚀
