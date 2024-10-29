import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLO model
model = YOLO("yolo11n.pt")

# Streamlit app
st.title("Object Detection App")

# Start webcam detection button
run_detection = st.checkbox("Run Webcam Detection")

# Placeholder for video feed
video_placeholder = st.empty()


# Webcam capture function
def get_frame():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return None

    ret, frame = cap.read()
    if not ret:
        st.error("Error: Could not read frame.")
        cap.release()
        return None

    # Perform object detection
    results = model.predict(frame)

    # Render detection results on the frame
    annotated_frame = results[0].plot()

    # Convert the frame to RGB (from BGR, which OpenCV uses)
    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    cap.release()
    return rgb_frame


# Run detection when checkbox is checked
if run_detection:
    while True:
        # Capture and display webcam frame with detections
        frame = get_frame()
        if frame is not None:
            video_placeholder.image(frame, channels="RGB")

        # Break the loop when 'q' is pressed or stop button is checked off
        if not st.checkbox("Stop Detection", value=False):
            break