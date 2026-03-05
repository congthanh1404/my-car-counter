import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np

st.set_page_config(page_title="Vehicle Counter", layout="wide")
st.title("🚗 Hệ thống Nhận dạng & Đếm xe")

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

uploaded_file = st.file_uploader("Tải video giao thông lên...", type=["mp4", "avi"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    st_frame = st.empty()
    counter = 0
    tracked_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        h, w, _ = frame.shape
        line_y = int(h * 0.7) # Vạch đếm ở 70% khung hình
        
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().numpy()
            
            for box, id in zip(boxes, ids):
                cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                if line_y - 5 < cy < line_y + 5:
                    if id not in tracked_ids:
                        tracked_ids.add(id)
                        counter += 1
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        
        cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 3)
        cv2.putText(frame, f"Count: {counter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    cap.release()