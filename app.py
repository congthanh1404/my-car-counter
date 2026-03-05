import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="AI Vehicle Detection", layout="wide")

# CSS để làm đẹp giao diện và tạo các khối hộp (Card)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stApp { color: white; }
    .status-box {
        background-color: #262730;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .author-footer {
        text-align: center;
        padding: 20px;
        font-size: 18px;
        font-weight: bold;
        color: #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

# --- TIÊU ĐỀ ---
st.markdown("<h1 style='text-align: center;'> Nhận dạng và đếm phương tiện xe</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Tải lên video, hệ thống sẽ tự động nhận diện và đếm phương tiện.</p>", unsafe_allow_html=True)

# --- SIDEBAR: CẤU HÌNH ---
with st.sidebar:
    st.header("⚙️ Cấu hình hệ thống")
    line_pos = st.slider("Vị trí vạch đếm (%)", 0, 100, 70)
    conf_thresh = st.slider("Độ tin cậy mô hình", 0.0, 1.0, 0.25)
    st.divider()
    st.info("Hệ thống sử dụng mô hình Deep Learning YOLOv8 để phát hiện vật thể.")

# --- BỐ CỤC CHÍNH (2 CỘT) ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📹 Video đầu vào")
    uploaded_file = st.file_uploader("", type=["mp4", "avi", "mov"])

with col2:
    st.subheader("🖼 Video đã xử lý")
    output_frame = st.empty() # Khung để hiển thị streaming video

# --- KHỐI HIỂN THỊ KẾT QUẢ ---
st.markdown("---")
res_col1, res_col2 = st.columns([2, 1])
with res_col2:
    st.markdown('<div class="status-box">', unsafe_allow_html=True)
    st.write("📊 **TỔNG SỐ XE ĐÃ ĐẾM**")
    count_text = st.empty()
    count_text.markdown("<h2 style='color: #ff4b4b; font-size: 48px;'>0</h2>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- LOGIC XỬ LÝ AI ---
@st.cache_resource
def get_model():
    return YOLO('yolov8n.pt')

model = get_model()

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    counter = 0
    tracked_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        h, w, _ = frame.shape
        line_y = int(h * (line_pos / 100))
        
        # Chạy mô hình Tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf_thresh, verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().numpy()
            
            for box, id in zip(boxes, ids):
                cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                
                # Vẽ khung nhận diện
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 75, 75), 2)
                
                # Logic đếm xe đi qua vạch
                if line_y - 10 < cy < line_y + 10:
                    if id not in tracked_ids:
                        tracked_ids.add(id)
                        counter += 1
        
        # Vẽ vạch đếm (Xanh lá)
        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 0), 3)
        
        # Hiển thị kết quả real-time
        count_text.markdown(f"<h2 style='color: #ff4b4b; font-size: 48px;'>{counter}</h2>", unsafe_allow_html=True)
        output_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    cap.release()

# --- FOOTER TÁC GIẢ ---
st.markdown(f"<div class='author-footer'>Author: Trương Công Thành - 223332852</div>", unsafe_allow_html=True)
