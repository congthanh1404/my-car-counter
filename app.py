import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile

# --- THIẾT LẬP GIAO DIỆN ---
st.set_page_config(page_title="AI Traffic Monitor", layout="centered")

# CSS tùy chỉnh để làm giao diện mượt mà và êm mắt hơn
st.markdown("""
    <style>
    /* Nền tối sâu và phông chữ sạch */
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    
    /* Làm đẹp khung tải file */
    .stFileUploader { border: 1px dashed #4b4b4b; border-radius: 10px; padding: 10px; }
    
    /* Khối hiển thị số lượng xe */
    .counter-card {
        background-color: #1f2937;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid #374151;
        margin-bottom: 20px;
    }
    .counter-value {
        color: #10b981; /* Màu xanh lá dịu mắt */
        font-size: 60px;
        font-weight: bold;
        margin: 0;
    }
    
    /* Footer tác giả */
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #9ca3af;
        font-size: 14px;
        border-top: 1px solid #374151;
        padding-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- TIÊU ĐỀ CHÍNH ---
st.markdown("<h2 style='text-align: center;'>🚗 Hệ thống Nhận dạng & Đếm xe</h2>", unsafe_allow_html=True)

# --- KHU VỰC CÀI ĐẶT (ẨN TRONG EXPANDER) ---
with st.expander("🛠 Cài đặt hệ thống (Tùy chỉnh vạch đếm & độ nhạy)"):
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        line_pos = st.slider("Vị trí vạch ngang (%)", 0, 100, 70)
    with col_cfg2:
        conf_thresh = st.slider("Độ nhạy mô hình (Confidence)", 0.1, 1.0, 0.25)

# --- KHU VỰC TẢI VIDEO ---
uploaded_file = st.file_uploader("Chọn video giao thông để bắt đầu...", type=["mp4", "mov", "avi"])

if uploaded_file:
    # Hiển thị số lượng xe ở phía trên video cho dễ nhìn
    count_container = st.container()
    with count_container:
        st.markdown('<div class="counter-card">', unsafe_allow_html=True)
        st.write("📊 TỔNG PHƯƠNG TIỆN")
        count_display = st.empty()
        count_display.markdown('<p class="counter-value">0</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Xử lý Video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    video_output = st.empty()
    
    # Load Model
    @st.cache_resource
    def load_yolo(): return YOLO('yolov8n.pt')
    model = load_yolo()
    
    counter = 0
    tracked_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        h, w, _ = frame.shape
        line_y = int(h * (line_pos / 100))
        
        # AI Tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf_thresh, verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().numpy()
            
            for box, id in zip(boxes, ids):
                cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                
                # Vẽ box mỏng và tinh tế hơn
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (16, 185, 129), 1)
                
                if line_y - 8 < cy < line_y + 8:
                    if id not in tracked_ids:
                        tracked_ids.add(id)
                        counter += 1
        
        # Vẽ vạch đếm mảnh
        cv2.line(frame, (0, line_y), (w, line_y), (239, 68, 68), 2)
        
        # Cập nhật kết quả
        count_display.markdown(f'<p class="counter-value">{counter}</p>', unsafe_allow_html=True)
        video_output.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    cap.release()

# --- FOOTER ---
st.markdown(f"<div class='footer'>Thiết kế bởi: Trương Công Thành - 223332852</div>", unsafe_allow_html=True)
