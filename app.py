import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile

# --- THIẾT LẬP GIAO DIỆN ---
st.set_page_config(page_title="AI Traffic Monitor Pro", layout="centered")

# CSS để tối ưu hóa khả năng đọc (Readability) và màu sắc êm mắt
st.markdown("""
    <style>
    /* Nền xanh than sâu chuyên nghiệp */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Làm nổi bật tiêu đề chính */
    h2 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
        margin-bottom: 30px !important;
    }

    /* Tùy chỉnh hộp Expander (Cài đặt) */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        color: #38bdf8 !important; /* Màu xanh dương sáng dễ đọc */
        border-radius: 8px !important;
    }

    /* Card hiển thị số lượng xe */
    .counter-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
    }
    .counter-label {
        color: #94a3b8;
        font-size: 16px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .counter-number {
        color: #2dd4bf; /* Màu xanh Teal cực kỳ êm mắt */
        font-size: 72px;
        font-weight: 800;
        margin: 10px 0;
    }
    
    /* Footer rõ ràng hơn */
    .footer-text {
        text-align: center;
        margin-top: 60px;
        color: #64748b;
        font-size: 15px;
        font-weight: 500;
        padding: 20px;
        border-top: 1px solid #1e293b;
    }
    </style>
    """, unsafe_allow_html=True)

# --- TIÊU ĐỀ ---
st.markdown("<h2>🚗 Hệ thống Giám sát & Đếm xe AI</h2>", unsafe_allow_html=True)

# --- CÀI ĐẶT THU GỌN ---
with st.expander("⚙️ Cấu hình vạch đếm & Độ nhạy"):
    c1, c2 = st.columns(2)
    with c1:
        line_pos = st.slider("Vị trí vạch (%)", 0, 100, 70)
    with c2:
        conf_thresh = st.slider("Độ nhạy (Confidence)", 0.1, 1.0, 0.25)

# --- KHU VỰC TẢI VIDEO ---
uploaded_file = st.file_uploader("📤 Kéo thả video giao thông vào đây", type=["mp4", "mov", "avi"])

if uploaded_file:
    # Card hiển thị kết quả
    st.markdown(f"""
        <div class="counter-container">
            <div class="counter-label">Tổng số phương tiện đã đếm</div>
            <div id="counter-val" class="counter-number">0</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Placeholder để cập nhật số thực tế
    count_placeholder = st.empty()
    
    # Xử lý luồng Video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    video_display = st.empty()
    
    @st.cache_resource
    def load_model(): return YOLO('yolov8n.pt')
    model = load_model()
    
    counter = 0
    tracked_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        h, w, _ = frame.shape
        line_y = int(h * (line_pos / 100))
        
        # YOLO Tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf_thresh, verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().numpy()
            
            for box, id in zip(boxes, ids):
                cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                
                # Vẽ khung mỏng màu Teal
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (191, 212, 45), 1)
                
                if line_y - 8 < cy < line_y + 8:
                    if id not in tracked_ids:
                        tracked_ids.add(id)
                        counter += 1
        
        # Vẽ vạch đếm màu xanh Cyan mờ
        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
        
        # Cập nhật số đếm vào giao diện
        count_placeholder.markdown(f"""
            <div class="counter-container">
                <div class="counter-label">Tổng số phương tiện đã đếm</div>
                <div class="counter-number">{counter}</div>
            </div>
        """, unsafe_allow_html=True)
        
        video_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

    cap.release()

# --- TÊN TÁC GIẢ ---
st.markdown(f"<div class='footer-text'>Phát triển bởi: Trương Công Thành — MSSV: 223332852</div>", unsafe_allow_html=True)
