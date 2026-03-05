import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile

# --- THIẾT LẬP GIAO DIỆN ---
st.set_page_config(page_title="AI Traffic Classifier Pro", layout="centered")

# CSS tinh chỉnh: Giảm nền, tăng đậm chữ
st.markdown("""
    <style>
    /* Nền màu xám xanh trung tính, không quá đen */
    .stApp {
        background-color: #1e293b; 
        color: #ffffff;
    }
    
    /* Tiêu đề trắng, đậm, sắc nét */
    h2 {
        color: #ffffff !important;
        font-weight: 800 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }

    /* Thẻ thống kê xe: Nằm trên 1 hàng, chữ đậm */
    .stat-container {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 20px;
    }
    .stat-card {
        flex: 1;
        background: #334155;
        border: 2px solid #475569;
        border-radius: 12px;
        padding: 10px;
        text-align: center;
    }
    .stat-label {
        color: #cbd5e1;
        font-size: 13px;
        font-weight: 900; /* Chữ đậm cực đại để dễ đọc */
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .stat-value {
        color: #2dd4bf; /* Màu xanh ngọc sáng */
        font-size: 32px;
        font-weight: 900;
        line-height: 1;
    }
    
    /* Footer đậm nét */
    .footer {
        text-align: center;
        margin-top: 40px;
        color: #f1f5f9;
        font-size: 16px;
        font-weight: 800;
        border-top: 1px solid #475569;
        padding-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>📊 Phân loại & Đếm phương tiện AI</h2>", unsafe_allow_html=True)

# --- CÀI ĐẶT THU GỌN ---
with st.expander("⚙️ Cài đặt hệ thống"):
    c1, c2 = st.columns(2)
    with c1: line_pos = st.slider("Vị trí vạch (%)", 0, 100, 70)
    with c2: conf_thresh = st.slider("Độ nhạy", 0.1, 1.0, 0.25)

uploaded_file = st.file_uploader("📤 Tải lên video để bắt đầu", type=["mp4", "mov", "avi"])

if uploaded_file:
    # Định nghĩa các loại xe
    class_names = {2: 'Ô tô', 3: 'Xe máy', 5: 'Xe bus', 7: 'Xe tải'}
    counts = {name: 0 for name in class_names.values()}
    
    # Khu vực video
    video_output = st.empty()
    
    # Khu vực thống kê nằm ngay dưới video
    stat_placeholder = st.empty()
    
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    @st.cache_resource
    def load_model(): return YOLO('yolov8n.pt')
    model = load_model()
    
    tracked_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        h, w, _ = frame.shape
        line_y = int(h * (line_pos / 100))
        
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf_thresh, verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().numpy()
            clss = results[0].boxes.cls.int().cpu().numpy()
            
            for box, id, cls in zip(boxes, ids, clss):
                if cls in class_names:
                    cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                    
                    if line_y - 8 < cy < line_y + 8:
                        if id not in tracked_ids:
                            tracked_ids.add(id)
                            counts[class_names[cls]] += 1
                    
                    # Vẽ khung nhận diện
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (45, 212, 191), 2)

        # Hiển thị video trước
        video_output.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Hiển thị thống kê ngay dưới video trên cùng 1 hàng
        stat_html = "<div class='stat-container'>"
        for name, val in counts.items():
            stat_html += f"""
                <div class='stat-card'>
                    <div class='stat-label'>{name}</div>
                    <div class='stat-value'>{val}</div>
                </div>"""
        stat_html += "</div>"
        stat_placeholder.markdown(stat_html, unsafe_allow_html=True)

        cv2.line(frame, (0, line_y), (w, line_y), (251, 146, 60), 3)

    cap.release()

st.markdown(f"<div class='footer'>Tác giả: Trương Công Thành — MSSV: 223332852</div>", unsafe_allow_html=True)
