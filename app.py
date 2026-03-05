import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile

# --- THIẾT LẬP GIAO DIỆN ---
st.set_page_config(page_title="AI Traffic Pro", layout="wide") # Dùng Wide để đủ chỗ cho 2 video

st.markdown("""
    <style>
    /* Nền xám xanh trung tính */
    .stApp { background-color: #1e293b; color: #ffffff; }
    
    /* Chữ tiêu đề và nhãn phải cực kỳ rõ nét */
    h2, h3, .stMarkdown p, label {
        color: #ffffff !important;
        font-weight: 900 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
    }
    
    /* Làm đậm các chữ trong thanh Slider */
    .stSlider label { font-size: 18px !important; }

    /* Thẻ thống kê xe nằm ngang */
    .stat-container {
        display: flex;
        justify-content: space-around;
        gap: 15px;
        margin-bottom: 30px;
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
    }
    .stat-card {
        flex: 1;
        background: #334155;
        border: 2px solid #64748b;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
    }
    .stat-label { color: #f1f5f9; font-size: 16px; font-weight: 900; margin-bottom: 5px; }
    .stat-value { color: #2dd4bf; font-size: 40px; font-weight: 900; }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #ffffff;
        font-size: 20px;
        font-weight: 900;
        border-top: 2px solid #ffffff;
        padding-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>📊 Hệ thống Phân loại & Đếm phương tiện AI</h2>", unsafe_allow_html=True)

# --- CÀI ĐẶT THU GỌN ---
with st.expander("⚙️ Cài đặt hệ thống (Vị trí vạch & Độ nhạy)"):
    c1, c2 = st.columns(2)
    with c1: line_pos = st.slider("Vị trí vạch đếm (%)", 0, 100, 70)
    with c2: conf_thresh = st.slider("Độ nhạy nhận diện", 0.1, 1.0, 0.25)

# --- TẢI VIDEO ---
uploaded_file = st.file_uploader("📤 Bước 1: Chọn video giao thông để bắt đầu", type=["mp4", "mov", "avi"])

if uploaded_file:
    # 1. Khu vực thống kê hàng ngang
    class_names = {2: 'Ô tô', 3: 'Xe máy', 5: 'Xe bus', 7: 'Xe tải'}
    counts = {name: 0 for name in class_names.values()}
    stat_placeholder = st.empty()
    
    # 2. Bước quan trọng: Tạo 2 cột cho video gốc và video xử lý
    col_input, col_output = st.columns(2)
    with col_input:
        st.markdown("### 📹 Video gốc")
        input_video_placeholder = st.empty()
    with col_output:
        st.markdown("### 🖼 Kết quả xử lý AI")
        output_video_placeholder = st.empty()

    # Xử lý video
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
        
        # Lưu frame gốc để hiển thị bên trái
        orig_frame = frame.copy()
        
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
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (45, 212, 191), 2)

        # Cập nhật video gốc và video xử lý song song
        input_video_placeholder.image(cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        cv2.line(frame, (0, line_y), (w, line_y), (251, 146, 60), 4)
        output_video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Cập nhật thống kê ngay bên dưới video
        stat_html = "<div class='stat-container'>"
        for name, val in counts.items():
            stat_html += f"<div class='stat-card'><div class='stat-label'>{name}</div><div class='stat-value'>{val}</div></div>"
        stat_html += "</div>"
        stat_placeholder.markdown(stat_html, unsafe_allow_html=True)

    cap.release()

st.markdown(f"<div class='footer'>Tác giả: Trương Công Thành — MSV: 223332852</div>", unsafe_allow_html=True)
