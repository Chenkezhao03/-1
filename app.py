import streamlit as st
import tempfile
import os
import time
import cv2
import imageio
from detector import VideoDetector
from video_utils import get_video_capture

st.set_page_config(page_title="YOLO Video Tracker", layout="wide")

TRANSLATIONS = {
    "ru": {
        "title": "🎥 Детекция и трекинг объектов на видео",
        "subtitle": "**YOLO + ByteTrack** — обнаружение и отслеживание объектов",
        "sidebar_header": "Источник видео", "source_label": "Выберите тип:",
        "source_file": "Загрузить файл", "source_url": "URL видео", "source_camera": "Веб-камера",
        "upload_label": "Загрузите видеофайл", "url_input": "Введите URL видео (прямая ссылка на mp4):",
        "url_examples": "Примеры рабочих URL:", "url_empty_warning": "Пожалуйста, введите URL видео",
        "btn_process": "▶ Начать обработку", "btn_stream": "▶ Запустить поток", "btn_camera": "▶ Включить камеру",
        "camera_index": "Индекс камеры:", "processing": "Обработка видео...", "done": "✅ Обработка завершена!",
        "processed_video": "🎬 Обработанное видео", "unique_objects": "📊 Уникальные объекты",
        "source_info": "Источник:", "camera_info": "Веб-камера доступна только в локальном запуске. Браузер должен быть на той же машине, где запущен Streamlit.",
        "error": "Ошибка:", "url_error": "Не удалось открыть URL. Убедитесь, что это прямая ссылка на MP4-файл.",
        "camera_error": "Не удалось открыть камеру. Проверьте, подключена ли она и не занята ли другим приложением.",
        "how_to_run": "**Как запустить:**", "how_to_run_1": "1. Установите зависимости",
        "how_to_run_2": "2. Выполните: `streamlit run app.py`", "how_to_run_3": "3. Откройте браузер: http://localhost:8501",
        "frame_progress": "Кадр", "time_elapsed": "Прошло времени", "time_remaining": "Осталось примерно",
        "processing_complete": "Обработка завершена за", "video_info": "Информация о видео",
        "resolution": "Разрешение", "fps": "Кадров в секунду", "total_frames": "Всего кадров",
        "processing_speed": "Скорость обработки", "no_file": "Пожалуйста, загрузите видеофайл",
        "no_objects": "Объекты не обнаружены", "stream_complete": "Поток обработан",
        "frames_processed": "кадров обработано", "captured_frames": "Захвачено кадров",
    },
    "en": {
        "title": "🎥 Object Detection and Tracking on Video",
        "subtitle": "**YOLO + ByteTrack** — object detection and tracking",
        "sidebar_header": "Video Source", "source_label": "Select type:",
        "source_file": "Upload File", "source_url": "Video URL", "source_camera": "Webcam",
        "upload_label": "Upload a video file", "url_input": "Enter video URL (direct mp4 link):",
        "url_examples": "Example working URLs:", "url_empty_warning": "Please enter a video URL",
        "btn_process": "▶ Start Processing", "btn_stream": "▶ Start Stream", "btn_camera": "▶ Turn On Camera",
        "camera_index": "Camera index:", "processing": "Processing video...", "done": "✅ Processing complete!",
        "processed_video": "🎬 Processed Video", "unique_objects": "📊 Unique Objects",
        "source_info": "Source:", "camera_info": "Webcam is available only in local run. The browser must be on the same machine where Streamlit is running.",
        "error": "Error:", "url_error": "Cannot open URL. Make sure it is a direct link to an MP4 file.",
        "camera_error": "Cannot open webcam. Check if it is connected and not used by another app.",
        "how_to_run": "**How to run:**", "how_to_run_1": "1. Install dependencies",
        "how_to_run_2": "2. Run: `streamlit run app.py`", "how_to_run_3": "3. Open browser: http://localhost:8501",
        "frame_progress": "Frame", "time_elapsed": "Elapsed time", "time_remaining": "Estimated remaining",
        "processing_complete": "Processing completed in", "video_info": "Video Information",
        "resolution": "Resolution", "fps": "FPS", "total_frames": "Total frames",
        "processing_speed": "Processing speed", "no_file": "Please upload a video file",
        "no_objects": "No objects detected", "stream_complete": "Stream processed",
        "frames_processed": "frames processed", "captured_frames": "Frames captured",
    },
    "zh": {
        "title": "🎥 视频目标检测与追踪",
        "subtitle": "**YOLO + ByteTrack** — 目标检测与追踪",
        "sidebar_header": "视频来源", "source_label": "选择类型：",
        "source_file": "上传文件", "source_url": "视频 URL", "source_camera": "摄像头",
        "upload_label": "上传视频文件", "url_input": "输入视频 URL（直接 mp4 链接）：",
        "url_examples": "示例可用 URL：", "url_empty_warning": "请输入视频 URL",
        "btn_process": "▶ 开始处理", "btn_stream": "▶ 开始流式传输", "btn_camera": "▶ 打开摄像头",
        "camera_index": "摄像头索引：", "processing": "正在处理视频...", "done": "✅ 处理完成！",
        "processed_video": "🎬 处理后的视频", "unique_objects": "📊 独特对象",
        "source_info": "来源：", "camera_info": "摄像头仅在本地运行时可用。浏览器必须与运行 Streamlit 的机器在同一台计算机上。",
        "error": "错误：", "url_error": "无法打开 URL。请确保它是 MP4 文件的直接链接。",
        "camera_error": "无法打开摄像头。请检查是否已连接且未被其他应用程序占用。",
        "how_to_run": "**如何运行：**", "how_to_run_1": "1. 安装依赖",
        "how_to_run_2": "2. 运行：`streamlit run app.py`", "how_to_run_3": "3. 打开浏览器：http://localhost:8501",
        "frame_progress": "帧", "time_elapsed": "已用时间", "time_remaining": "预计剩余",
        "processing_complete": "处理完成，用时", "video_info": "视频信息",
        "resolution": "分辨率", "fps": "帧率", "total_frames": "总帧数",
        "processing_speed": "处理速度", "no_file": "请上传视频文件",
        "no_objects": "未检测到对象", "stream_complete": "流处理完成",
        "frames_processed": "帧已处理", "captured_frames": "已捕获帧数",
    },
}

if "lang" not in st.session_state:
    st.session_state.lang = "zh"

lang_options = {"en": "English", "ru": "Русский", "zh": "中文"}
selected_lang = st.sidebar.selectbox(
    "🌐 Language / Язык / 语言",
    options=list(lang_options.keys()),
    format_func=lambda x: lang_options[x],
    index=list(lang_options.keys()).index(st.session_state.lang)
)
st.session_state.lang = selected_lang
t = TRANSLATIONS[st.session_state.lang]

st.title(t["title"])
st.markdown(t["subtitle"])

@st.cache_resource
def load_detector():
    return VideoDetector(model_name="yolo11n.pt", tracker="bytetrack.yaml")

detector = load_detector()

def process_video_with_progress(input_path, output_path, detector, t):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.subheader(t["video_info"])
    col1, col2, col3 = st.columns(3)
    col1.metric(t["resolution"], f"{width}×{height}")
    col2.metric(t["fps"], f"{fps}")
    col3.metric(t["total_frames"], f"{total_frames}")

    st.markdown("---")
    st.write("🔴 **实时处理预览（请观察此处框框）**")
    st_frame = st.empty() # 实时显示处理画面的占位符

    # 使用 imageio 生成浏览器支持的 H.264 MP4
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)

    st.markdown("---")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    detector.reset()
    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated_frame, _ = detector.process_frame(frame)
        
        # 实时显示到网页上
        st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        
        # 写入视频
        writer.append_data(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

        frame_idx += 1
        progress = frame_idx / total_frames if total_frames > 0 else 0
        progress_bar.progress(min(progress, 1.0))
        
        elapsed = time.time() - start_time
        status_text.text(f"{t['frame_progress']}: {frame_idx}/{total_frames} ({progress*100:.1f}%) | {t['time_elapsed']}: {elapsed:.1f}s")

    cap.release()
    writer.close()
    
    progress_bar.progress(1.0)
    status_text.text(f"✅ {t['processing_complete']} {time.time() - start_time:.1f}s")
    
    return detector.unique_objects

# ============================================
# 侧边栏逻辑
# ============================================
st.sidebar.markdown("---")
st.sidebar.header(t["sidebar_header"])
source_type = st.sidebar.radio(t["source_label"], [t["source_file"], t["source_url"], t["source_camera"]])

if source_type == t["source_file"]:
    st.header("📁 " + t["source_file"])
    uploaded_file = st.file_uploader(t["upload_label"], type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        st.video(tfile.name)
        if st.button(t["btn_process"]):
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            stats = process_video_with_progress(tfile.name, output_path, detector, t)
            st.success(t["done"])
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                st.subheader(t["processed_video"])
                with open(output_path, 'rb') as video_file:
                    st.video(video_file.read())
            st.subheader(t["unique_objects"])
            if stats:
                for cls_name, ids in stats.items():
                    st.write(f"- **{cls_name}**: {len(ids)}")
            else:
                st.info(t["no_objects"])
    else:
        st.info(t["no_file"])

elif source_type == t["source_url"]:
    st.header("🌐 " + t["source_url"])
    url = st.text_input(t["url_input"], placeholder="https://samplelib.com/lib/preview/mp4/sample-5s.mp4")
    if url:
        if st.button(t["btn_stream"]):
            try:
                cap = get_video_capture("url", url)
                stframe = st.empty()
                detector.reset()
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret or frame_count >= 500:
                        break
                    annotated_frame, _ = detector.process_frame(frame)
                    stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                    frame_count += 1
                cap.release()
                st.success(f"✅ {t['stream_complete']}: {frame_count} {t['frames_processed']}")
                if detector.unique_objects:
                    for cls_name, ids in detector.unique_objects.items():
                        st.write(f"- **{cls_name}**: {len(ids)}")
                else:
                    st.info(t["no_objects"])
            except Exception as e:
                st.error(f"{t['error']} {e}")

elif source_type == t["source_camera"]:
    st.header("📷 " + t["source_camera"])
    st.info(t["camera_info"])
    camera_index = st.number_input(t["camera_index"], min_value=0, max_value=5, value=0, step=1)
    if st.button(t["btn_camera"]):
        cap = get_video_capture("camera", str(camera_index))
        if not cap.isOpened():
            st.error(t["camera_error"])
        else:
            stframe = st.empty()
            detector.reset()
            frame_count = 0
            while frame_count < 300:
                ret, frame = cap.read()
                if not ret: break
                annotated_frame, _ = detector.process_frame(frame)
                stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                frame_count += 1
            cap.release()
            st.success(f"✅ {t['captured_frames']}: {frame_count}")
            if detector.unique_objects:
                for cls_name, ids in detector.unique_objects.items():
                    st.write(f"- **{cls_name}**: {len(ids)}")
            else:
                st.info(t["no_objects"])

st.sidebar.markdown("---")
st.sidebar.info(f"{t['how_to_run']}\n{t['how_to_run_1']}\n{t['how_to_run_2']}\n{t['how_to_run_3']}")