import cv2
import os


def get_video_capture(source_type, source_value):
    """
    Возвращает объект VideoCapture для разных источников.

    Args:
        source_type: "file" | "url" | "camera"
        source_value: путь к файлу, URL или индекс камеры
    """
    if source_type == "file":
        if not os.path.exists(source_value):
            raise FileNotFoundError(f"Файл не найден: {source_value}")
        cap = cv2.VideoCapture(source_value)

    elif source_type == "url":
        cap = cv2.VideoCapture(source_value)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть URL: {source_value}")

    elif source_type == "camera":
        cap = cv2.VideoCapture(int(source_value) if source_value else 0)

    else:
        raise ValueError(f"Неизвестный тип источника: {source_type}")

    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть источник: {source_value}")

    return cap


def process_video_file(input_path, output_path, detector):
    """
    Обрабатывает видеофайл и сохраняет результат.
    """
    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detector.reset()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, stats = detector.process_frame(frame)
        out.write(annotated_frame)

    cap.release()
    out.release()

    return detector.unique_objects
