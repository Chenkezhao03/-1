import cv2
from ultralytics import YOLO


class VideoDetector:
    def __init__(self, model_name="yolo11n.pt", tracker="bytetrack.yaml"):
        """Инициализация модели YOLO и трекера."""
        self.model = YOLO(model_name)
        self.tracker = tracker
        self.unique_objects = {}

    def process_frame(self, frame):
        """
        Обработка одного кадра: детекция + трекинг.
        Возвращает кадр с нарисованными рамками и статистику.
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            verbose=False
        )

        # Подсчёт уникальных объектов
        if results[0].boxes.id is not None:
            boxes = results[0].boxes
            for i, box_id in enumerate(boxes.id):
                cls_id = int(boxes.cls[i])
                cls_name = results[0].names[cls_id]
                obj_id = int(box_id)

                if cls_name not in self.unique_objects:
                    self.unique_objects[cls_name] = set()
                self.unique_objects[cls_name].add(obj_id)

        # Получаем кадр с нарисованными рамками
        annotated_frame = results[0].plot()

        # Добавляем статистику на кадр
        stats_text = self._get_stats_text()
        cv2.putText(
            annotated_frame, stats_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        return annotated_frame, self.unique_objects

    def _get_stats_text(self):
        """Формирует текст статистики для отображения."""
        lines = ["Unique objects:"]
        for cls_name, ids in self.unique_objects.items():
            lines.append(f"  {cls_name}: {len(ids)}")
        return " | ".join(lines)

    def reset(self):
        """Сброс счётчиков."""
        self.unique_objects = {}
