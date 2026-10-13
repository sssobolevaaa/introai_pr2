from ultralytics import YOLO

print("Загрузка модели для детекции...")
detection_model = YOLO("yolov8n.pt")
print("Модель детекции загружена успешно.")

print("Загрузка модели для сегментации...")
segmentation_model = YOLO("yolov8n-seg.pt")
print("Модель сегментации загружена успешно.")

print("Проверка моделей завершена.")