import json
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent
FRAMES_DIR = PROJECT_DIR / "frames"
RESULTS_DIR = PROJECT_DIR / "det_results"
MODEL_PATH = PROJECT_DIR / "yolov8n.pt"

CONFIDENCE_THRESHOLD = 0.25

frame_paths = sorted(FRAMES_DIR.glob("frame_*.jpg"))

if not frame_paths:
    raise FileNotFoundError(
        f"Кадры не найдены в папке: {FRAMES_DIR}"
    )

RESULTS_DIR.mkdir(exist_ok=True)

print("Загружаем модель детекции...")
model = YOLO(str(MODEL_PATH))

total_detections = 0

for index, frame_path in enumerate(frame_paths, start=1):
    result = model.predict(
        source=str(frame_path),
        conf=CONFIDENCE_THRESHOLD,
        device="cpu",
        verbose=False,
    )[0]

    annotated_image = result.plot()
    image_path = RESULTS_DIR / frame_path.name

    saved = cv2.imwrite(str(image_path), annotated_image)

    if not saved:
        raise RuntimeError(
            f"Не удалось сохранить изображение: {image_path}"
        )

    objects = []

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        coordinates = box.xyxy[0].tolist()

        objects.append({
            "class_id": class_id,
            "class_name": result.names[class_id],
            "confidence": confidence,
            "bbox_xyxy": coordinates,
        })

    frame_data = {
        "frame": frame_path.name,
        "model": "yolov8n.pt",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "objects": objects,
    }

    json_path = RESULTS_DIR / f"{frame_path.stem}.json"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(frame_data, file, ensure_ascii=False, indent=2)

    total_detections += len(objects)

    print(
        f"[{index}/{len(frame_paths)}] "
        f"{frame_path.name}: найдено объектов — {len(objects)}"
    )

print()
print("Детекция завершена.")
print("Обработано кадров:", len(frame_paths))
print("Всего обнаружений по всем кадрам:", total_detections)
print("Результаты сохранены в:", RESULTS_DIR)