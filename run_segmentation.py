import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)

PROJECT_DIR = Path(__file__).resolve().parent
FRAMES_DIR = PROJECT_DIR / "frames"
RESULTS_DIR = PROJECT_DIR / "seg_results"

frame_paths = sorted(FRAMES_DIR.glob("frame_*.jpg"))

if not frame_paths:
    raise FileNotFoundError(
        f"Кадры не найдены: {FRAMES_DIR}"
    )

RESULTS_DIR.mkdir(exist_ok=True)

print("Загружаем модель семантической сегментации...")

weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
model = deeplabv3_mobilenet_v3_large(weights=weights)
model.to("cpu")
model.eval()

preprocess = weights.transforms()
class_names = weights.meta["categories"]

palette = np.array([
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
], dtype=np.uint8)

class_mapping = {
    str(index): {
        "name": name,
        "color_rgb": palette[index].tolist(),
    }
    for index, name in enumerate(class_names)
}

with (RESULTS_DIR / "classes.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(class_mapping, file, ensure_ascii=False, indent=2)

for index, frame_path in enumerate(frame_paths, start=1):
    with Image.open(frame_path) as opened_image:
        image = opened_image.convert("RGB")

    width, height = image.size
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.inference_mode():
        scores = model(input_tensor)["out"]

        scores = F.interpolate(
            scores,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        class_map = (
            scores.argmax(dim=1)[0]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

    mask_path = RESULTS_DIR / f"{frame_path.stem}_mask.png"
    Image.fromarray(class_map).save(mask_path)

    original = np.array(image)
    colored_mask = palette[class_map]
    foreground = class_map != 0

    blended = (
        0.55 * original.astype(np.float32)
        + 0.45 * colored_mask.astype(np.float32)
    ).astype(np.uint8)

    overlay = original.copy()
    overlay[foreground] = blended[foreground]

    visual_image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(visual_image)

    class_ids, pixel_counts = np.unique(
        class_map, return_counts=True
    )

    present_classes = []
    legend_y = 8

    for class_id, pixel_count in zip(class_ids, pixel_counts):
        class_id = int(class_id)

        present_classes.append({
            "class_id": class_id,
            "class_name": class_names[class_id],
            "pixel_count": int(pixel_count),
        })

        if class_id == 0:
            continue

        color = tuple(int(value) for value in palette[class_id])
        label = class_names[class_id]

        draw.rectangle(
            (8, legend_y, 180, legend_y + 18),
            fill="black",
        )
        draw.rectangle(
            (12, legend_y + 3, 24, legend_y + 15),
            fill=color,
        )
        draw.text(
            (30, legend_y + 3),
            label,
            fill="white",
        )

        legend_y += 22

    visual_image.save(
        RESULTS_DIR / frame_path.name,
        quality=95,
    )

    frame_data = {
        "frame": frame_path.name,
        "model": "DeepLabV3-MobileNetV3-Large",
        "mask_file": mask_path.name,
        "classes": present_classes,
    }

    with (RESULTS_DIR / f"{frame_path.stem}.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(frame_data, file, ensure_ascii=False, indent=2)

    labels = [
        item["class_name"]
        for item in present_classes
        if item["class_id"] != 0
    ]

    print(
        f"[{index}/{len(frame_paths)}] "
        f"{frame_path.name}: "
        f"{', '.join(labels) if labels else 'только фон'}"
    )

print()
print("Семантическая сегментация завершена.")
print("Обработано кадров:", len(frame_paths))
print("Результаты сохранены в:", RESULTS_DIR)