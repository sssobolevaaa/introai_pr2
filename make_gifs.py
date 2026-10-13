from pathlib import Path

from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
GIFS_DIR = PROJECT_DIR / "gifs"

FPS = 2
FRAME_DURATION_MS = round(1000 / FPS)
MAX_SIZE = (800, 800)

GIFS_DIR.mkdir(exist_ok=True)


def make_gif(folder_name, output_name):
    source_dir = PROJECT_DIR / folder_name
    frame_paths = sorted(source_dir.glob("frame_*.jpg"))

    if not frame_paths:
        raise FileNotFoundError(
            f"Не найдены JPG-кадры в папке: {source_dir}"
        )

    images = []

    try:
        for frame_path in frame_paths:
            with Image.open(frame_path) as source:
                image = source.convert("RGB")
                image.thumbnail(
                    MAX_SIZE,
                    Image.Resampling.LANCZOS,
                )
                images.append(image)

        output_path = GIFS_DIR / output_name

        images[0].save(
            output_path,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=FRAME_DURATION_MS,
            loop=0,
            disposal=2,
        )

        size_mb = output_path.stat().st_size / (1024 * 1024)

        print("Создана GIF:", output_path.name)
        print("Исходных кадров:", len(images))
        print("Длительность кадра, мс:", FRAME_DURATION_MS)
        print("Размер файла, МБ:", round(size_mb, 2))
        print()

    finally:
        for image in images:
            image.close()


make_gif("det_results", "detection.gif")
make_gif("seg_results", "segmentation.gif")

print("Обе GIF созданы.")
print("Папка:", GIFS_DIR)