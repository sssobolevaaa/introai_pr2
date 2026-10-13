import shutil
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
VIDEO_PATH = PROJECT_DIR / "video_input.mp4"
FRAMES_DIR = PROJECT_DIR / "frames"
FPS = 2

ffmpeg_path = shutil.which("ffmpeg")

if ffmpeg_path is None:
    raise FileNotFoundError(
        "ffmpeg не найден. Проверь PATH и перезапусти PyCharm."
    )

if not VIDEO_PATH.is_file():
    raise FileNotFoundError(
        f"Видео не найдено: {VIDEO_PATH}"
    )

FRAMES_DIR.mkdir(exist_ok=True)

if any(FRAMES_DIR.glob("frame_*.jpg")):
    raise RuntimeError(
        "В папке frames уже есть кадры. "
        "Перед повторным извлечением перенеси или удали старые frame_*.jpg."
    )

command = [
    ffmpeg_path,
    "-hide_banner",
    "-nostdin",
    "-i", str(VIDEO_PATH),
    "-vf", f"fps={FPS}",
    "-q:v", "2",
    str(FRAMES_DIR / "frame_%04d.jpg"),
]

print("Видео:", VIDEO_PATH)
print("Извлекаем кадров в секунду:", FPS)

subprocess.run(command, check=True)

frames = sorted(FRAMES_DIR.glob("frame_*.jpg"))

if not frames:
    raise RuntimeError("ffmpeg завершился, но кадры не найдены.")

print("Готово! Количество кадров:", len(frames))
print("Папка с кадрами:", FRAMES_DIR)
print("Первый кадр:", frames[0].name)
print("Последний кадр:", frames[-1].name)