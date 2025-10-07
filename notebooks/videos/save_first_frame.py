# save_first_frame.py
import sys
import cv2
from pathlib import Path

def save_first_frame(video_path: str, out_path: str | None = None) -> None:
    vp = Path(video_path)
    if not vp.exists():
        raise FileNotFoundError(f"Video not found: {vp}")

    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {vp}")

    ok, frame = cap.read()  # reads frame index 0
    cap.release()

    if not ok or frame is None:
        raise RuntimeError("Could not read the first frame (video may be empty or corrupted).")

    if out_path is None:
        out_path = vp.with_suffix(".png").name  # e.g., merged_webcam_2.png

    # Write image (cv2 handles BGR->file encoding internally)
    if not cv2.imwrite(out_path, frame):
        raise RuntimeError(f"Failed to write image to: {out_path}")

    print(f"Saved first frame to: {out_path}")

if __name__ == "__main__":
    # Usage:
    #   python save_first_frame.py                # uses default 'merged_webcam_2.mp4' -> 'merged_webcam_2.png'
    #   python save_first_frame.py input.mp4      # saves to input.png
    #   python save_first_frame.py input.mp4 out.jpg
    args = sys.argv[1:]
    video = args[0] if args else "merged_webcam_2.mp4"
    out = args[1] if len(args) > 1 else None
    save_first_frame(video, out)

