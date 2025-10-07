# polygon_mask_tool.py
import sys
from pathlib import Path
import cv2
import numpy as np

points = []
img = None
img_vis = None
win = "Polygon Mask Tool"

def draw_preview():
    global img_vis
    img_vis = img.copy()

    # draw clicked points and connecting lines
    if points:
        for p in points:
            cv2.circle(img_vis, p, 3, (0, 255, 255), -1)
        if len(points) > 1:
            cv2.polylines(img_vis, [np.array(points, np.int32)], False, (0, 255, 255), 2)

    # semi-transparent filled preview if 3+ points
    if len(points) >= 3:
        h, w = img.shape[:2]
        preview_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(preview_mask, [np.array(points, np.int32)], 255)
        overlay = img_vis.copy()
        overlay[preview_mask == 255] = (0, 200, 0)  # color area inside polygon
        img_vis = cv2.addWeighted(overlay, 0.3, img_vis, 0.7, 0)

def on_mouse(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        draw_preview()
        cv2.imshow(win, img_vis)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            points.pop()
            draw_preview()
            cv2.imshow(win, img_vis)

def make_and_save_mask(out_path: Path):
    if len(points) < 3:
        raise RuntimeError("Need at least 3 points to make a polygon.")
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, np.int32)], 255)  # inside=True → 255
    if not cv2.imwrite(str(out_path), mask):
        raise RuntimeError(f"Failed to save mask to {out_path}")
    print(f"Saved mask to: {out_path} (inside=True→255, outside=False→0)")

def main():
    global img, img_vis

    in_path = Path(sys.argv[1] if len(sys.argv) > 1 else "image.png")
    out_path = Path(sys.argv[2] if len(sys.argv) > 2 else in_path.with_stem(in_path.stem + "_mask"))

    if not in_path.exists():
        raise FileNotFoundError(f"Image not found: {in_path}")

    img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to load image: {in_path}")

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    draw_preview()
    cv2.imshow(win, img_vis)

    print(
        "\nControls:\n"
        "  Left-click: add point\n"
        "  Right-click or 'z': undo last point\n"
        "  'r': reset (clear all points)\n"
        "  's': save mask PNG\n"
        "  'q' or ESC: quit\n"
    )

    while True:
        cv2.imshow(win, img_vis)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord('q')):  # ESC or q
            break
        elif key == ord('z'):      # undo
            if points:
                points.pop()
                draw_preview()
        elif key == ord('r'):      # reset
            points.clear()
            draw_preview()
        elif key == ord('s'):      # save
            try:
                make_and_save_mask(out_path.with_suffix(".png"))
            except Exception as e:
                print(f"Error: {e}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
