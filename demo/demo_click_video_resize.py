import os
import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2_camera_predictor
import time

# --------------------------------
# Global Parameters
# --------------------------------
SAM2_CHECKPOINT = "../checkpoints/sam2.1_hiera_tiny.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

INPUT_VIDEO = "../notebooks/videos/merged_webcam_2.mp4"  # webcam_2_merged.mp4
OUTPUT_VIDEO = "../notebooks/videos/masked_webcam_2.mp4"  # webcam_2_masked.mp4

# If True, we skip showing frames in real time and process the entire video as fast as possible
FAST_PROCESSING = True  # False True # 640x480x3 = 0.06s/frame

# -----------------------------
# NEW FEATURE: Channel Deletion
# -----------------------------
# Set DELETE_CHANNEL to one of 0, 1, or 2 corresponding to the R, G, or B channel (after conversion).
# For example, set DELETE_CHANNEL = 0 to remove the red channel.
# Set DELETE_CHANNEL = None to leave all channels intact.
DELETE_CHANNEL = 2  # Change as desired, or set to None

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

predictor = build_sam2_camera_predictor(MODEL_CFG, SAM2_CHECKPOINT)
rebar_points = []
rack_points = []
rebar_labels = []
rack_labels = []


def mouse_callback_rebar(event, x, y, flags, param):
    global rebar_points, rebar_labels
    if event == cv2.EVENT_LBUTTONDOWN:  # Left-click
        rebar_points.append([x, y])
        rebar_labels.append(1)  # Label for rebar
        print("Rebar clicked (Positive):", (x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:  # Right-click
        rebar_points.append([x, y])
        rebar_labels.append(0)  # Label for not rebar
        print("Rebar clicked (Negative):", (x, y))


def mouse_callback_rack(event, x, y, flags, param):
    global rack_points, rack_labels
    if event == cv2.EVENT_LBUTTONDOWN:
        rack_points.append([x, y])
        rack_labels.append(1)  # Label for rack
        print("Rack clicked (Positive):", (x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:  # Right-click
        rack_points.append([x, y])
        rack_labels.append(0)  # Label for not rack
        print("Rack clicked (Negative):", (x, y))


def main():
    global rebar_points, rack_points, rebar_labels, rack_labels

    cap = cv2.VideoCapture(INPUT_VIDEO)
    ret, first_frame = cap.read()
    if not ret:
        print("Error reading video")
        cap.release()
        return

    # Convert first frame from BGR to RGB for processing
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    # --- NEW: Optionally remove one channel from the first frame ---
    if DELETE_CHANNEL is not None:
        # This will set the chosen channel to zero for all pixels.
        first_frame[:, :, DELETE_CHANNEL] = 0

    # --- Step 1: Gather user clicks (unless you want to skip for FAST_PROCESSING) ---
    cv2.namedWindow("Select Rebar (first frame)")
    cv2.setMouseCallback("Select Rebar (first frame)", mouse_callback_rebar)
    while True:
        # For display, convert back to BGR.
        display_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Select Rebar (first frame)", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyWindow("Select Rebar (first frame)")

    cv2.namedWindow("Select Rack (first frame)")
    cv2.setMouseCallback("Select Rack (first frame)", mouse_callback_rack)
    while True:
        display_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Select Rack (first frame)", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyWindow("Select Rack (first frame)")

    if not rebar_points and not rack_points:
        print("No points selected.")
        cap.release()
        return

    rebar_points = np.array(rebar_points, dtype=np.float32)
    rebar_labels = np.array(rebar_labels, dtype=np.int32)
    rack_points = np.array(rack_points, dtype=np.float32)
    rack_labels = np.array(rack_labels, dtype=np.int32)

    print("Rebar points:", rebar_points)
    print("Rebar labels:", rebar_labels)
    print("Rack points:", rack_points)
    print("Rack labels:", rack_labels)

    predictor.load_first_frame(first_frame)

    # Add prompts for rebar (obj_id=1)
    _, rebar_out_ids, rebar_out_logits = predictor.add_new_prompt(
        frame_idx=0,
        obj_id=1,
        points=rebar_points,
        labels=rebar_labels
    )

    # Add prompts for rack (obj_id=2)
    _, rack_out_ids, rack_out_logits = predictor.add_new_prompt(
        frame_idx=0,
        obj_id=2,
        points=rack_points,
        labels=rack_labels
    )

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start_time = time.time()

        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- NEW: Optionally remove the selected channel ---
        if DELETE_CHANNEL is not None:
            frame_rgb[:, :, DELETE_CHANNEL] = 0

        # Track rebar using the modified frame_rgb
        rebar_out_ids, rebar_out_logits = predictor.track(frame_rgb)
        rebar_mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(len(rebar_out_ids)):
            m = (rebar_out_logits[i] > 0).squeeze(0).cpu().numpy().astype(np.uint8) * 255
            rebar_mask = cv2.bitwise_or(rebar_mask, m)
        all_mask = rebar_mask

        # If you want to track rack as well, uncomment the following:
        # rack_out_ids, rack_out_logits = predictor.track(frame_rgb, obj_id=2)
        # rack_mask = np.zeros((h, w), dtype=np.uint8)
        # for i in range(len(rack_out_ids)):
        #     m = (rack_out_logits[i] > 0).squeeze(0).cpu().numpy().astype(np.uint8) * 255
        #     rack_mask = cv2.bitwise_or(rack_mask, m)
        # all_mask = cv2.bitwise_or(rebar_mask, rack_mask)

        # Create the masked frame (using the original frame for visualization)
        masked_frame = cv2.bitwise_and(frame, frame, mask=all_mask)
        out.write(masked_frame)
        end_time = time.time()
        print(f"segment time: {end_time - start_time}")

        # Show frames only if FAST_PROCESSING is False
        if not FAST_PROCESSING:
            overlay = cv2.addWeighted(frame, 1.0, cv2.cvtColor(all_mask, cv2.COLOR_GRAY2BGR), 0.5, 0)
            cv2.imshow("Tracking", overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()

    if not FAST_PROCESSING:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
