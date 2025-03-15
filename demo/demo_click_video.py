import os
import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2_camera_predictor
import time

# --------------------------------
# Global Parameters
# --------------------------------
# SAM2_CHECKPOINT = "../checkpoints/sam2.1_hiera_tiny.pt"
# MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

# SAM2_CHECKPOINT = "../checkpoints/sam2.1_hiera_base_plus.pt"
# MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

SAM2_CHECKPOINT = "../checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


INPUT_VIDEO = "../notebooks/videos_merged/merged_webcam_1.mp4" # webcam_2_merged.mp4
OUTPUT_VIDEO = "../notebooks/videos_merged/merged_webcam_1_masked.mp4" # webcam_2_masked.mp4
Prompt_frame_id = 1 # 50
# Set to True to process frames as fast as possible (and not display them)
FAST_PROCESSING = False  # False True
Rack_include = True # Gripper also




# # NEW: Resizing factor (e.g., 0.5 to half the resolution, 1.0 for original)
# RESIZE_FACTOR = 1

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
    if event == cv2.EVENT_LBUTTONDOWN:  # Left-click (Positive)
        rebar_points.append([x, y])
        rebar_labels.append(1)  # Label for rebar
        print("Rebar clicked (Positive):", (x, y))
    elif event == cv2.EVENT_MBUTTONDOWN:  # Right-click (Negative)
        rebar_points.append([x, y])
        rebar_labels.append(0)  # Label for not rebar
        print("Rebar clicked (Negative):", (x, y))

def mouse_callback_rack(event, x, y, flags, param):
    global rack_points, rack_labels
    if event == cv2.EVENT_LBUTTONDOWN:
        rack_points.append([x, y])
        rack_labels.append(1)  # Label for rack
        print("Rack clicked (Positive):", (x, y))
    elif event == cv2.EVENT_MBUTTONDOWN:
        rack_points.append([x, y])
        rack_labels.append(0)  # Label for not rack
        print("Rack clicked (Negative):", (x, y))


def main():
    global rebar_points, rack_points, rebar_labels, rack_labels

    cap = cv2.VideoCapture(INPUT_VIDEO)

    # Get total number of frames in the input video
    total_frames_input = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in input video: {total_frames_input}")

    for _ in range(Prompt_frame_id):
        ret, first_frame = cap.read()
    if not ret:
        print("Error reading video")
        cap.release()
        return

    # first_frame is read in BGR; convert to RGB for processing
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    # Get original dimensions
    orig_h, orig_w = first_frame.shape[:2]

    # # NEW: Compute resized dimensions and resize the first frame for processing and display
    # if RESIZE_FACTOR != 1.0:
    #     resized_w = int(orig_w * RESIZE_FACTOR)
    #     resized_h = int(orig_h * RESIZE_FACTOR)
    #     first_frame_resized = cv2.resize(first_frame, (resized_w, resized_h))
    # else:

    resized_w, resized_h = orig_w, orig_h
    first_frame_resized = first_frame

    # --- Step 1: Gather user clicks on the resized first frame ---
    cv2.namedWindow("Select Rebar (first frame)")
    cv2.setMouseCallback("Select Rebar (first frame)", mouse_callback_rebar)
    while True:
        # Convert back to BGR for displaying with OpenCV
        display_frame = cv2.cvtColor(first_frame_resized, cv2.COLOR_RGB2BGR)
        cv2.imshow("Select Rebar (first frame)", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyWindow("Select Rebar (first frame)")

    if Rack_include:
        cv2.namedWindow("Select Rack (first frame)")
        cv2.setMouseCallback("Select Rack (first frame)", mouse_callback_rack)
        while True:
            display_frame = cv2.cvtColor(first_frame_resized, cv2.COLOR_RGB2BGR)
            cv2.imshow("Select Rack (first frame)", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyWindow("Select Rack (first frame)")

    if not rebar_points and not rack_points:
        print("No points selected.")
        cap.release()
        return

    # Convert click points and labels to NumPy arrays.
    rebar_points = np.array(rebar_points, dtype=np.float32)
    rebar_labels = np.array(rebar_labels, dtype=np.int32)
    rack_points = np.array(rack_points, dtype=np.float32)
    rack_labels = np.array(rack_labels, dtype=np.int32)

    print("Rebar points:", rebar_points)
    print("Rebar labels:", rebar_labels)
    print("Rack points:", rack_points)
    print("Rack labels:", rack_labels)

    # Load the first (resized) frame into the predictor.
    predictor.load_first_frame(first_frame_resized)

    # Add prompts for rebar (obj_id=1)
    _, rebar_out_ids, rebar_out_logits = predictor.add_new_prompt(
        frame_idx=0,
        obj_id=1,
        points=rebar_points,
        labels=rebar_labels
    )

    if Rack_include:
        # (Optionally, add prompts for rack as well (obj_id=2))
        _, rack_out_ids, rack_out_logits = predictor.add_new_prompt(
            frame_idx=0,
            obj_id=2,
            points=rack_points,
            labels=rack_labels
        )

    # Use the original video properties for the output video.
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Note: Output video will be at the original resolution.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (orig_w, orig_h))

    # Now reset to the first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start_time = time.time()

        # NEW: Resize the current frame before processing.
        frame_resized = cv2.resize(frame, (resized_w, resized_h))
        # Convert resized frame from BGR to RGB.
        frame_resized_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Check if frame_resized_rgb is completely black
        if np.all(frame_resized_rgb == 0):
            print("Warning: Frame is completely black. Skipping this frame.")
            out.write(frame_resized_rgb)
            continue

        # Track rebar on the resized frame.
        rebar_out_ids, rebar_out_logits = predictor.track(frame_resized_rgb)
        # Create a mask for the resized frame.
        rebar_mask = np.zeros((resized_h, resized_w), dtype=np.uint8)
        for i in range(len(rebar_out_ids)):
            # (Assuming each output mask has shape [1, H, W].)
            m = (rebar_out_logits[i] > 0).squeeze(0).cpu().numpy().astype(np.uint8) * 255
            rebar_mask = cv2.bitwise_or(rebar_mask, m)

        # (If tracking rack as well, you can uncomment and add similarly.)
        # rack_out_ids, rack_out_logits = predictor.track(frame_resized_rgb, obj_id=2)
        # rack_mask = np.zeros((resized_h, resized_w), dtype=np.uint8)
        # for i in range(len(rack_out_ids)):
        #     m = (rack_out_logits[i] > 0).squeeze(0).cpu().numpy().astype(np.uint8) * 255
        #     rack_mask = cv2.bitwise_or(rack_mask, m)
        # all_mask_resized = cv2.bitwise_or(rebar_mask, rack_mask)

        all_mask_resized = rebar_mask  # Only rebar is being tracked here.
        # NEW: Upscale the mask back to the original frame size.
        all_mask = cv2.resize(all_mask_resized, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        # Apply the upscaled mask to the original frame.
        masked_frame = cv2.bitwise_and(frame, frame, mask=all_mask)
        out.write(masked_frame)
        end_time = time.time()
        print(f"Segment time: {end_time - start_time:.3f} seconds")
        # Show frames only if FAST_PROCESSING is False.
        if not FAST_PROCESSING:
            overlay = cv2.addWeighted(frame, 1.0, cv2.cvtColor(all_mask, cv2.COLOR_GRAY2BGR), 0.5, 0)
            cv2.imshow("Tracking", overlay)
            # 'q' to stop early when displaying
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    if not FAST_PROCESSING:
        cv2.destroyAllWindows()

        # Check total frames in the output video
    cap_out = cv2.VideoCapture(OUTPUT_VIDEO)
    total_frames_output = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_out.release()
    print(f"Total frames counted in output video: {total_frames_output}")
    if total_frames_input != total_frames_output:
        print(f"WARNING: Frame count mismatch! Input: {total_frames_input}, Output: {total_frames_output}")
    else:
        print(f"WARNING: Frame count match! Input: {total_frames_input}, Output: {total_frames_output}")


if __name__ == "__main__":
    main()