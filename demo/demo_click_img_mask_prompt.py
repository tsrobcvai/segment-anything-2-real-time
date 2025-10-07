import os
import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2_camera_predictor
import time
import argparse
# --------------------------------
# Global Parameters
# --------------------------------
SAM2_CHECKPOINT = "../checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
INPUT_IMG = "../notebooks/rebar_insertion/test.png"  
OUTPUT_IMG = "../notebooks/rebar_insertion/test_result.png"
REF_IMG = "../notebooks/rebar_insertion/ref.jpg"
REF_MASK_list = ["../notebooks/rebar_insertion/ref_mask_1.png",
                "../notebooks/rebar_insertion/ref_mask_2.png"
                ]

color_palette = [
        (255, 0, 0),     # red
        (0, 255, 0),     # green
        (0, 0, 255),     # blue
        (255, 255, 0),   # yellow
        (255, 0, 255),   # magenta
        (0, 255, 255),   # cyan
        (255, 128, 0),   # orange
        (128, 0, 255),   # purple
        (0, 128, 255),   # sky blue
        (128, 255, 0),   # lime
    ]
# # NEW: Resizing factor (e.g., 0.5 to half the resolution, 1.0 for original)
# RESIZE_FACTOR = 1
rebar_points = []
rebar_labels = []

def mouse_callback_rebar(event, x, y, flags, param):
    global rebar_points, rebar_labels
    if event == cv2.EVENT_LBUTTONDOWN:  # Left-click (Positive)
        rebar_points.append([x, y])
        rebar_labels.append(1)  # Label for rebar
        print("Rebar clicked (Positive):", (x, y))
    elif event == cv2.EVENT_MBUTTONDOWN:  # Mid-click (Negative)
        rebar_points.append([x, y])
        rebar_labels.append(0)  # Label for not rebar
        print("Rebar clicked (Negative):", (x, y))

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

predictor = build_sam2_camera_predictor(MODEL_CFG, SAM2_CHECKPOINT)

def main(use_points=False):
    ref_img = cv2.imread(REF_IMG, cv2.IMREAD_UNCHANGED)
    # first_frame is read in BGR; convert to RGB for processing
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    # Get original dimensions
    orig_h, orig_w = ref_img.shape[:2]
    resized_w, resized_h = orig_w, orig_h
    ref_img_resized = ref_img
    # Load the first (resized) frame into the predictor.
    predictor.load_first_frame(ref_img_resized)

    if use_points:
        # points = np.array([[660, 267], [250, 220]], dtype=np.float32) # np.array([[210, 350], [250, 220]], dtype=np.float32)
        # labels = np.array([1,1], dtype=np.int32)
        cv2.namedWindow("Select Rebar (first frame)")
        cv2.setMouseCallback("Select Rebar (first frame)", mouse_callback_rebar)
        while True:
            display_frame = cv2.cvtColor(ref_img_resized, cv2.COLOR_RGB2BGR)
            cv2.imshow("Select Rebar (first frame)", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyWindow("Select Rebar (first frame)")
        points = np.array(rebar_points, dtype=np.float32)
        labels = np.array(rebar_labels, dtype=np.int32)

    # object_ids_list = []
    # video_res_masks_list = []
    for i, mask_path in enumerate(REF_MASK_list):
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        _, object_ids, video_res_masks = predictor.add_new_mask(
            frame_idx=0,
            obj_id=i,
            mask=mask
        )
        print(f"Added mask {i} with object_ids: {object_ids}")
        if use_points:
            _, object_ids, video_res_masks = predictor.add_new_points(
            frame_idx=0,
            obj_id=i,
            points=points,
            labels=labels)
        # object_ids_list.append(object_ids)
        # video_res_masks_list.append(video_res_masks)

    input_img = cv2.imread(INPUT_IMG, cv2.IMREAD_UNCHANGED)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    # Track rebar on the resized frame.
    object_ids_predict, video_res_masks_predict = predictor.track(input_img)
    # Create a colored mask per object and compose into the output frame.
    masked_frame = np.zeros_like(input_img)
    for i in range(len(object_ids_predict)):
        # Each output mask has shape [1, H, W]
        m = (video_res_masks_predict[i] > 0).squeeze(0).cpu().numpy().astype(bool)
        color = color_palette[int(object_ids_predict[i]) % len(color_palette)]
        masked_frame[m] = color
    cv2.imwrite(OUTPUT_IMG, masked_frame)

if __name__ == "__main__":
    # add argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_points", type=bool, default=False)
    args = parser.parse_args()
    main(args.use_points)