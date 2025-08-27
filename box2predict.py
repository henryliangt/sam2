import os
import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ----------------------------
# Paths
# ----------------------------
cropped_dir = r"C:\Users\henry\PycharmProjects\rnsh\image\caroline4back250X16\dino_base368_0.9424\assembled_back\boundingBox\croppedLesionsRefined2"
output_dir = os.path.join(cropped_dir, "sam2")
os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# Load SAM2 model
# ----------------------------
sam_checkpoint = "checkpoints/sam2_hiera_large.pt"
model_cfg = os.path.join(r"C:\Users\henry\PycharmProjects\rnsh\sam2\sam2_configs", "sam2_hiera_l.yaml")

device = "cuda" if torch.cuda.is_available() else "cpu"
sam2_model = build_sam2(model_cfg, sam_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

print("✅ SAM2 model loaded for inference on", device)

# ----------------------------
# Run prediction on cropped lesions
# ----------------------------
valid_exts = (".jpg", ".jpeg", ".png")  # <-- allow multiple extensions

for fname in os.listdir(cropped_dir):
    if not fname.lower().endswith(valid_exts):
        continue

    img_path = os.path.join(cropped_dir, fname)
    img = cv2.imread(img_path)

    if img is None:
        print("⚠️ Failed to load:", fname)
        continue

    H, W = img.shape[:2]
    center_point = np.array([[W // 2, H // 2]])  # (x, y) center

    # Set image for predictor
    predictor.set_image(img)

    # Predict mask using center point
    masks, scores, logits = predictor.predict(
        point_coords=center_point,
        point_labels=np.array([1]),  # 1 = foreground
        multimask_output=False
    )

    mask = masks[0].astype(np.uint8) * 255  # binary mask

    # Get basename without extension
    base_name, _ = os.path.splitext(fname)

    # --- Save mask ---
    mask_out = os.path.join(output_dir, f"{base_name}_mask.png")
    cv2.imwrite(mask_out, mask)

    # --- Save overlay visualization ---
    overlay = img.copy()
    overlay[mask > 0] = (
        0.3 * overlay[mask > 0] + 0.7 * np.array([0, 255, 0])
    ).astype(np.uint8)
    overlay_out = os.path.join(output_dir, f"{base_name}_overlay.jpg")
    cv2.imwrite(overlay_out, overlay)

    # --- Save cut-out (background blacked out) ---
    cutout = np.zeros_like(img)
    cutout[mask > 0] = img[mask > 0]
    cutout_out = os.path.join(output_dir, f"{base_name}_cutout.png")
    cv2.imwrite(cutout_out, cutout)

    print(f"✅ Saved: {mask_out}, {overlay_out}, {cutout_out}")
