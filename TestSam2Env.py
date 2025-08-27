import torch
import sam2
import numpy as np
import os

print(np.__version__)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("SAM2 version OK")



from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam_checkpoint = "checkpoints/sam2_hiera_large.pt"
model_cfg = os.path.join("C:/Users/henry/PycharmProjects/rnsh/sam2/sam2_configs", "sam2_hiera_l.yaml")

sam2_model = build_sam2(model_cfg, sam_checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
predictor = SAM2ImagePredictor(sam2_model)

print("✅ SAM2 model loaded successfully!")
