import os
import sys
import torch
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
import importlib
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch.nn.functional as F
import pynvml
from fileprocess.walkfolder import read_from_json


def torch_gpu_status():
    pynvml.nvmlInit()
    packages = ['torch', 'torchvision', 'torchaudio', 'torchtext']
    versions = [{"Py Version", sys.version}, {"Cuda version from torch", torch.version.cuda}, {"cudatoolkit_nvcc_--version":os.system("nvcc --version")}]
    torch_gpu_status = []
    for package in packages:
        module = importlib.util.find_spec(package)
        if module is not None:
            lib = importlib.import_module(package)
            versions.append([package, lib.__version__])
            # print(f"{package} version: {lib.__version__}")
        else:
            versions.append([package, "Not Installed"])
    print(versions)
    # print(torch.cuda.is_available())
    # print("Available GPUs:", torch.cuda.device_count())
    torch_gpu_status.append([{"torch.cuda.is_available": torch.cuda.is_available()}, {"Available GPUs": torch.cuda.device_count()}])
    if torch.cuda.is_available():
        device = torch.device("cuda")

        for i in range(torch.cuda.device_count()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            torch_gpu_status.append(f"GPU {i}: {name.encode('utf-8')}")
            torch_gpu_status.append(f"  Total Memory: {mem_info.total / 1024 ** 2 :.0f} MB")
            torch_gpu_status.append(f"  Utilization: {util}")
            torch_gpu_status.append(f"  Used Memory: {mem_info.used / 1024 ** 2 :.0f} MB")
            torch_gpu_status.append(f"  Free Memory: {mem_info.free / 1024 ** 2 :.0f} MB")
            torch_gpu_status.append(f"  Memory Utilization: {util.memory}%")
            # print(f"\nGPU {i}: {name.encode('utf-8')}")
            # print(f"  Total Memory: {mem_info.total / 1024 ** 2 :.0f} MB")
            # print(f"  Used Memory: {mem_info.used / 1024 ** 2 :.0f} MB")
            # print(f"  Free Memory: {mem_info.free / 1024 ** 2 :.0f} MB")
            # print(f"  GPU Utilization: {util.gpu}%")
            # print(f"  Memory Utilization: {util.memory}%")
    elif torch.backends.mps.is_available():
        print("CUDA is not available! ")
        device = torch.device("mps")
    else:
        print("CUDA is not available! ")
        device = torch.device("cpu")
    print(torch_gpu_status)
    return device, versions, torch_gpu_status


def cuda_setting(device):
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return

def check_flash_attention():
    sdpa_enabled = os.getenv('TORCH_CUDNN_SDPA_ENABLED')
    print(sdpa_enabled)
    # Sample test for scaled_dot_product_attention
    q = torch.randn(1, 256, 64, 64, device='cuda')
    k = torch.randn(1, 256, 64, 64, device='cuda')
    v = torch.randn(1, 256, 64, 64, device='cuda')

    out = F.scaled_dot_product_attention(q, k, v)
    print("Flash Attention Test Passed, Output shape:", out.shape)
    return


def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)



def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    print("box:", box)
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            print(box_coords.shape)
            print(box_coords)
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


def save_major_masks(masks, image_abs_path, mask_path="./predicted_masks", prefix = "sam2_", suffix = ".png", replace1=["ISIC","ISICsam2Mask"], replace2=[".jpg",""] ):
    base_name = os.path.basename(image_abs_path)
    base_name = prefix+base_name.replace(replace1[0], replace1[1]).replace(replace2[0], replace2[1])+suffix
    print(base_name)

    mask = masks[0]
    # Scale the values to the range [0, 255]
    mask_scaled = (mask * 255).astype(np.uint8)
    # Create an image from the array
    mask_image = Image.fromarray(mask_scaled)
    # Save as a PNG file
    save_path = os.path.join(mask_path, base_name)
    print(save_path)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    mask_image.save(save_path)
    return mask_image



def load_image(img_abs_path, edge_distance: int = 5):
    """
    Loads an image and computes its centroid and corner coordinates.

    Parameters:
    - img_abs_path (str): The absolute path to the image file.
    - edge_distance (int): The distance in pixels from the image edges to define the corner points.

    Returns:
    - image_np (numpy.ndarray): The image represented as a NumPy array in RGB format.
    - image (PIL.Image.Image): The PIL Image object.
    - img_abs_path (str): The absolute path to the image file.
    - centroid (list): The [x, y] coordinates of the image centroid.
    - edges (list of lists): Coordinates of the four corner points:
        [left top, left bottom, right top, right bottom]
    """

    # Open the image and convert it to RGB
    image = Image.open(img_abs_path)
    image = image.convert("RGB")
    image_np = np.array(image)

    # Get image dimensions
    height, width = image_np.shape[:2]

    # Calculate centroid coordinates
    centroid = [int(width / 2), int(height / 2)]

    # Calculate coordinates of the four corner pixels
    edges = [
        [edge_distance, edge_distance],  # left top
        [edge_distance, height - edge_distance - 1],  # left bottom
        [width - edge_distance - 1, edge_distance],  # right top
        [width - edge_distance - 1, height - edge_distance - 1]  # right bottom
    ]

    return image_np, image, img_abs_path, centroid, edges



def load_model(sam2_checkpoint , model_cfg):
    # sam2_checkpoint_large = "./checkpoints/sam2_hiera_large.pt"
    checkpoint = sam2_checkpoint
    # model_cfg = "./sam2_configs/sam2_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor

def load_large_model(device, sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt", model_cfg = "sam2_hiera_l.yaml"):
    # sam2_checkpoint_large = "./checkpoints/sam2_hiera_large.pt"
    checkpoint = sam2_checkpoint
    # model_cfg = "./sam2_configs/sam2_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor

# with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

def single_prompt(predictor, image_np, image_abs_path, image, adjust=0, input_point = np.array([[500, 500]]), input_label = np.array([1]), display_original = 0, display_masks=0):
    predictor.set_image(image_np)
    # input_point = np.array([[500, 375]])
    input_label = np.array([1])

    if display_original:
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # show_points(input_point, input_label, plt.gca())
        # plt.axis('on')
        # plt.show()
        pass

    # print("predictor._features=", predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks  = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    # print(masks.shape)
    # print(scores.shape)
    # print(logits.shape)

    if adjust != 0:
        masks[0] = adjust_mask(masks[0], adjust)

    if display_masks:
        show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
    return masks, scores, logits


def prompt2points(predictor, scores, image_np, img_abs_path, image):
    # # # 1/0 input points starts # # #
    # input_point = np.array([[500, 375], [1000, 725]])
    # input_label = np.array([1, 0])
    # print("Image shape:", image_np.shape)
    height, width = image_np.shape[0], image_np.shape[1]
    # print(f"Image width: {width}, height: {height}")

    input_point = np.array([[int(width / 2), int(height / 2)], [3, 3]])  # e.g., [[128, 128]]
    input_label = np.array([1, 0])

    mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )

    # show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label)

    return masks, scores, logits
# # # 1/0 input points ends # # #
def prompt_box_points(predictor, image_np, image_abs_path, image, input_box = np.array([52, 37, 952, 684]), input_point = np.array([[997, 45]]), input_label= np.array([0]),  adjust=0):
    predictor.set_image(image_np)
    # # # 1/0 input box & points ends # # #
    input_box = np.array([52, 37, 952, 684])
    input_point = np.array([[997, 45]])
    input_label = np.array([0])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=False,
    )
    if adjust != 0:
        masks[0] = adjust_mask(masks[0], adjust)
    show_masks(image, masks, scores, box_coords=input_box, point_coords=input_point, input_labels=input_label)
    save_major_masks(masks, mask_path="./predicted_masks", image_abs_path=img_abs_path)
    # print(type(masks), masks.shape)
    # print(masks)
    # print(np.min(masks))
    # print(np.max(masks))
    # print(np.mean(masks))
    # print(np.std(masks))
    return masks, scores, logits



def adjust_mask(mask, adjust: int = 1):
    """
    Adjusts a binary mask by expanding or shrinking it by a specified number of pixels.

    Parameters:
    - mask (numpy.ndarray): The binary mask to adjust. Should contain values of 0 and 1.
    - adjust (int): Number of pixels to adjust the mask by.
        - If adjust > 0, expands the mask outward.
        - If adjust < 0, shrinks the mask inward.
        - If adjust == 0, returns the original mask.

    Returns:
    - adjusted_mask (numpy.ndarray): The adjusted mask.
    """

    # Ensure the mask is binary
    mask = (mask > 0).astype(np.uint8)

    if adjust > 0:
        # Expand the mask outward
        adjusted_mask = binary_dilation(mask, iterations=adjust).astype(np.uint8)
    elif adjust < 0:
        # Shrink the mask inward
        adjusted_mask = binary_erosion(mask, iterations=abs(adjust)).astype(np.uint8)
    else:
        # No adjustment
        adjusted_mask = mask.copy()

    return adjusted_mask



def calculate_metrics(mask, label_path):
    """
    Calculate the IoU and Dice coefficient between a predicted mask and a ground truth mask.
    Parameters:
    - mask (numpy.ndarray): The predicted mask as a binary numpy array (values of 0 and 1).
    - label_path (str): The absolute file path to the ground truth mask image (PNG format).

    Returns:
    - iou (float): The Intersection over Union metric.
    - dice (float): The Dice coefficient metric.
    """
    # Load the ground truth mask
    gt_image = Image.open(label_path).convert('L')  # Convert to grayscale
    gt_mask = np.array(gt_image)

    # Ensure the ground truth mask is binary
    gt_mask = (gt_mask > 0).astype(np.uint8)

    # Check if shapes match
    if mask.shape != gt_mask.shape:
        print("The shapes of the predicted mask and ground truth mask do not match.")
        return None, None

    # Ensure the predicted mask is binary
    mask = (mask > 0).astype(np.uint8)

    # Compute intersection and union
    intersection = np.logical_and(mask, gt_mask).sum()
    union = np.logical_or(mask, gt_mask).sum()
    # Calculate IoU
    iou = intersection / union if union != 0 else 0
    # Calculate Dice coefficient
    total_pixels = mask.sum() + gt_mask.sum()
    dice = (2 * intersection) / total_pixels if total_pixels != 0 else 0
    return iou, dice


if __name__ == "__main__":
    device, versions, torch_gpu_status = torch_gpu_status()
    cuda_setting(device)
    check_flash_attention()
    np.random.seed(3)
    predictor = load_large_model(device=device)

    data = read_from_json("isic2016.json")
    jpgs = data["test_image"]
    print(len(jpgs))

    img_abs_path = "E:\\data\\rnsh\\ISBI2016\\ISBI2016Task1\\ISBI2016_ISIC_Part1_Training_Data\\ISIC_0000000.jpg"
    for img_abs_path in jpgs:
        # img_abs_path = ".\\images_input\\truck.jpg"
        image_np, image, img_abs_path, centroid, edges = load_image(img_abs_path)

        masks, scores, logits = single_prompt(predictor, image_np, img_abs_path, image, adjust=0, input_point = np.array([centroid]), input_label = np.array([1]), )
        # iou, dice = calculate_metrics(masks[0], label_path="E:\\data\\rnsh\\ISBI2016\\ISBI2016Task1\\ISBI2016_ISIC_Part1_Training_GroundTruth\\ISIC_0000000_Segmentation.png")

        masks, scores, logits = prompt2points(predictor, scores, image_np, img_abs_path, image)
        save_major_masks(masks, mask_path="./predicted_masks", image_abs_path=img_abs_path)
        # masks, scores, logits = prompt_box_points(predictor, image_np, img_abs_path, image, adjust=0)
        # print(scores, logits)
        # iou, dice = calculate_metrics(masks[0], label_path="E:\\data\\rnsh\\ISBI2016\\ISBI2016Task1\\ISBI2016_ISIC_Part1_Training_GroundTruth\\ISIC_0000000_Segmentation.png")
        # print(iou, dice)

