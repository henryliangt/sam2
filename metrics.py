import os
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from fileprocess.walkfolder import read_from_json, write_to_json
from fileprocess.csvreadwrite import write_list_to_csv


def evaluate_segmentation(true_mask_path, pred_mask_path, display=1, save_dir='./predicted_masks/'):
    """
    Evaluate the segmentation performance by comparing the ground truth mask with the predicted mask.

    Parameters:
    - true_mask_path (str): The absolute path to the ground truth mask image (PNG format).
    - pred_mask_path (str): The absolute path to the predicted mask image (PNG format).

    Returns:
    - basenames (str): A string combining the basenames of the true and predicted masks.
    - accuracy (float): The pixel-wise accuracy of the prediction.
    - iou (float): The Intersection over Union metric.
    - dice (float): The Dice coefficient metric.
    """
    # Get the basenames of the mask files
    basename_true = os.path.basename(true_mask_path)
    basename_pred = os.path.basename(pred_mask_path)
    basenames = f"{basename_true}_{basename_pred}"

    # Load the ground truth and predicted masks
    true_mask_img = Image.open(true_mask_path).convert('L')  # Convert to grayscale
    pred_mask_img = Image.open(pred_mask_path).convert('L')   # Convert to grayscale

    # Convert images to numpy arrays
    true_mask = np.array(true_mask_img)
    pred_mask = np.array(pred_mask_img)

    # Check if shapes match
    if true_mask.shape != pred_mask.shape:
        print("Error: The shapes of the ground truth mask and predicted mask do not match.")
        return basenames, None, None, None

    # Ensure masks are binary (values 0 or 1)
    true_mask = (true_mask > 0).astype(np.uint8)
    pred_mask = (pred_mask > 0).astype(np.uint8)

    # Calculate pixel-wise accuracy
    total_pixels = true_mask.size
    correct_pixels = (true_mask == pred_mask).sum()
    accuracy = correct_pixels / total_pixels

    # Compute intersection and union
    intersection = np.logical_and(true_mask, pred_mask).sum()
    union = np.logical_or(true_mask, pred_mask).sum()

    # Calculate IoU
    iou = intersection / union if union != 0 else 0

    # Calculate Dice coefficient
    sum_masks = true_mask.sum() + pred_mask.sum()
    dice = (2 * intersection) / sum_masks if sum_masks != 0 else 0

    # Display masks if requested
    if display:
        # Create a figure with two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Display ground truth mask
        axes[0].imshow(true_mask, cmap='gray')
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')

        # Display predicted mask
        axes[1].imshow(pred_mask, cmap='gray')
        axes[1].set_title('Predicted')
        axes[1].axis('off')

        # Set the main title of the figure using the basename of the ground truth mask
        fig.suptitle(basename_true, fontsize=16)

        # Adjust layout and display the figure
        plt.tight_layout()
        plt.show()
    else:
        if save_dir:
            # Ensure the save directory exists
            os.makedirs(save_dir, exist_ok=True)

            # Create a comparison image and save it
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Display ground truth mask
            axes[0].imshow(true_mask, cmap='gray')
            axes[0].set_title('Ground Truth')
            axes[0].axis('off')

            # Display predicted mask
            axes[1].imshow(pred_mask, cmap='gray')
            axes[1].set_title('Predicted')
            axes[1].axis('off')

            # Set the main title of the figure using the basename of the ground truth mask
            fig.suptitle(basename_true, fontsize=16)

            # Adjust layout
            plt.tight_layout()

            # Save the figure to the specified directory with a specific filename
            save_path = os.path.join(save_dir, f"compare_sam2_{basename_true}")
            fig.savefig(save_path)

            # Close the figure to avoid memory issues
            plt.close()
            # print(f"Comparison saved to {save_path}")

    return basenames, accuracy, iou, dice




def match_masks(true_masks, pred_masks):
    """
    Matches ground truth masks with predicted masks based on a unique identifier in their filenames.
    Parameters:
    - true_masks (list of str): List of absolute paths to the ground truth mask files.
    - pred_masks (list of str): List of absolute paths to the predicted mask files.

    Returns:
    - true_pred_masks_pair (list of dict): A list of dictionaries containing matched pairs:
        [
            {
                'true_mask': absolute path to the ground truth mask,
                'pred_mask': absolute path to the predicted mask,
            },
            ...
        ]
    """
    # Dictionary to hold mapping from unique identifier to file path for ground truth masks
    true_mask_dict = {}
    for true_mask in true_masks:
        # Extract the basename
        basename = os.path.basename(true_mask)
        # print("True", true_mask, basename)
        # Use regex to extract the numeric identifier
        match = re.search(r'(\d+)', basename)
        if match:
            identifier = match.group(1)
            true_mask_dict[identifier] = true_mask
            # print("true identifier", identifier)
        else:
            print(f"No identifier found in ground truth mask filename: {basename}")

    # Dictionary to hold mapping from unique identifier to file path for predicted masks
    pred_mask_dict = {}
    pred_mask_pattern = re.compile(r'^sam2_ISICsam2Mask_(\d{7})\.png$')
    for pred_mask in pred_masks:
        basename = os.path.basename(pred_mask)
        match = pred_mask_pattern.match(basename)
        # print("Predict: ", true_mask, basename)
        if match:
            identifier = match.group(1)
            pred_mask_dict[identifier] = pred_mask
            # print("predict identifier", identifier)
        else:
            print(f"No identifier found in predicted mask filename: {basename}")

    # Match the masks based on the identifier
    true_pred_masks_pair = []
    for identifier, true_mask_path in true_mask_dict.items():
        pred_mask_path = pred_mask_dict.get(identifier)
        if pred_mask_path:
            pair = {
                'true_mask': true_mask_path,
                'pred_mask': pred_mask_path
            }
            true_pred_masks_pair.append(pair)
        else:
            print(f"No matching predicted mask found for ground truth mask with identifier {identifier}")

    return true_pred_masks_pair


def read_and_match_pair_write_json():
    data      = read_from_json("isic2016.json")
    true_mask = data["train_label"]
    sam2_mask = data["train_label_sam2"]

    #as another group option
    # true_mask = data["test_label"]
    # sam2_mask = data["test_label_sam2"]
    #as another group option
    true_mask__pred_mask_pair = match_masks(true_mask, sam2_mask)
    print(true_mask__pred_mask_pair)
    data_dict = {
        "true_mask__pred_mask_train_pair": true_mask__pred_mask_pair
    }
    write_to_json(data_dict, filename="../sam2/isic2016.json", mode='a')
    return


if __name__ == "__main__":
    train_pair = read_from_json("isic2016.json")["true_mask__pred_mask_train_pair"]
    test_pair = read_from_json("isic2016.json")["true_mask__pred_mask_test_pair"]
    pairs = test_pair     # choose 1, test_pair or train_pair
    # pairs = train_pair  # choose 1, test_pair or train_pair
    performance = []

    for pair in pairs:
        # print(pair)
        # print(pair["true_mask"], pair["pred_mask"])
        basenames, accuracy, iou, dice = evaluate_segmentation(pair["true_mask"], pair["pred_mask"], display=0, save_dir='./predicted_masks/')
        performance.append([basenames, accuracy, iou, dice])
        print(basenames, accuracy, iou, dice)
        # print(performance)
    write_list_to_csv(performance, "sam2performisic2016.csv", mode="a")
