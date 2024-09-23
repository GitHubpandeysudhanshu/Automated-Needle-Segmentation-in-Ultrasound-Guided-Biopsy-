# import os
# import time
# from operator import add
# import numpy as np
# from glob import glob
# import cv2
# from tqdm import tqdm
# import torch
# from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
# import argparse
# from loguru import logger
# from model.unet import UNet
# from utils import create_dir, seeding, check_data_empty
# import torch.nn as nn
# from scipy.ndimage import binary_dilation
# from skimage.measure import shannon_entropy
#
# def calculate_metrics(y_true, y_pred):
#     # Convert tensors to numpy arrays if necessary
#     if isinstance(y_true, torch.Tensor):
#         y_true = y_true.cpu().numpy()
#     if isinstance(y_pred, torch.Tensor):
#         y_pred = y_pred.cpu().numpy()
#
#     # Binarize predictions and ground truth
#     y_true = y_true > 0.5
#     y_true = y_true.astype(np.uint8)
#     y_true = y_true.reshape(-1)
#
#     y_pred = y_pred > 0.5
#     y_pred = y_pred.astype(np.uint8)
#     y_pred = y_pred.reshape(-1)
#
#     # Calculate metrics
#     score_jaccard = jaccard_score(y_true, y_pred)
#     score_f1 = f1_score(y_true, y_pred)
#     score_recall = recall_score(y_true, y_pred)
#     score_precision = precision_score(y_true, y_pred)
#     score_acc = accuracy_score(y_true, y_pred)
#
#     return [score_jaccard, score_f1, score_recall, score_precision, score_acc]
#
# def mask_parse(mask):
#     mask = np.expand_dims(mask, axis=-1)  # (512, 512, 1)
#     mask = np.concatenate([mask, mask, mask], axis=-1)  # (512, 512, 3)
#     return mask
#
# if __name__ == "__main__":
#     # Command args
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--test_path', type=str, default='data/test/', help='root dir for the test data directory')
#     parser.add_argument('--output', type=str, default='results', help="output dir for saving the segmentation results")
#     parser.add_argument('--seed', type=int, default=42, help='random seed')
#     parser.add_argument('--ckpt', type=str, default='checkpoints/checkpoint.pth', help='pretrained checkpoint')
#     parser.add_argument('--img_size', type=int, default=512, help='input image size of network input')
#     parser.add_argument('--dilation_pixels', type=int, default=0, help='dilation pixels for DriveDataset')
#     args = parser.parse_args()
#
#     args.exp = args.ckpt.split('/')[1]
#     output_path = os.path.join(args.output, "{}".format(args.exp))
#     segmentation_results_path = os.path.join(output_path, 'segmentation_results')
#     log_path = os.path.join(output_path, "runs")
#
#     # Create results folder if it doesn't exist
#     if not os.path.exists(segmentation_results_path):
#         os.makedirs(segmentation_results_path)
#
#     # Load the test dataset
#     test_x = sorted(glob(os.path.join(os.getcwd(), args.test_path, 'images/*')))
#     test_y = sorted(glob(os.path.join(os.getcwd(), args.test_path, 'masks/*')))
#
#     # Load the checkpoint
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = UNet()
#     if torch.cuda.device_count() > 1:
#         model = nn.DataParallel(model)
#     model = model.to(device)
#
#     checkpoint = torch.load(os.path.join(os.getcwd(), args.ckpt), map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
#
#     # Calculate the metrics
#     metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
#     time_taken = []
#     avg_entropy = 0
#
#     # Open a text file to save the results
#     with open(os.path.join(output_path, 'results.txt'), 'w') as f:
#         for idx, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
#             name = x.split('/')[-1].split('.')[0]
#
#             # Reading the image
#             img = cv2.imread(x, cv2.IMREAD_COLOR)  # (512, 512, 3)
#             img = cv2.resize(img, (args.img_size, args.img_size))
#             x = np.transpose(img, (2, 0, 1))  # (3, 512, 512)
#             x = x / 255.0
#             x = np.expand_dims(x, axis=0)  # (1, 3, 512, 512), batch of 1 image
#             x = x.astype(np.float32)
#             x = torch.from_numpy(x).to(device)
#
#             # Reading the mask
#             mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  # (512, 512)
#             mask = cv2.resize(mask, (args.img_size, args.img_size))
#             dilation_pixels = args.dilation_pixels
#             if dilation_pixels > 0:
#                 original_mask = binary_dilation(mask, iterations=dilation_pixels)
#             else:
#                 original_mask = mask / 255.0
#
#             y = np.expand_dims(original_mask, axis=(0, 1))  # (1, 1, 512, 512)
#             y = y.astype(np.float32)
#             y = torch.from_numpy(y).to(device)
#
#             # Prediction
#             with torch.no_grad():
#                 start_time = time.time()
#                 pred_y = model(x)
#                 pred_y = torch.sigmoid(pred_y).cpu().numpy().squeeze()
#                 elapsed_time = time.time() - start_time
#
#             # Calculate metrics
#             metrics = calculate_metrics(y.cpu().numpy().squeeze(), pred_y)
#             metrics_score = list(map(add, metrics_score, metrics))
#             time_taken.append(elapsed_time)
#
#             # Save intersection mask
#             intersection_mask = (original_mask == (pred_y > .5 ).astype(np.uint8)).astype(np.uint8)
#
#             # Save entropy image
#             entropy_img = -pred_y * np.log2(pred_y + 1e-7) - (1 - pred_y) * np.log2(1 - pred_y + 1e-7)
#             correct_pixel_entropy_image = entropy_img * intersection_mask
#             # Compute the average of non-zero entropy values
#             non_zero_entropy_values = correct_pixel_entropy_image[correct_pixel_entropy_image > 0]
#             avg_non_zero_entropy = np.mean(non_zero_entropy_values) if len(non_zero_entropy_values) > 0 else 0
#
#             # Write the average of non-zero entropy values to the file
#             f.write(f"{name} Average Non-Zero Entropy: {avg_non_zero_entropy:.4f}\n")
#
#             # Average entropy
#             avg_entropy += avg_non_zero_entropy
#
#             intersection_mask_image = (intersection_mask * 255).astype(np.uint8)
#             cv2.imwrite(os.path.join(segmentation_results_path, f"{name}_intersection.png"), intersection_mask_image)
#
#         # Compute averages and write them to the file
#         avg_metrics = np.array(metrics_score) / len(test_x)
#         avg_time = np.mean(time_taken)
#         avg_entropy = avg_entropy / len(test_x)
#
#         # Debugging information
#         print(f"Debug Info - Average Metrics: {avg_metrics}")
#         print(f"Debug Info - Average Time: {avg_time}")
#         print(f"Debug Info - Average Entropy: {avg_entropy}")
#
#         f.write(f"\nJaccard Score: {avg_metrics[0]:.4f}\n")
#         f.write(f"F1 Score: {avg_metrics[1]:.4f}\n")
#         f.write(f"Recall Score: {avg_metrics[2]:.4f}\n")
#         f.write(f"Precision Score: {avg_metrics[3]:.4f}\n")
#         f.write(f"Dice Score: {avg_metrics[4]:.4f}\n")
#         f.write(f"Average Inference Time: {avg_time:.4f} seconds\n")
#         f.write(f"Average Entropy: {avg_entropy:.4f}\n")
#
#     print("Average Entropy and metrics calculated and saved.")



import os
import time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import argparse
from loguru import logger
from model.unet import UNet
from utils import create_dir, seeding, check_data_empty
import torch.nn as nn
from scipy.ndimage import binary_dilation
from skimage.measure import shannon_entropy

def calculate_metrics(y_true, y_pred):
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Binarize predictions and ground truth
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    # Calculate metrics
    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)  # (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  # (512, 512, 3)
    return mask

if __name__ == "__main__":
    # Command args
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default='data/test/', help='root dir for the test data directory')
    parser.add_argument('--output', type=str, default='results', help="output dir for saving the segmentation results")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--ckpt', type=str, default='checkpoints/checkpoint.pth', help='pretrained checkpoint')
    parser.add_argument('--img_size', type=int, default=512, help='input image size of network input')
    parser.add_argument('--dilation_pixels', type=int, default=0, help='dilation pixels for DriveDataset')
    args = parser.parse_args()

    args.exp = args.ckpt.split('/')[1]
    output_path = os.path.join(args.output, "{}".format(args.exp))
    segmentation_results_path = os.path.join(output_path, 'segmentation_results')
    log_path = os.path.join(output_path, "runs")

    # Create results folder if it doesn't exist
    if not os.path.exists(segmentation_results_path):
        os.makedirs(segmentation_results_path)

    # Load the test dataset
    test_x = sorted(glob(os.path.join(os.getcwd(), args.test_path, 'images/*')))
    test_y = sorted(glob(os.path.join(os.getcwd(), args.test_path, 'masks/*')))

    # Load the checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    checkpoint = torch.load(os.path.join(os.getcwd(), args.ckpt), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Calculate the metrics
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []
    avg_entropy = 0

    # Open a text file to save the results
    with open(os.path.join(output_path, 'results.txt'), 'w') as f:
        for idx, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
            name = x.split('/')[-1].split('.')[0]

            # Reading the image
            img = cv2.imread(x, cv2.IMREAD_COLOR)  # (512, 512, 3)
            img = cv2.resize(img, (args.img_size, args.img_size))
            x = np.transpose(img, (2, 0, 1))  # (3, 512, 512)
            x = x / 255.0
            x = np.expand_dims(x, axis=0)  # (1, 3, 512, 512), batch of 1 image
            x = x.astype(np.float32)
            x = torch.from_numpy(x).to(device)

            # Reading the mask
            mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  # (512, 512)
            mask = cv2.resize(mask, (args.img_size, args.img_size))
            dilation_pixels = args.dilation_pixels
            if dilation_pixels > 0:
                original_mask = binary_dilation(mask, iterations=dilation_pixels)
            else:
                original_mask = mask / 255.0

            y = np.expand_dims(original_mask, axis=(0, 1))  # (1, 1, 512, 512)
            y = y.astype(np.float32)
            y = torch.from_numpy(y).to(device)

            # Prediction
            with torch.no_grad():
                start_time = time.time()
                pred_y = model(x)
                pred_y = torch.sigmoid(pred_y).cpu().numpy().squeeze()
                elapsed_time = time.time() - start_time

            # Calculate metrics
            metrics = calculate_metrics(y.cpu().numpy().squeeze(), pred_y)
            metrics_score = list(map(add, metrics_score, metrics))
            time_taken.append(elapsed_time)

            # Save intersection mask
            intersection_mask = (original_mask == (pred_y > .5 ).astype(np.uint8)).astype(np.uint8)

            # Save entropy image
            entropy_img = -pred_y * np.log2(pred_y + 1e-7) - (1 - pred_y) * np.log2(1 - pred_y + 1e-7)
            correct_pixel_entropy_image = entropy_img * intersection_mask

            # Compute the average of non-zero entropy values
            non_zero_entropy_values = correct_pixel_entropy_image[correct_pixel_entropy_image > 0]
            avg_non_zero_entropy = np.mean(non_zero_entropy_values) if len(non_zero_entropy_values) > 0 else 0

            # Write the average of non-zero entropy values to the file
            f.write(f"{name} Average Non-Zero Entropy: {avg_non_zero_entropy:.4f}\n")

            # Average entropy
            avg_entropy += avg_non_zero_entropy

            # Create color-coded mask based on entropy
            color_coded_mask = np.zeros((entropy_img.shape[0], entropy_img.shape[1], 3), dtype=np.uint8)
            color_coded_mask[(entropy_img > avg_non_zero_entropy)] = [0, 0, 255]  # Red for high entropy
            color_coded_mask[(entropy_img <= avg_non_zero_entropy)] = [255, 255, 255]  # White for low entropy

            # Save the results
            intersection_mask_image = (intersection_mask * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(segmentation_results_path, f"{name}_intersection.png"), intersection_mask_image)

            color_coded_mask_image = color_coded_mask
            cv2.imwrite(os.path.join(segmentation_results_path, f"{name}_color_coded.png"), color_coded_mask_image)

        # Compute averages and write them to the file
        avg_metrics = np.array(metrics_score) / len(test_x)
        avg_time = np.mean(time_taken)
        avg_entropy = avg_entropy / len(test_x)

        # Debugging information
        print(f"Debug Info - Average Metrics: {avg_metrics}")
        print(f"Debug Info - Average Time: {avg_time}")
        print(f"Debug Info - Average Entropy: {avg_entropy}")

        f.write(f"\nJaccard Score: {avg_metrics[0]:.4f}\n")
        f.write(f"F1 Score: {avg_metrics[1]:.4f}\n")
        f.write(f"Recall Score: {avg_metrics[2]:.4f}\n")
        f.write(f"Precision Score: {avg_metrics[3]:.4f}\n")
        f.write(f"Dice Score: {avg_metrics[4]:.4f}\n")
        f.write(f"Average Inference Time: {avg_time:.4f} seconds\n")
        f.write(f"Average Entropy: {avg_entropy:.4f}\n")

    print("Average Entropy and metrics calculated and saved.")
