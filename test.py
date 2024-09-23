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
from sklearn.metrics import confusion_matrix
from model.unet import UNet
from scipy.ndimage import binary_dilation

from test_needle_fit import mask_parse
from utils import create_dir, seeding, check_data_empty


def calculate_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    # Avoid undefined metrics by setting zero_division parameter
    score_jaccard = jaccard_score(y_true, y_pred, zero_division=0)
    score_f1 = f1_score(y_true, y_pred, zero_division=0)
    score_recall = recall_score(y_true, y_pred, zero_division=0)
    score_precision = precision_score(y_true, y_pred, zero_division=0)
    score_acc = accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    score_dice = (2.0 * tp) / ((2.0 * tp) + fp + fn) if ( (2.0 * tp) + fp + fn ) > 0 else 0

    # print("Confusion Matrix:")
    # print(f"True Negative (TN): {tn}")
    # print(f"False Positive (FP): {fp}")
    # print(f"False Negative (FN): {fn}")
    # print(f"True Positive (TP): {tp}")

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_dice]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask

if __name__ == "__main__":
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

    seeding(args.seed)

    create_dir(segmentation_results_path)

    test_x = sorted(glob(os.path.join(os.getcwd(), args.test_path, 'images/*')))
    test_y = sorted(glob(os.path.join(os.getcwd(), args.test_path, 'masks/*')))

    logger.info(f"Test image size: {len(test_x)}, test mask size: {len(test_y)}")


    check_data_empty(test_x, test_y, 'testing')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet()
    model = model.to(device)
    checkpoint = torch.load(os.path.join(os.getcwd(), args.ckpt), map_location=device)

    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(new_state_dict)
    print("Model state dict loaded successfully")
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0 , 0.0]
    time_taken = []

    batch_size = 590
    
    for i in range(0, len(test_x), batch_size):
        batch_test_x = test_x[i:i+batch_size]
        batch_test_y = test_y[i:i+batch_size]

        batch_metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0 , 0.0]
        batch_time_taken = []

        for idx, (x, y) in tqdm(enumerate(zip(batch_test_x, batch_test_y)), total=len(batch_test_x)):
            name = x.split('/')[-1].split('.')[0]

            img = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (args.img_size, args.img_size))
            x = np.expand_dims(img, axis=0)  # Adding a channel dimension
            x = np.expand_dims(x, axis=0)  # Adding a batch dimension
            x = x / 255.0
           # x = np.expand_dims(x, axis=0)
            x = x.astype(np.float32)
            x = torch.from_numpy(x)
            x = x.to(device)

            mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (args.img_size, args.img_size))
            #print("shape of mask=", mask.shape)
            #print("max and min in mask=", np.max(mask), np.min(mask))

            dilation_pixels = args.dilation_pixels
            if dilation_pixels > 0:
                original_mask = binary_dilation(mask, iterations=dilation_pixels)
            else:
                original_mask = mask / 255.0

            y = np.expand_dims(original_mask, axis=0)
            y = np.expand_dims(y, axis=0)
            y = y.astype(np.float32)
            y = torch.from_numpy(y)
            y = y.to(device)

            with torch.no_grad():
                start_time = time.time()
                pred_y = model(x)
              #  print("max and min in predicted mask=", np.max(pred_y.cpu().numpy()), np.min(pred_y.cpu().numpy()))


                pred_y = torch.sigmoid(pred_y)
              #  print("max and min in predicted mask after sigmoid=", np.max(pred_y.cpu().numpy()), np.min(pred_y.cpu().numpy()))

                total_time = time.time() - start_time
                batch_time_taken.append(total_time)
              #  print("max and min in original mask=", np.max(y.cpu().numpy()), np.min(y.cpu().numpy()))
                score = calculate_metrics(y, pred_y)
                batch_metrics_score = list(map(add, batch_metrics_score, score))

                pred_y = pred_y[0].cpu().numpy()
                pred_y = np.squeeze(pred_y, axis=0)
                pred_y = pred_y > 0.5
                pred_y = np.array(pred_y, dtype=np.uint8)
            
          #  original_mask= original_mask<.5
           # original_mask = mask_parse(original_mask)
           # pred_y = mask_parse(pred_y)
            line = np.ones([args.img_size, 10]) * 128

            cat_img = np.concatenate(
                [img, line, original_mask*255, line, pred_y * 255], axis=1
            )

            cv2.imwrite(os.path.join(os.getcwd(), segmentation_results_path, f'{name}.png'), cat_img)

        jaccard = batch_metrics_score[0]/len(batch_test_x)
        f1 = batch_metrics_score[1]/len(batch_test_x)
        recall = batch_metrics_score[2]/len(batch_test_x)
        precision = batch_metrics_score[3]/len(batch_test_x)
        acc = batch_metrics_score[4]/len(batch_test_x)
        dice = batch_metrics_score[5]/len(batch_test_x)

        print(f"Batch Jaccard Score: {jaccard}")
        print(f"Batch F1 Score: {f1}")
        print(f"Batch Recall Score: {recall}")
        print(f"Batch Precision Score: {precision}")
        print(f"Batch Accuracy Score: {acc}")
        print(f"Batch Dice Score: {dice}")
        
        total_time = np.sum(batch_time_taken)
        mean_time = np.mean(batch_time_taken)
        print(f"Total time: {total_time}")
        print(f"Mean time: {mean_time}")

        metrics_score = list(map(add, metrics_score, batch_metrics_score))
        time_taken.extend(batch_time_taken)

    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    dice = metrics_score[5]/len(test_x)

    total_time = np.sum(time_taken)
    mean_time = np.mean(time_taken)

    logger.info(f"Jaccard Score: {jaccard}")
    logger.info(f"F1 Score: {f1}")
    logger.info(f"Recall Score: {recall}")
    logger.info(f"Precision Score: {precision}")
    logger.info(f"Accuracy Score: {acc}")
    logger.info(f"Dice Score: {dice}")

    logger.info(f"Total time: {total_time}")
    logger.info(f"Mean time: {mean_time}")
