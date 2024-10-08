import os
from typing import Any
import cv2
import imageio.v2 as imageio
from glob  import glob
from tqdm import tqdm
from albumentations import HorizontalFlip, VerticalFlip, Rotate
import random
import argparse
from loguru import logger
from utils import check_data_empty

class Augmentation:
    def __init__(self, dataset):
        super(Augmentation, self).__init__()

        self.dataset_path = os.path.join(os.getcwd(), 'datasets', dataset)
        self.img_size = (512,512)
        self.rand_rotation = 180

        # check if the dataset folder exists
        if not os.path.exists(self.dataset_path):
            raise NotADirectoryError(f"Dataset {dataset} doesn't exist in datasets folder.")

    def load_data(self):
        train_x = sorted(glob(os.path.join(self.dataset_path, 'train', 'images', '*.png')))
        train_y = sorted(glob(os.path.join(self.dataset_path, 'train', 'masks', '*.png'))) # mask gt

        val_x = sorted(glob(os.path.join(self.dataset_path, 'val', 'images', '*.png')))
        val_y = sorted(glob(os.path.join(self.dataset_path, 'val', 'masks', '*.png'))) # mask gt

        test_x = sorted(glob(os.path.join(self.dataset_path, 'test', 'images', '*.png')))
        test_y = sorted(glob(os.path.join(self.dataset_path, 'test', 'masks', '*.png'))) # mask gt
        
        return (train_x, train_y), (val_x, val_y), (test_x, test_y)
    
    def create_dirs(self, paths):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    def crop_image(self, img, target_size=(677, 753)):
        # Get the dimensions of the image
        height, width, _ = img.shape

        # Calculate the starting points of the crop
        start_y = (height - target_size[0]) // 2
        start_x = (width - target_size[1]) // 2

        # Crop the image
        cropped_img = img[start_y:start_y + target_size[0], start_x:start_x + target_size[1]]

        return cropped_img

    def crop_mask(self, img, target_size=(677, 753)):
        # Get the dimensions of the image
        height, width = img.shape

        # Calculate the starting points of the crop
        start_y = (height - target_size[0]) // 2
        start_x = (width - target_size[1]) // 2

        # Crop the image
        cropped_mask = img[start_y:start_y + target_size[0], start_x:start_x + target_size[1]]

        return cropped_mask

    def augment_data(self, images, masks, save_dir, augment=True):
        for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
            img_name = os.path.splitext(os.path.basename(x))[0]

            train = cv2.imread(x, cv2.IMREAD_COLOR)
            mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

           # train = self.crop_image(train)
          #  mask = self.crop_mask(mask)

            if augment:
                augmentations = [
                    HorizontalFlip(p=1.0),
                    VerticalFlip(p=1.0),
                    # Rotate(limit=45, p=1.0)
                ]

                augment_results = [{"image": train, "mask": mask}]
                
                for aug in augmentations:
                    augmented = aug(image=train, mask=mask)
                    augment_results.append(augmented)

            else:
                augment_results = [{"image": train, "mask": mask}]

            for index, result in enumerate(augment_results):
               # i = cv2.resize(result["image"], self.img_size)
              #  m = cv2.resize(result["mask"], self.img_size)

                image_path = os.path.join(save_dir, "images", f"{img_name}_{index}.png")
                mask_path = os.path.join(save_dir, "masks", f"{img_name}_{index}.png")

                # cv2.imwrite(image_path, i)
                # cv2.imwrite(mask_path, m)
                cv2.imwrite(image_path, result["image"])  # Save augmented image
                cv2.imwrite(mask_path, result["mask"])  # Save augmented mask




if __name__ == "__main__":
    # command args
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--dataset', type=str, default='needle_dataset', help='dataset folder name')  # <dataset folder name>

    args = parser.parse_args()

    # set seeds
    random.seed(args.seed)

    # init class
    aug = Augmentation(dataset=args.dataset)

    # load train and test data
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = aug.load_data()

    logger.info(f"train image size: {len(train_x)}, train mask size: {len(train_y)}")    
    logger.info(f"valid image size: {len(val_x)}, valid mask size: {len(val_y)}")    
    logger.info(f"test image size: {len(test_x)}, test mask size: {len(test_y)}")    

    check_data_empty(train_x, train_y, 'training')
    check_data_empty(val_x, val_y, 'validation')
    check_data_empty(test_x, test_y, 'testing')

    # validate data length
    assert len(train_x) == len(train_y), "Train and ground truth data are not equal in length"
    assert len(val_x) == len(val_y), "Validation and ground truth data are not equal in length"
    assert len(test_x) == len(test_y), "Test and ground truth data are not equal in length"

    # create new dir for augmented data
    aug.create_dirs(
        [os.path.join(os.getcwd(), 'datasets', args.dataset, 'processed', 'train/images'),
         os.path.join(os.getcwd(), 'datasets', args.dataset, 'processed', 'train/masks'),

         os.path.join(os.getcwd(), 'datasets', args.dataset, 'processed', 'val/images'),
         os.path.join(os.getcwd(), 'datasets', args.dataset, 'processed', 'val/masks'),

         os.path.join(os.getcwd(), 'datasets', args.dataset, 'processed', 'test/images'),
         os.path.join(os.getcwd(), 'datasets', args.dataset, 'processed', 'test/masks')]
    )

    aug.augment_data(train_x, train_y, os.path.join(os.getcwd(), 'datasets/needle_dataset_new/processed/train/'), augment=True)
    aug.augment_data(test_x, test_y, os.path.join(os.getcwd(), 'datasets/needle_dataset_new/processed/val/'), augment=True)
    aug.augment_data(test_x, test_y, os.path.join(os.getcwd(), 'datasets/needle_dataset_new/processed/test/'), augment=True)
