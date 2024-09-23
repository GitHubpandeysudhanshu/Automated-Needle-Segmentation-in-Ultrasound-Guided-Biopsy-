# Automated Needle Segmentation in Ultrasound-Guided Biopsy

This repository contains a U-Net implementation for image segmentation tasks, built using PyTorch.

## Requirements

- **Operating System**: Linux (tested on Ubuntu 22.04.3 LTS)
- **Python Version**: Python 3.10.4 (other versions should work as well)
- **PyTorch**: Version 2.0.1+cu117

The code should work with other versions of the above tools, but these were used during testing.

## Data Preprocessing

To preprocess the data, run the following command:

```bash
python data_aug.py --dataset <dataset-folder-name>


## Training
To train the model:
Place your dataset (both images and masks) in datasets/<your-folder-name>.
Run the data augmentation script to resize and process the dataset.
Train the model by running:
python train.py --train_path "datasets/<dataset-folder-name>/processed/train" \
                --val_path "datasets/<dataset-folder-name>/processed/val" \
                --dataset "<dataset-name>" --dilation_pixels "<no of pixels to dilate>"

You can use python train.py --help for any help


## Testing
python test.py --test_path "datasets/<dataset-folder-name>/processed/test" \
               --ckpt "results/<dataset-name>/checkpoints/<checkpoint-file-name>.pth"

