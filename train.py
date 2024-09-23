import os
import time
from glob import glob 
import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from model.unet import UNet
from dataset import DriveDataset
from callbacks.early_stopping import EarlyStopping

from loss import DiceLoss, DiceBCELoss, CombinedLoss, FocalLoss, IoULoss
from utils import seeding, create_dir, epoch_time, check_data_empty

from loguru import logger
from torchsummary import summary
import wandb
from tqdm import tqdm


def train(model, train_loader, optimizer, loss_fn, device):
    epoch_loss = 0.0
    model.train()

    progress_bar = tqdm(train_loader, desc='Training', total=len(train_loader))

    for x, y in train_loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        # print("x_shape=", x.shape)
        # print("y_shape=", y.shape)
        optimizer.zero_grad()

        y_pred = model(x)
        # print(f'y_pred shape: {y_pred.shape}, y shape: {y.shape}')
 
        loss = loss_fn(y_pred, y)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(x))})
        progress_bar.update()

    epoch_loss = epoch_loss / len(train_loader)
    return epoch_loss

def evaluate(model, valid_loader, loss_fn, device):
    epoch_loss = 0.0
    model.eval()

    progress_bar = tqdm(valid_loader, desc='Validation', total=len(valid_loader))

    with torch.no_grad():
        for x, y in valid_loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            progress_bar.set_postfix({'validation_loss': '{:.3f}'.format(loss.item()/len(x))})
            progress_bar.update()

    epoch_loss = epoch_loss / len(valid_loader)
    return epoch_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str, default='data/train', help='root dir for train data')
    parser.add_argument('--dilation_pixels', type=int, default=0, help='number of dilation iterations for masks')
    parser.add_argument('--val_path', type=str, default='data/val', help='root dir for validation data')
    parser.add_argument('--output', type=str, default='results', help="output dir for saving the segmentation results")
    parser.add_argument('--dataset', type=str, default='kvasir', help='experiment_name')
    parser.add_argument('--max_epochs', type=int, default=50, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='segmentation network learning rate')
    parser.add_argument('--patience', type=int, default=7, help='patience for lr scheduler')
    parser.add_argument('--img_size', type=int, default=512, help='input image size of network input')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--ckpt', type=str, default='checkpoints/checkpoint.pth', help='pretrained checkpoint')
    parser.add_argument('--wandb', type=bool, default=False, help='wandb logging control flag')

    args = parser.parse_args()

    args.exp = args.dataset + '_' + str(args.img_size)
    output_path = os.path.join(args.output, "{}".format(args.exp))
    snapshot_path = output_path + '_epo' + str(args.max_epochs)
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_s' + str(args.seed)

    checkpoint_path = os.path.join(output_path, args.ckpt)
    
    checkpoint_file = args.exp + '_epo' + str(args.max_epochs)
    checkpoint_file = checkpoint_file + '_bs' + str(args.batch_size)
    checkpoint_file = checkpoint_file + '_lr' + str(args.base_lr)
    checkpoint_file = checkpoint_file + '_s' + str(args.seed)
    checkpoint_file = os.path.join(checkpoint_path ,checkpoint_file)

    log_path = os.path.join(output_path, "runs")

    seeding(args.seed)

    create_dir(checkpoint_path)
    create_dir(log_path)

    train_x = sorted(glob(os.path.join(os.getcwd(), args.train_path, 'images/*')))
    train_y = sorted(glob(os.path.join(os.getcwd(), args.train_path, 'masks/*')))
    valid_x = sorted(glob(os.path.join(os.getcwd(), args.val_path, 'images/*')))
    valid_y = sorted(glob(os.path.join(os.getcwd(), args.val_path, 'masks/*')))

    logger.info(f"train image size: {len(train_x)}, train mask size: {len(train_y)}")    
    logger.info(f"valid image size: {len(valid_x)}, valid mask size: {len(valid_y)}")    

    check_data_empty(train_x, train_y, 'training')
    check_data_empty(valid_x, valid_y, 'validation')

    train_dataset = DriveDataset(train_x, train_y, args.dilation_pixels, args.img_size)
    valid_dataset = DriveDataset(valid_x, valid_y, args.dilation_pixels, args.img_size)

    print("length of x_train=", len(train_x))
    print("shape of train_dataset image=", train_dataset[0][0].shape)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Using device: {device}")

    model = UNet()

    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)
    summary(model, (1, args.img_size, args.img_size))

    if args.wandb:
        wandb.init(
            project=args.exp,
            config={
                "learning_rate": args.base_lr,
                "batch_size": args.batch_size,
                "architecture": "UNet",
                "dataset": args.dataset,
                "epochs": args.max_epochs,
            }
        )

    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    loss_fn = DiceBCELoss()

    best_valid_loss = float('inf')
    runs_file = os.path.join(log_path, snapshot_path.split('/')[-1] + '.txt')
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=checkpoint_file)

    with open(runs_file, "a") as f:
        f.write(f'--train_path "{args.train_path}" --val_path "{args.val_path}" --output "{args.output}" --dataset "{args.dataset}" --max_epochs {args.max_epochs} --batch_size {args.batch_size} --base_lr {args.base_lr} --patience {args.patience} --img_size {args.img_size} --seed {args.seed} --ckpt "{args.ckpt}" \n')
        f.write(f'\ncheckpoint_file path: {checkpoint_file}\n')
        f.write(f'runs_file path: {runs_file}\n\n')

        for epoch in range(args.max_epochs):
            start_time = time.time()

            train_loss = train(model, train_loader, optimizer, loss_fn, device)
            valid_loss = evaluate(model, valid_loader, loss_fn, device)

            scheduler.step(valid_loss)
            after_lr = optimizer.param_groups[0]["lr"]

            early_stopping(valid_loss, model, optimizer=optimizer, epoch=epoch)
            
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f'Epoch: {epoch+1:02}/{args.max_epochs} | epoch time: {epoch_mins}m {epoch_secs:04}s | lr: {after_lr} | train/loss: {train_loss:.5f} | val/loss: {valid_loss:.5f}')
            f.write(f'Epoch: {epoch+1:02}/{args.max_epochs} | epoch time: {epoch_mins}m {epoch_secs:04}s | lr: {after_lr} | train/loss: {train_loss:.5f} | val/loss: {valid_loss:.5f}\n')

            if args.wandb:
                wandb.log({"train/loss": train_loss, "val/loss": valid_loss})

            if early_stopping.early_stop:
                logger.warning(f"Early stopping triggered at epoch {epoch+1}. Ending model training.")
                break
