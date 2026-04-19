import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
sys.path.append("/workspace/infonet/cg/WITT_TCM")
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from model import DeepJSCC, ratio2filtersize
from torch.nn.parallel import DataParallel
from utils import image_normalization, set_seed, save_model, view_model_param
from fractions import Fraction
from dataset import Vanilla
import numpy as np
import time
from tensorboardX import SummaryWriter
import glob
from data_utils import get_dataloader
import argparse


def evaluate_epoch(model, data_loader):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="Testing: ") as t:
            for iter, images in enumerate(data_loader):
                images = images.cuda() 
                outputs = model(images)
                # outputs = image_normalization('denormalization')(outputs)
                # images = image_normalization('denormalization')(images)
                loss = model.loss(images, outputs)
                epoch_loss += loss.detach().item()
                t.set_postfix(loss=f"{loss.item():.3f}")
                t.update(1)
            epoch_loss /= (iter + 1)

    return epoch_loss

def main():
    parser = argparse.ArgumentParser(description="Train DeepJSCC")
    parser.add_argument("--mode", type=str, default='train', choices=['train', 'val'])
    parser.add_argument(
        "-d", "--datasets", default="/workspace/infonet/cg/WITT_TCM/datasets/data/mini_imagenet", type=str, help="Training dataset"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    args = parser.parse_args()
    train_loader = get_dataloader(args)
    args.mode = 'val'
    test_loader = get_dataloader(args)
    args.mode = 'train'
    snr = [1, 4, 7, 10, 13]
    c = ratio2filtersize(torch.randn(3, 3, 256, 256), 1/6)
    model = DeepJSCC(c=c, channel_type="awgn", snr=snr).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=False)
    epoches = 100
    writer = SummaryWriter(log_dir='out/logs/mini_imagenet/no_denorm')
    ckpt = torch.load("/workspace/infonet/cg/WITT_TCM/Deep_JSCC/out/checkpoints/no_denorm/mini_imagenet_epoch_15.pth", map_location='cpu')
    model.load_state_dict(ckpt)
    best_loss = 1e5
    model.train()
    for epoch in range(16, epoches):
        epoch_loss = 0
        with tqdm(range(len(train_loader)), desc=f"Epoch {epoch}") as t:
            for i, imgs in enumerate(train_loader):
                imgs = imgs.cuda()
                optimizer.zero_grad()
                outputs = model(imgs)
                # outputs = image_normalization('denormalization')(outputs)
                # images = image_normalization('denormalization')(imgs)
                loss = model.loss(imgs, outputs)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
                t.set_postfix(progress=f"{i}/{len(train_loader)}", loss=f"{loss.item():.3f}")
                t.update(1)
            epoch_loss /= (i + 1)
            
            print(f"Epoch {epoch+1}/{epoches}: Train Loss: {epoch_loss:.4f}")
            writer.add_scalar('train_loss', epoch_loss, epoch)
            epoch_val_loss = evaluate_epoch(model, test_loader)
            writer.add_scalar('test_loss', epoch_val_loss, epoch)
            scheduler.step(epoch_val_loss)
            if epoch_val_loss < best_loss:
                torch.save(model.state_dict(), "out/checkpoints/no_denorm/mini_imagenet_best.pth")
                best_loss = epoch_val_loss
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"out/checkpoints/no_denorm/mini_imagenet_epoch_{epoch}.pth")

main()