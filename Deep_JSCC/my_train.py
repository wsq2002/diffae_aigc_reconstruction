import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append("/home/cg/semcom/TCC")
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
from pytorch_msssim import ms_ssim
import math
import torch.nn.functional as F

def compute_msssim(a, b):
    return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

def compute_psnr(a, b):
    mse = F.mse_loss(a * 255., b * 255.)
    psnr = 10 * (torch.log10(255. * 255. / mse))
    return psnr


def evaluate_epoch(model, data_loader):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="Testing: ") as t:
            for iter, images in enumerate(data_loader):
                images = images.cuda() 
                outputs = model(images)
                outputs = image_normalization('denormalization')(outputs)
                images = image_normalization('denormalization')(images)
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
        "-d", "--datasets", default="DIV2K", type=str, choices=["mini_imagenet", "DIV2K"], help="Training dataset"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument(
    "-t", "--test-datasets", default="Kodak", type=str, choices=["Kodak", "CLIC2022"], help="Test dataset"
    )
    args = parser.parse_args()

    args.datasets = os.path.join("/home/cg/semcom/mar/data", args.datasets)
    train_loader = get_dataloader(args)
    args.mode = 'val'
    if args.test_datasets == "Kodak":
        args.datasets = os.path.join("/home/cg/semcom/mar/data/DIV2K/test", "Kodak")
    elif args.test_datasets == "CLIC2022":
        args.datasets = os.path.join("/home/cg/semcom/mar/data/DIV2K/test", "CLIC2022")
    test_loader = get_dataloader(args)
    args.mode = 'train'

    if args.mode == 'train':
        snr = [1, 4, 7, 10, 13]
        c = ratio2filtersize(torch.randn(3, 3, 256, 256), 1/6)
        model = DeepJSCC(c=c, channel_type="awgn", snr=snr).cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=False)
        epoches = 5000
        writer = SummaryWriter(log_dir='/home/cg/semcom/TCC/Deep_JSCC/out/logs/DIV2K')
        # ckpt = torch.load("/home/cg/semcom/TCC/Deep_JSCC/out/checkpoints/DIV2K/epoch_299.pth", map_location='cpu')
        # model.load_state_dict(ckpt)
        model.train()
        best_loss = 300.
        avg_loss_list = []
        avg_psnr_list, avg_msssim_list = [], []
        for epoch in range(epoches):
            # epoch_loss = 0
            loss_list = []
            psnr_list = []
            msssim_list = []
            with tqdm(range(len(train_loader)), desc=f"Epoch {epoch}") as t:
                for i, imgs in enumerate(train_loader):
                    imgs = imgs.cuda()
                    optimizer.zero_grad()
                    outputs = model(imgs)
                    loss = model.loss(imgs, outputs)  # tensor
                    msssim = compute_msssim(imgs, outputs)  # float
                    outputs = image_normalization('denormalization')(outputs)
                    images = image_normalization('denormalization')(imgs)
                    mse = model.loss(images, outputs)  # MSE loss, tensor
                    psnr = 10 * (torch.log10(255. * 255. / mse))  # tensor
                    psnr_list.append(psnr.item())
                    
                    
                    msssim_list.append(msssim)
                    loss_list.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    t.set_postfix(loss=f"{loss.item():.3f}", MSE_loss=f"{mse.item():.3f}", PSNR=f"{psnr.item():.3f}", MS_SSIM=f"{msssim:.3f}")
                    t.update(1)
                # epoch_loss /= (i + 1)
                avg_loss = np.mean(loss_list)
                avg_loss_list.append(avg_loss)
                avg_psnr = np.mean(psnr_list)
                avg_psnr_list.append(avg_psnr)
                avg_msssim = np.mean(msssim_list)
                avg_msssim_list.append(avg_msssim)
                print(f"Epoch {epoch+1}/{epoches}: Train Loss: {avg_loss:.4f}, Avg PSNR: {avg_psnr:.4f}, Avg MS_SSIM: {avg_msssim:.4f}")
                writer.add_scalar('train_loss', np.mean(loss_list), epoch)
                # epoch_val_loss = evaluate_epoch(model, test_loader)
                # writer.add_scalar('test_loss', epoch_val_loss, epoch)
                # scheduler.step(epoch_val_loss)
                # if epoch_val_loss < best_loss:
                #     torch.save(model.state_dict(), "out/checkpoints/mini_imagenet_best.pth")
                #     best_loss = epoch_val_loss
            ckpt = {"epoch": epoch,
                    "state_dict": model.state_dict(), 
                        "optimizer": optimizer.state_dict(),
                        "avg_loss_list": avg_loss_list,
                        "avg_psnr_list": avg_psnr_list,
                        "avg_msssim_list": avg_msssim_list
                        }
            torch.save(ckpt, f"/home/cg/semcom/TCC/Deep_JSCC/out/checkpoints/DIV2K/last.pth")
            if epoch % 10 == 0:
                torch.save(ckpt, f"/home/cg/semcom/TCC/Deep_JSCC/out/checkpoints/DIV2K/{epoch}.pth")

    else:   # val
        snr = [1]   
        c = ratio2filtersize(torch.randn(3, 3, 256, 256), 1/6)
        model = DeepJSCC(c=c, channel_type="awgn", snr=snr).cuda()
        ckpt = to
        model.eval()
        with torch.no_grad():
            for i, imgs in enumerate(test_loader): 
                imgs = imgs.cuda()
                outputs = model(imgs)
                loss = model.loss(imgs, outputs)  # tensor
                msssim = compute_msssim(imgs, outputs)  # float
                outputs = image_normalization('denormalization')(outputs)
                images = image_normalization('denormalization')(imgs)
                mse = model.loss(images, outputs)  # MSE loss, tensor
                psnr = 10 * (torch.log10(255. * 255. / mse))  # tensor
                psnr_list.append(psnr.item())
                
                
                msssim_list.append(msssim)
                loss_list.append(loss.item())
                t.set_postfix(MSE_loss=f"{mse.item():.3f}", PSNR=f"{psnr.item():.3f}", MS_SSIM=f"{msssim:.3f}")
                t.update(1)
            print(f"SNR {snr[0]}: Avg PSNR: {avg_psnr:.4f}, Avg MS_SSIM: {avg_msssim:.4f}")

if __name__ == "__main__":
    main()