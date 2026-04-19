import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# import sys
# sys.path.append("/home/cg/semcom/diffae/Deep_JSCC")
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
# from tensorboardX import SummaryWriter
import glob
# from data_utils import get_dataloader
from ffhq_dataset import FFHQlmdb
import argparse
from pytorch_msssim import ms_ssim
from distortion import MS_SSIM
import math
import torch.nn.functional as F

def compute_msssim(a, b):
    return -10 * math.log10(1-ms_ssim(a, b, data_range=1., levels=3).item())

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
    parser.add_argument("--mode", type=str, default='val', choices=['train', 'val'])
    parser.add_argument(
        "-d", "--datasets", default="ffhq", type=str, help="Training dataset"
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(128, 128),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument(
    "-t", "--test-datasets", default="celebahq", type=str, help="Test dataset"
    )
    parser.add_argument(
    "-c", "--continue-train", default=True, type=bool, help="Continue training from a checkpoint"
    )
    args = parser.parse_args()
    data_path = "/home/cg/semcom/diffae/datasets/ffhq256.lmdb"
    dataset = FFHQlmdb(data_path, 128)
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # MSSSIM metric
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()


    if args.mode == 'train':
        snr = [5, 10, 15, 20]
        c = ratio2filtersize(torch.randn(3, 3, 256, 256), 1/6)
        model = DeepJSCC(c=c, channel_type="awgn", snr=snr).cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=False)
        epoches = 5000
        # writer = SummaryWriter(log_dir='/home/cg/semcom/TCC/Deep_JSCC/out/logs/FFHQ')
        start_epoch = 0
        if args.continue_train:
            ckpt = torch.load("/home/cg/semcom/diffae/Deep_JSCC/out/checkpoints/FFHQ/last.pth", map_location='cpu')
            model.load_state_dict(ckpt["state_dict"])
            start_epoch = ckpt["epoch"] + 1
            optimizer.load_state_dict(ckpt["optimizer"])
            avg_loss_list = ckpt["avg_loss_list"]
            avg_psnr_list = ckpt["avg_psnr_list"]
            avg_msssim_list = ckpt["avg_msssim_list"]
        model.train()
        best_loss = 300.
        avg_loss_list = []
        avg_psnr_list, avg_msssim_list = [], []
        for epoch in range(start_epoch, epoches):
            # epoch_loss = 0
            loss_list = []
            psnr_list = []
            msssim_list = []
            with tqdm(range(len(train_loader)), desc=f"Epoch {epoch}") as t:
                for i, imgs in enumerate(train_loader):
                    imgs = imgs['img']
                    imgs = imgs.cuda()
                    optimizer.zero_grad()
                    outputs = model(imgs)
                    imgs = (imgs + 1) / 2.
                    loss = model.loss(imgs, outputs)  # tensor
                    # msssim = compute_msssim(imgs, outputs)  # float
                    
                    msssim = 1 - CalcuSSIM(imgs, outputs).mean().item()
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
                # writer.add_scalar('train_loss', np.mean(loss_list), epoch)
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
            torch.save(ckpt, f"/home/cg/semcom/diffae/Deep_JSCC/out/checkpoints/FFHQ/last.pth")
            if epoch % 100 == 0:
                torch.save(ckpt, f"/home/cg/semcom/diffae/Deep_JSCC/out/checkpoints/FFHQ/{epoch}.pth")

    else:   # val
        snr = [1]   
        c = ratio2filtersize(torch.randn(3, 3, 256, 256), 1/6)
        model = DeepJSCC(c=c, channel_type="awgn", snr=snr).cuda()
        ckpt = torch.load("/home/cg/semcom/diffae/Deep_JSCC/out/checkpoints/FFHQ/last.pth", map_location='cpu')
        model.load_state_dict(ckpt["state_dict"])
        test_dataset = FFHQlmdb("/home/cg/semcom/diffae/datasets/ffhq256.lmdb", 128)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        psnr_list = []
        msssim_list = []
        loss_list = []
        
        model.eval()
        with torch.no_grad():
            with tqdm(total=len(test_loader), desc="Testing: ") as t:
                for i, imgs in enumerate(test_loader): 
                    imgs = imgs['img'].cuda()
                    # 数据预处理：从[-1,1]转换到[0,1]

                    outputs = model(imgs)
                    imgs = (imgs + 1) / 2.
                    # 计算损失
                    loss = model.loss(imgs, outputs)  # tensor
                    
                    # 计算MS-SSIM
                    msssim = 1 - CalcuSSIM(imgs, outputs).mean().item()
                    
                    # 反归一化到[0,255]用于计算PSNR
                    outputs_denorm = image_normalization('denormalization')(outputs)
                    images_denorm = image_normalization('denormalization')(imgs)
                    
                    # 计算MSE和PSNR
                    mse = model.loss(images_denorm, outputs_denorm)  # MSE loss, tensor
                    psnr = 10 * (torch.log10(255. * 255. / mse))  # tensor
                    
                    # 记录结果
                    psnr_list.append(psnr.item())
                    msssim_list.append(msssim)
                    loss_list.append(loss.item())
                    
                    t.set_postfix(MSE_loss=f"{mse.item():.3f}", PSNR=f"{psnr.item():.3f}", MS_SSIM=f"{msssim:.3f}")
                    t.update(1)
            
            print(f"SNR {snr[0]}: Avg PSNR: {np.mean(psnr_list):.4f}, Avg MS_SSIM: {np.mean(msssim_list):.4f}, Avg Loss: {np.mean(loss_list):.4f}")

if __name__ == "__main__":
    main()