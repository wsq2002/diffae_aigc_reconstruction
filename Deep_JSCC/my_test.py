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
from utils import image_normalization, set_seed, save_model, view_model_param, get_psnr
from fractions import Fraction
from dataset import Vanilla
import numpy as np
import time
from tensorboardX import SummaryWriter
import glob
from data_utils import get_dataloader
import argparse
import math
from pytorch_msssim import ms_ssim
from matplotlib import pyplot as plt
from PIL import Image

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    if len(a.shape) == 3:
        a = a.unsqueeze(0)
    if len(b.shape) == 3:
        b = b.unsqueeze(0)
    return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())


def main():
    parser = argparse.ArgumentParser(description="Train DeepJSCC")
    parser.add_argument("--mode", type=str, default='val', choices=['train', 'val'])
    parser.add_argument(
        "-d", "--datasets", default="DIV2K", type=str, choices=["mini_imagenet", "DIV2K"], help="Training dataset"
    )
    parser.add_argument(
    "-t", "--test-datasets", default="CLIC2022", type=str, choices=["Kodak", "CLIC2022"], help="Test dataset"
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

    """
    torch.random.manual_seed(1)
    np.random.seed(0)
    
    args = parser.parse_args()
    args.datasets = os.path.join("/home/cg/semcom/mar/data", args.datasets)
    if args.test_datasets == "Kodak":
        args.datasets = os.path.join("/home/cg/semcom/mar/data/DIV2K/test", "Kodak")
    elif args.test_datasets == "CLIC2022":
        args.datasets = os.path.join("/home/cg/semcom/mar/data/DIV2K/test", "CLIC2022")
    test_loader = get_dataloader(args)
    snr = [13]
    c = ratio2filtersize(torch.randn(3, 3, 256, 256), 1/6)
    model = DeepJSCC(c=c, channel_type="awgn", snr=snr).cuda()
    ckpt = torch.load("Deep_JSCC/out/checkpoints/DIV2K/790.pth", map_location='cpu')["state_dict"]
    model.load_state_dict(ckpt)
    model.eval()
    psnr_list = []
    msssim_list = []
    
    # mini-imagenet
    
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Test") as t:
            for i, imgs in enumerate(test_loader):
                imgs = imgs.cuda()
                # transform = transforms.Compose([transforms.ToTensor()])
                # imgs = transform(Image.open("/workspace/infonet/cg/WITT_TCM/jpeg_ldpc/img_1.png")).cuda()
                outputs = model(imgs)
                outputs = image_normalization('denormalization')(outputs)
                images = image_normalization('denormalization')(imgs)
                loss = model.loss(outputs, images)
                # plt.imshow(outputs.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                # plt.title("output")
                # plt.axis('off')
                # plt.show()
                # input = images[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                # plt.imshow(input)
                # plt.title("input")
                # plt.axis('off')
                # plt.show()
                # img_sample = imgs.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                # output_sample = outputs.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                # Image.fromarray(output_sample).save(f"/workspace/infonet/cg/WITT_TCM/visualizations/JSCC/img_13.png")
                psnr = get_psnr(outputs, images).item()
                msssim = compute_msssim(outputs, images)
                # print(f"PSNR: {psnr:.3f}, MSSSIM: {msssim:.3f}")
                psnr_list.append(psnr)
                msssim_list.append(msssim)
                t.set_postfix(progress=f"{i}/{len(test_loader)}", PSNR=f"{psnr:.2f}", MSSSIM=f"{msssim:.2f}")
                t.update(1)
                
    print(f"PSNR: {np.mean(psnr_list):.3f}, MSSSIM: {np.mean(msssim_list):.3f}")   
"""  

    torch.random.manual_seed(1)
    np.random.seed(0)
    image_path = "jpeg_ldpc/kodim01_cropped.png"
    snr = [1]
    c = ratio2filtersize(torch.randn(3, 3, 256, 256), 1/6)
    model = DeepJSCC(c=c, channel_type="awgn", snr=snr).cuda()
    ckpt = torch.load("Deep_JSCC/out/checkpoints/DIV2K/790.pth", map_location='cpu')["state_dict"]
    model.load_state_dict(ckpt)
    model.eval()
    with torch.no_grad():
        psnr_list = []
    msssim_list = []
    with torch.no_grad():
        # imgs = imgs.cuda()
        transform = transforms.Compose([transforms.ToTensor()])
        imgs = transform(Image.open(image_path)).cuda()
        outputs = model(imgs.unsqueeze(0))
        outputs = image_normalization('denormalization')(outputs)
        images = image_normalization('denormalization')(imgs)
        loss = model.loss(outputs, images)
        # plt.imshow(outputs.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        # plt.title("output")
        # plt.axis('off')
        # plt.show()
        # input = images[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        # plt.imshow(input)
        # plt.title("input")
        # plt.axis('off')
        # plt.show()
        # img_sample = imgs.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        output_sample = outputs.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        Image.fromarray(output_sample).save(f"visualizations/JSCC/kodim01_{snr[0]}.png")
        # psnr = get_psnr(outputs, images).item()
        psnr = get_psnr(outputs, images).item()
        msssim = compute_msssim(outputs, images)
        # print(f"PSNR: {psnr:.3f}, MSSSIM: {msssim:.3f}")
        psnr_list.append(psnr)
        msssim_list.append(msssim)   
    print(f"PSNR: {np.mean(psnr_list):.3f}, MSSSIM: {np.mean(msssim_list):.3f}")   


if __name__ == '__main__':
    main()  