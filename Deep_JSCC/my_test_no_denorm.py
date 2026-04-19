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


def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    return -10 * math.log10(1 - ms_ssim(a, b, data_range=1.).item())


def main():
    parser = argparse.ArgumentParser(description="Train DeepJSCC")
    parser.add_argument("--mode", type=str, default='val', choices=['train', 'val'])
    parser.add_argument(
        "-d", "--datasets", default="/workspace/infonet/cg/WITT_TCM/datasets/data/mini_imagenet", type=str,
        help="Training dataset"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    args = parser.parse_args()
    test_loader = get_dataloader(args)
    snr = [1, 4, 7, 10, 13]
    c = ratio2filtersize(torch.randn(3, 3, 256, 256), 1 / 6)
    model = DeepJSCC(c=c, channel_type="awgn", snr=snr).cuda()
    ckpt = torch.load("/workspace/infonet/cg/WITT_TCM/Deep_JSCC/out/checkpoints/no_denorm/mini_imagenet_epoch_15.pth",
                      map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    model = model.cuda()
    psnr_list = []
    msssim_list = []
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Test") as t:
            for i, imgs in enumerate(test_loader):
                images = imgs.cuda()
                outputs = model(images)
                # outputs = image_normalization('normalization')(outputs)
                # images = image_normalization('normalization')(images)
                # loss = model.loss(outputs, images)
                plt.imshow(outputs.permute(1, 2, 0).cpu().numpy())
                plt.title("output")
                plt.axis('off')
                # plt.show()
                plt.savefig(f"output.png")
                input = images[0].permute(1, 2, 0).cpu().numpy()
                plt.imshow(input)
                plt.title("input")
                plt.axis('off')
                # plt.show()
                plt.savefig("input.png")
                psnr = get_psnr(outputs, images)
                msssim = compute_msssim(outputs, images)
                psnr_list.append(psnr)
                msssim_list.append(msssim)
                t.set_postfix(progress=f"{i}/{len(test_loader)}", PSNR=f"{psnr:.2f}", MSSSIM=f"{msssim:.2f}")
                t.update(1)

    print(f"PSNR: {np.mean(psnr_list):.3f}, MSSSIM: {np.mean(msssim_list):.3f}")


main()