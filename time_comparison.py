#!/usr/bin/env python3
"""
manipulate.py - Image manipulation using diffusion autoencoder
Converted from manipulate.ipynb
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from torchvision.utils import save_image

from templates import *
from templates_cls import *
from experiment_classifier import ClsModel

def main():
    # Setup device
    device = 'cuda:0'
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load autoencoder model
    conf = ffhq128_autoenc_130M()
    model = LitModel(conf)
    state = torch.load(f'checkpoints/{conf.name}/last_ori.ckpt', map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)
    
    
    # Load data
    data = ImageDataset('imgs_interpolate', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
    # 273.png
    batch = data[4]['img'][None]
    
    # Encode image
    cond = model.encode(batch.to(device))
    xT = model.encode_stochastic(batch.to(device), cond, T=250)
    
    conf = ffhq128_ddpm_130M()
    model = LitModel(conf)
    state = torch.load(f'checkpoints/{conf.name}/last_ori.ckpt', map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)
    
    # 创建一个无条件的采样器
    sampler = conf._make_diffusion_conf(T=250).make_sampler()

    # 直接使用采样器的逆过程（如果支持）
    # 这通常需要修改采样器代码来支持无条件逆过程
    xT = sampler.ddim_reverse_sample_loop(model.ema_model, batch.to(device), model_kwargs={})['sample']

    t_list = torch.arange(2, 11)
    for t in t_list:
        x_t = model.render(xT, cond=None, T=t)
        # x_t_cond = model.render(xT, cond=cond, T=t)
        save_image(x_t[0], f'results/time_comparison/no_cond/xT_{t}.png')
        # save_image(x_t_cond[0], f'results/time_comparison/cond/xT_cond_{t}.png')
    # t = 100  # Example timestep
    # x_t = model.render(xT['sample'], cond=None, T=t)
    # save_image(x_t[0], f'results/time_comparison/no_cond/xT_{t}.png')
    # save_image(x_t_cond[0], f'results/time_comparison/cond/xT_cond_{t}.png')

    # save_image(img[0], 'results/manipulate/669/Wavy_Hair.png')
    # save_image(reconstructed[0], 'results/manipulate/21576/reconstructed.png')
    print("Saved manipulated image to imgs_manipulated/output.png")

if __name__ == "__main__":
    main()
