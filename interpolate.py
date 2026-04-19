#!/usr/bin/env python3
"""
interpolate.py - Image interpolation using diffusion autoencoder
Converted from interpolate.ipynb
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from templates import *

def slerp(a, b, alpha):
    """
    Spherical linear interpolation between two vectors.
    
    Args:
        a, b: Input tensors to interpolate between
        alpha: Interpolation factor (0 to 1)
    
    Returns:
        Interpolated tensor
    """
    def cos(a, b):
        a = a.view(-1)
        b = b.view(-1)
        a = F.normalize(a, dim=0)
        b = F.normalize(b, dim=0)
        return (a * b).sum()

    theta = torch.arccos(cos(a, b))
    x_shape = a.shape
    
    intp_x = (torch.sin((1 - alpha[:, None]) * theta) * a.flatten(0, 2)[None] + 
              torch.sin(alpha[:, None] * theta) * b.flatten(0, 2)[None]) / torch.sin(theta)
    intp_x = intp_x.view(-1, *x_shape)
    
    return intp_x

def main():
    # Setup device
    device = 'cuda:0'
    
    # Load model
    conf = ffhq128_autoenc_130M()
    model = LitModel(conf)
    state = torch.load(f'checkpoints/{conf.name}/last_ori.ckpt', map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)
    
    # Load data - two images for interpolation
    data = ImageDataset('imgs_interpolate', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
    batch = torch.stack([
        data[0]['img'],   # 327
        data[2]['img'],
    ])
    
    # Display first image
    plt.figure(figsize=(5, 5))
    plt.imshow(batch[0].permute([1, 2, 0]) / 2 + 0.5)
    plt.title('First Image')
    plt.axis('off')
    plt.show()
    
    # Encode images
    cond = model.encode(batch.to(device))
    xT = model.encode_stochastic(batch.to(device), cond, T=250)
    
    # Display original and encoded versions
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ori = (batch + 1) / 2
    ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
    ax[1].imshow(xT[0].permute(1, 2, 0).cpu())
    ax[0].set_title('Original')
    ax[1].set_title('Encoded')
    plt.show()
    
    print("Performing interpolation...")
    print("Semantic codes are interpolated using convex combination,")
    print("while stochastic codes are interpolated using spherical linear interpolation.")
    
    # Create interpolation steps
    alpha = torch.tensor(np.linspace(0, 1, 5, dtype=np.float32)).to(cond.device)
    
    # Interpolate semantic codes (convex combination)
    intp = cond[0][None] * (1 - alpha[:, None]) + cond[1][None] * alpha[:, None]
    
    # Interpolate stochastic codes (spherical linear interpolation)
    intp_x = slerp(xT[0], xT[1], alpha)
    
    # Generate interpolated images
    pred = model.render(intp_x, intp, T=20)
    # ret = pred[0].numpy().permute(1, 2, 0).cpu() * 255
    # Image.fromarray(ret.astype(np.uint8)).save('results/interpolate/interpolated_image.png')
    for i in range(len(alpha)):
        ret = pred[i].cpu().permute(1, 2, 0).numpy() * 255.
        Image.fromarray(ret.astype(np.uint8)).save(f'results/interpolate/669/image_{i}.png')
    # # Display interpolation results
    # fig, ax = plt.subplots(1, 10, figsize=(50, 5))
    # for i in range(len(alpha)):
    #     ax[i].imshow(pred[i].permute(1, 2, 0).cpu())
    #     ax[i].set_title(f'α={alpha[i]:.1f}')
    #     ax[i].axis('off')
    
    # plt.suptitle('Image Interpolation Results')
    # plt.tight_layout()
    # plt.savefig('imgs_interpolate/interpolation_results_1.pdf', dpi=300, bbox_inches='tight')
    # # plt.show()
    
    # print("Saved interpolation results to imgs_interpolate/interpolation_results.png")

if __name__ == "__main__":
    main()
