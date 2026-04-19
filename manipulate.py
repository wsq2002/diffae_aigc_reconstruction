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
    
    # Load classifier model
    cls_conf = ffhq128_autoenc_cls()
    cls_model = ClsModel(cls_conf)
    state = torch.load(f'checkpoints/{cls_conf.name}/last.ckpt', map_location='cpu')
    print('latent step:', state['global_step'])
    cls_model.load_state_dict(state['state_dict'], strict=False)
    cls_model.to(device)
    
    # Load data
    data = ImageDataset('imgs_interpolate', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
    # 21576, 869, 5, sandy
    batch = data[2]['img'][None]
    
    # Encode image
    cond = model.encode(batch.to(device))
    xT = model.encode_stochastic(batch.to(device), cond, T=250)
    
    # Display original and encoded images
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ori = (batch + 1) / 2
    # ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
    # ax[1].imshow(xT[0].permute(1, 2, 0).cpu())
    # plt.title('Original vs Encoded')
    # plt.show()
    
    # Print available attributes
    print("Available attributes:")
    print(CelebAttrDataset.id_to_cls)
    
    # Select attribute to manipulate (Wavy_Hair)
    # cls_id = CelebAttrDataset.cls_to_id['Wavy_Hair']
    cls_id = CelebAttrDataset.cls_to_id['Wavy_Hair']  # Example: change to 'Eyeglasses', Wearing_Hat, Wearing_Earrings Wearing_Necklace

    # Manipulate latent code
    cond2 = cls_model.normalize(cond)
    cond2 = cond2 + 0.5 * math.sqrt(512) * F.normalize(cls_model.classifier.weight[cls_id][None, :], dim=1)
    cond2 = cls_model.denormalize(cond2)
    
    # Generate manipulated image
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    xT_sample = torch.randn_like(xT)
    reconstructed = model.render(xT_sample, cond, T=100)
    img = model.render(xT_sample, cond2, T=100)
    ori = (batch + 1) / 2
    ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
    ax[1].imshow(img[0].permute(1, 2, 0).cpu())
    ax[0].set_title('Original')
    ax[1].set_title('Manipulated (Wavy Hair)')
    # plt.savefig('imgs_manipulated/compare.png')
    plt.show()
    
    # Save output image
    
    save_image(img[0], 'results/manipulate/669/Wavy_Hair.png')
    # save_image(reconstructed[0], 'results/manipulate/21576/reconstructed.png')
    print("Saved manipulated image to imgs_manipulated/output.png")

if __name__ == "__main__":
    main()
