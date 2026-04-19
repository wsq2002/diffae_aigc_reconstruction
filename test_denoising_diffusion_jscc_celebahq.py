import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from PIL import Image
import math

# Add the project root to the Python path
sys.path.append('/home/cg/semcom/diffae')

from Deep_JSCC.model import DeepJSCC, ratio2filtersize
from Deep_JSCC.utils import image_normalization
from ffhq_dataset import CelebaHQ
from model.unet import BeatGANsUNetConfig, BeatGANsUNetModel
from distortion import MS_SSIM
from pytorch_msssim import ms_ssim
from train_denoising_diffusion import Channel
from lpips import LPIPS

# class Channel:
#     """A simple AWGN channel model."""
#     def __init__(self, channel_type='awgn'):
#         if channel_type != 'awgn':
#             raise NotImplementedError("Only 'awgn' channel type is supported.")
#         self.channel_type = channel_type

#     def forward(self, z, snr, return_noise=False):
#         """
#         Adds AWGN noise to the input tensor.
#         Assumes input z is normalized to have power 1.
#         """
#         # Calculate noise standard deviation from SNR
#         sigma = (10**(-snr / 10.0))**0.5
#         noise = torch.randn_like(z) * sigma
#         if return_noise:
#             return z + noise, noise
#         return z + noise

def compute_psnr(a, b):
    """Compute PSNR between two tensors."""
    mse = torch.mean((a - b)**2).item()
    return 10 * math.log10(255 ** 2 / mse) if mse > 0 else float('inf')

def compute_msssim(a, b):
    """Compute MS-SSIM between two tensors."""
    if len(a.shape) == 3:
        a = a.unsqueeze(0)
    if len(b.shape) == 3:
        b = b.unsqueeze(0)
    return 1 - ms_ssim(a, b, data_range=1.).item()

def load_encoder_weights(encoder, ckpt_path):
    """Load only encoder weights from DeepJSCC checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt["state_dict"]
    
    # Extract encoder weights
    encoder_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('encoder.'):
            encoder_state_dict[key[8:]] = value  # Remove 'encoder.' prefix
    
    encoder.load_state_dict(encoder_state_dict)
    print(f"Loaded encoder weights from {ckpt_path}")

def load_decoder_weights(decoder, ckpt_path):
    """Load only decoder weights from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if "decoder_state_dict" in ckpt:
        decoder.load_state_dict(ckpt["decoder_state_dict"])
    else:
        # Extract decoder weights from full model state_dict
        state_dict = ckpt["state_dict"]
        decoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('decoder.'):
                decoder_state_dict[key[8:]] = value  # Remove 'decoder.' prefix
        decoder.load_state_dict(decoder_state_dict)
    print(f"Loaded decoder weights from {ckpt_path}")

def test_single_images(args, encoder, diffusion_model, decoder, channel):
    """Test on specified image paths."""
    # MS-SSIM metric
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()
    # LPIPS metric
    lpips_loss = LPIPS(net='vgg').cuda()
    
    psnr_list = []
    msssim_list = []
    lpips_list = []

    # Image preprocessing (same as in test_celebahq.py)
    transform = [
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    transform = transforms.Compose(transform)
    
    T = 1000  # Total timesteps for diffusion model reference
    
    print(f"Testing {len(args.image_paths)} images at SNR {args.snr}...")
    
    with torch.no_grad():
        for i, img_path in enumerate(args.image_paths):
            print(f"Processing image {i+1}/{len(args.image_paths)}: {img_path}")
            
            # Load and preprocess image
            img = transform(Image.open(img_path)).cuda().unsqueeze(0)
            
            # 1. Encode with DeepJSCC encoder
            z = encoder(img)
            
            # 2. Add noise through channel
            noisy_z = channel.forward(z, args.snr)
            
            # 3. Map SNR to timestep for diffusion model
            t = (1 - (args.snr - 5) / (20 - 5)) * (T - 1)
            t_tensor = torch.full((img.size(0),), int(t), device='cuda', dtype=torch.long)
            
            # 4. Denoise using diffusion model
            predicted_noise = diffusion_model(noisy_z.unsqueeze(0), t_tensor)
            if hasattr(predicted_noise, 'pred'):
                predicted_noise = predicted_noise.pred
            denoised_z = noisy_z - predicted_noise
            
            # 5. Decode with DeepJSCC decoder
            outputs = decoder(denoised_z)  # outputs是[0, 1]
            lpips_dist = lpips_loss(outputs * 2 - 1, img).mean().item()
            # 6. Calculate metrics
            # Convert input image from [-1,1] to [0,1]
            target_imgs = (img + 1) / 2.
            
            # Calculate MS-SSIM
            msssim = 1 - CalcuSSIM(target_imgs, outputs).mean().item()
            
            # Calculate PSNR
            outputs_denorm = image_normalization('denormalization')(outputs)
            images_denorm = image_normalization('denormalization')(target_imgs)
            mse = torch.mean((images_denorm - outputs_denorm) ** 2)
            psnr = 10 * (torch.log10(255. * 255. / mse)).item()
            pred = (outputs_denorm[0].permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
            Image.fromarray(pred).save(f"results/visualizations/denoising_diffusion/output_{args.snr}.png")
            psnr_list.append(psnr)
            msssim_list.append(msssim)
            lpips_list.append(lpips_dist)

            print(f"Image {i+1} - PSNR: {psnr:.3f}, MS-SSIM: {msssim:.3f}, LPIPS: {lpips_dist:.3f}")

    print(f"\nResults for SNR {args.snr}:")
    print(f"Average PSNR: {np.mean(psnr_list):.3f}")
    print(f"Average MS-SSIM: {np.mean(msssim_list):.3f}")
    print(f"Average LPIPS: {np.mean(lpips_list):.3f}")

def test_dataset(args, encoder, diffusion_model, decoder, channel):
    """Test on entire CelebA-HQ dataset."""
    # Setup dataset
    dataset = CelebaHQ("/home/cg/semcom/diff_auto/datasets/celebahq/CelebAMask-HQ/CelebA-HQ-img/", 128)
    test_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # MS-SSIM metric
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()
    # LPIPS metric
    lpips_loss = LPIPS(net='vgg').cuda()
    
    psnr_list = []
    msssim_list = []
    lpips_list = []
    
    T = 1000  # Total timesteps for diffusion model reference
    
    print(f"Testing on entire CelebA-HQ dataset at SNR {args.snr}...")
    
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Testing") as t:
            for i, imgs in enumerate(test_loader):
                imgs = imgs.cuda()
                
                # 1. Encode with DeepJSCC encoder
                z = encoder(imgs)
                
                # 2. Add noise through channel
                noisy_z = channel.forward(z, args.snr)
                
                # 3. Map SNR to timestep for diffusion model
                t_val = (1 - (args.snr - 5) / (20 - 5)) * (T - 1)
                t_tensor = torch.full((imgs.size(0),), int(t_val), device='cuda', dtype=torch.long)
                
                # 4. Denoise using diffusion model
                predicted_noise = diffusion_model(noisy_z, t_tensor)
                if hasattr(predicted_noise, 'pred'):
                    predicted_noise = predicted_noise.pred
                denoised_z = noisy_z - predicted_noise
                
                # 5. Decode with DeepJSCC decoder
                outputs = decoder(denoised_z)
                lpips_dist = lpips_loss(outputs, imgs).mean().item()
                # 6. Calculate metrics
                # Convert input images from [-1,1] to [0,1]
                target_imgs = (imgs + 1) / 2.
                
                # Calculate MS-SSIM
                msssim = 1 - CalcuSSIM(target_imgs, outputs).mean().item()
                
                # Calculate PSNR
                outputs_denorm = image_normalization('denormalization')(outputs)
                images_denorm = image_normalization('denormalization')(target_imgs)
                mse = torch.mean((images_denorm - outputs_denorm) ** 2)
                psnr = 10 * (torch.log10(255. * 255. / mse)).item()
                
                psnr_list.append(psnr)
                msssim_list.append(msssim)
                lpips_list.append(lpips_dist)

                t.set_postfix(PSNR=f"{psnr:.2f}", MSSSIM=f"{msssim:.3f}", LPIPS=f"{lpips_dist:.3f}")
                t.update(1)
    log_msssim = -10 * np.log10(1 - np.mean(msssim_list))
    print(f"\nResults for SNR {args.snr} on entire dataset:")
    print(f"Average PSNR: {np.mean(psnr_list):.3f}")
    print(f"Average MS-SSIM: {log_msssim:.3f}")
    print(f"Average LPIPS: {np.mean(lpips_list):.3f}")

def main():
    parser = argparse.ArgumentParser(description="Test Denoising Diffusion + JSCC on CelebA-HQ")
    parser.add_argument("--snr", type=int, choices=[5, 10, 15, 20], default=20, required=False, 
                       help="SNR value for testing")
    parser.add_argument("--test-dataset", action='store_true', default=False,
                       help="Test on entire dataset")
    parser.add_argument("--image-paths", nargs='+', type=str, default=["imgs_interpolate/21576.jpg"],
                       help="List of image paths to test (required if not testing dataset)")
    parser.add_argument("--batch-size", type=int, default=256, 
                       help="Batch size for dataset testing")
    parser.add_argument("--encoder-ckpt", type=str, 
                       default="/home/cg/semcom/diffae/Deep_JSCC/out/checkpoints/FFHQ/last.pth",
                       help="Path to DeepJSCC encoder checkpoint")
    parser.add_argument("--diffusion-ckpt", type=str,
                       default="/home/cg/semcom/diffae/checkpoints/baseline/DM/last.pth",
                       help="Path to diffusion model checkpoint")
    parser.add_argument("--decoder-ckpt", type=str,
                       default="/home/cg/semcom/diffae/checkpoints/baseline/DM/JSCC_decoder/last.pth",
                       help="Path to DeepJSCC decoder checkpoint")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.test_dataset and not args.image_paths:
        parser.error("Either --test-dataset or --image-paths must be specified")
    
    # Fix random seed for reproducibility
    torch.random.manual_seed(1)
    np.random.seed(0)
    
    # Setup models
    c = ratio2filtersize(torch.randn(1, 3, 128, 128), 1/6)
    
    # 1. Load DeepJSCC encoder
    deep_jscc_model = DeepJSCC(c=c, channel_type="awgn", snr=[1]).cuda()
    load_encoder_weights(deep_jscc_model.encoder, args.encoder_ckpt)
    deep_jscc_model.encoder.eval()
    
    # 2. Load diffusion model
    unet_config = BeatGANsUNetConfig(
        image_size=32,
        in_channels=2*c,
        model_channels=128,
        out_channels=2*c,
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        dropout=0.1,
        channel_mult=(1, 2, 3, 4),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        resblock_updown=True,
        use_new_attention_order=False,
    )
    diffusion_model = BeatGANsUNetModel(unet_config).cuda()
    diffusion_ckpt = torch.load(args.diffusion_ckpt, map_location='cpu')
    diffusion_model.load_state_dict(diffusion_ckpt)
    diffusion_model.eval()
    print(f"Loaded diffusion model from {args.diffusion_ckpt}")
    
    # 3. Load DeepJSCC decoder
    decoder_model = DeepJSCC(c=c, channel_type="awgn", snr=[1]).cuda()
    load_decoder_weights(decoder_model.decoder, args.decoder_ckpt)
    decoder_model.decoder.eval()
    
    # 4. Setup channel
    channel = Channel('awgn')
    
    print(f"Models loaded successfully. Testing at SNR = {args.snr}")
    
    # Run testing
    if args.test_dataset:
        test_dataset(args, deep_jscc_model.encoder, diffusion_model, 
                    decoder_model.decoder, channel)
    else:
        test_single_images(args, deep_jscc_model.encoder, diffusion_model, 
                          decoder_model.decoder, channel)

if __name__ == "__main__":
    main()
