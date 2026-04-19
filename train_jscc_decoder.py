import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys
import torch.nn as nn

# Add the project root to the Python path to allow for local imports
sys.path.append('/home/cg/semcom/diffae')

from Deep_JSCC.model import DeepJSCC, ratio2filtersize
from Deep_JSCC.utils import image_normalization
from ffhq_dataset import FFHQlmdb
from model.unet import BeatGANsUNetConfig, BeatGANsUNetModel
from distortion import MS_SSIM
from train_denoising_diffusion import Channel

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

def main():
    parser = argparse.ArgumentParser(description="Train DeepJSCC Decoder with Diffusion Denoising")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--data-path", type=str, default="/home/cg/semcom/diffae/datasets/ffhq256.lmdb", help="Path to FFHQ dataset")
    parser.add_argument("--deepjscc-ckpt", type=str, default="/home/cg/semcom/diffae/Deep_JSCC/out/checkpoints/FFHQ/last.pth", help="Path to DeepJSCC checkpoint")
    parser.add_argument("--diffusion-ckpt", type=str, default="/home/cg/semcom/diffae/checkpoints/baseline/DM/best.pth", help="Path to diffusion model checkpoint")
    parser.add_argument("--save-dir", type=str, default="/home/cg/semcom/diffae/checkpoints/baseline/DM/JSCC_decoder", help="Directory to save checkpoints")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Setup dataset
    dataset = FFHQlmdb(args.data_path, 128)
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # Load and freeze Deep JSCC model
    c = ratio2filtersize(torch.randn(1, 3, 128, 128), 1/6)
    deep_jscc_model = DeepJSCC(c=c, channel_type="awgn", snr=[1]).cuda()
    print(f"Loading Deep JSCC checkpoint from {args.deepjscc_ckpt}...")
    ckpt = torch.load(args.deepjscc_ckpt, map_location='cpu')
    deep_jscc_model.load_state_dict(ckpt["state_dict"])
    
    # Freeze encoder weights
    for param in deep_jscc_model.encoder.parameters():
        param.requires_grad = False
    deep_jscc_model.encoder.eval()
    
    # Load and freeze diffusion model
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
    print(f"Loading diffusion model checkpoint from {args.diffusion_ckpt}...")
    diffusion_ckpt = torch.load(args.diffusion_ckpt, map_location='cpu')
    diffusion_model.load_state_dict(diffusion_ckpt)
    diffusion_model.eval()
    for param in diffusion_model.parameters():
        param.requires_grad = False
    
    print("Deep JSCC encoder and diffusion model loaded and frozen.")
    
    # Only train the decoder
    decoder = deep_jscc_model.decoder
    decoder.train()
    
    # Setup channel and SNR values
    channel = Channel('awgn')
    snrs = [5, 10, 15, 20]
    T = 1000  # Total timesteps for diffusion model reference
    
    # Setup optimizer for decoder only
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=5e-4)
    
    # MS-SSIM metric
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()
    
    best_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        loss_list = []
        psnr_list = []
        msssim_list = []
        
        for batch in pbar:
            images = batch['img'].cuda()
            
            with torch.no_grad():
                # 1. Get latent vector from frozen encoder
                z = deep_jscc_model.encoder(images)
                
                # 2. Pick a random SNR and add noise
                snr = np.random.choice(snrs)
                noisy_z, noise = channel.forward(z, snr, return_noise=True)
                
                # 3. Map SNR to timestep for diffusion model
                t = (1 - (snr - 5) / (20 - 5)) * (T - 1)
                t_tensor = torch.full((images.size(0),), int(t), device='cuda', dtype=torch.long)
                
                # 4. Denoise using frozen diffusion model
                predicted_noise = diffusion_model(noisy_z, t_tensor)
                if hasattr(predicted_noise, 'pred'):
                    predicted_noise = predicted_noise.pred
                denoised_z = noisy_z - predicted_noise
            
            optimizer.zero_grad()
            
            # 5. Decode using trainable decoder
            outputs = decoder(denoised_z)
            
            # 6. Calculate losses (similar to my_train_ffhq.py)
            # Convert images from [-1,1] to [0,1]
            target_imgs = (images + 1) / 2.
            
            # Main loss
            loss = deep_jscc_model.loss(target_imgs, outputs)
            
            # Calculate MS-SSIM
            msssim = 1 - CalcuSSIM(target_imgs, outputs).mean().item()
            
            # Calculate PSNR
            outputs_denorm = image_normalization('denormalization')(outputs)
            images_denorm = image_normalization('denormalization')(target_imgs)
            mse = deep_jscc_model.loss(images_denorm, outputs_denorm)
            psnr = 10 * (torch.log10(255. * 255. / mse))
            
            # Record metrics
            psnr_list.append(psnr.item())
            msssim_list.append(msssim)
            loss_list.append(loss.item())
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.3f}", 
                MSE_loss=f"{mse.item():.3f}", 
                PSNR=f"{psnr.item():.3f}", 
                MS_SSIM=f"{msssim:.3f}",
                SNR=snr
            )
        
        # Calculate average metrics
        avg_loss = np.mean(loss_list)
        avg_psnr = np.mean(psnr_list)
        avg_msssim = np.mean(msssim_list)
        
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss: {avg_loss:.4f}, Avg PSNR: {avg_psnr:.4f}, Avg MS_SSIM: {avg_msssim:.4f}")
        
        # Save checkpoints
        ckpt = {
            "epoch": epoch,
            "state_dict": deep_jscc_model.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "avg_loss": avg_loss,
            "avg_psnr": avg_psnr,
            "avg_msssim": avg_msssim
        }
        
        # Save last checkpoint
        torch.save(ckpt, os.path.join(args.save_dir, 'last.pth'))
        
        # Save checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            torch.save(ckpt, os.path.join(args.save_dir, f'ckpt_{epoch+1}.pth'))
            print(f"Saved checkpoint at epoch {epoch+1}.")
        
        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, os.path.join(args.save_dir, 'best.pth'))
            print(f"Saved new best model with loss {best_loss:.4f}")

if __name__ == "__main__":
    main()
