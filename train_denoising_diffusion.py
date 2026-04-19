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
from ffhq_dataset import FFHQlmdb
from model.unet import BeatGANsUNetConfig, BeatGANsUNetModel

class Channel:
    """A simple AWGN channel model."""
    def __init__(self, channel_type='awgn'):
        if channel_type != 'awgn':
            raise NotImplementedError("Only 'awgn' channel type is supported.")
        self.channel_type = channel_type

    def forward(self, z, snr, return_noise=False):
        """
        Adds AWGN noise to the input tensor.
        Assumes input z is normalized to have power 1.
        """
        # Calculate noise standard deviation from SNR
        sigma = (10**(-snr / 10.0))**0.5
        noise = torch.randn_like(z) * sigma
        if return_noise:
            return z + noise, noise
        return z + noise

def main():
    parser = argparse.ArgumentParser(description="Train Denoising Model in Latent Space")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--data-path", type=str, default="/home/cg/semcom/diffae/datasets/ffhq256.lmdb", help="Path to FFHQ dataset")
    parser.add_argument("--deepjscc-ckpt", type=str, default="/home/cg/semcom/diffae/Deep_JSCC/out/checkpoints/FFHQ/last.pth", help="Path to DeepJSCC checkpoint")
    parser.add_argument("--save-dir", type=str, default="/home/cg/semcom/diffae/checkpoints/baseline/DM", help="Directory to save checkpoints")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Setup dataset
    dataset = FFHQlmdb(args.data_path, 128)
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # Load and freeze Deep JSCC encoder
    # Calculate 'c' based on a dummy tensor, similar to my_train_ffhq.py
    c = ratio2filtersize(torch.randn(1, 3, 128, 128), 1/6)
    deep_jscc_model = DeepJSCC(c=c, channel_type="awgn", snr=[1]).cuda() # SNR is a placeholder
    print(f"Loading Deep JSCC checkpoint from {args.deepjscc_ckpt}...")
    ckpt = torch.load(args.deepjscc_ckpt, map_location='cpu')
    deep_jscc_model.load_state_dict(ckpt["state_dict"])
    deep_jscc_model.eval()
    for param in deep_jscc_model.parameters():
        param.requires_grad = False
    print("Deep JSCC model loaded and frozen.")
    
    jscc_encoder = deep_jscc_model.encoder
    channel = Channel('awgn')
    snrs = [5, 10, 15, 20]
    T = 1000 # Total timesteps for diffusion model reference

    # Setup Denoising Model (U-Net)
    # The U-Net input/output channels must match the JSCC encoder's output channels (2*c)
    # The U-Net input size must match the JSCC encoder's output spatial dimensions (32x32 for 128x128 input)
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
    denoise_model = BeatGANsUNetModel(unet_config).cuda()
    
    optimizer = optim.Adam(denoise_model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_loss = float('inf')

    print("Starting training...")
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        for batch in pbar:
            images = batch['img'].cuda()
            
            with torch.no_grad():
                # 1. Get latent vector from frozen encoder
                z = jscc_encoder(images)
                # 2. Pick a random SNR and add noise
                snr = np.random.choice(snrs)
                noisy_z, noise = channel.forward(z, snr, return_noise=True)

            optimizer.zero_grad()
            
            # Map SNR to a timestep 't' for the U-Net's time embedding
            # We map high SNR (low noise) to low t, and low SNR (high noise) to high t.
            # This is a simple linear mapping.
            t = (1 - (snr - 5) / (20 - 5)) * (T - 1)
            t_tensor = torch.full((images.size(0),), int(t), device='cuda', dtype=torch.long)
            
            # 3. Predict the noise using the U-Net
            predicted_noise = denoise_model(noisy_z, t_tensor)
            
            # 4. Calculate loss between the actual noise and the predicted noise
            loss = criterion(predicted_noise.pred, noise)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", snr=snr)

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # Save checkpoints
        state_dict = denoise_model.state_dict()
        torch.save(state_dict, os.path.join(args.save_dir, 'last.pth'))
        
        if (epoch + 1) % 50 == 0:
            torch.save(state_dict, os.path.join(args.save_dir, f'ckpt_{epoch+1}.pth'))
            print(f"Saved checkpoint at epoch {epoch+1}.")
            
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(state_dict, os.path.join(args.save_dir, 'best.pth'))
            print(f"Saved new best model with loss {best_loss:.4f}")

if __name__ == "__main__":
    main()
