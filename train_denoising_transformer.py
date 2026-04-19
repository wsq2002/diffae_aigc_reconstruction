import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment import LitModel
import torch
import torch.nn as nn
from model.latentnet import MLPSkipNetConfig, MLPSkipNet
from choices import *
from templates import *
from templates_latent import *
from tqdm import tqdm
import random
from diff_transformer import DiffTransformer




denosing_model = DiffTransformer(vocab_size=4, d_model=512, num_heads=8, num_layers=6, max_seq_length=512)
conf = ffhq128_autoenc_latent()
diffae = LitModel(conf)
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
diffae.load_state_dict(state['state_dict'], strict=False)
# diffae.ema_model.eval()
# diffae = diffae.cuda()
denosing_model = denosing_model.cuda()
# denosing_model = nn.DataParallel(denosing_model)
dataloader = diffae.train_dataloader()
print(f"latent shape: {diffae.conds.shape}\nmean: {diffae.conds_mean.mean()}, std: {diffae.conds_std.mean()}")
optimizer = torch.optim.Adam(denosing_model.parameters(), lr=1e-4)

SNR_list = list(range(5, 21, 5))
max_epochs = 3000

start_epoch = 0
# resume
# if os.path.exists(f'checkpoints/ffhq128_denoising_trans/last.ckpt'):
#     ckpt = torch.load(f'checkpoints/ffhq128_denoising_trans/last.ckpt', map_location='cpu')
#     denosing_model.load_state_dict(ckpt['state_dict'])
#     loss_list = ckpt['loss_list']
#     optimizer.load_state_dict(ckpt['optimizer'])
#     start_epoch = ckpt['epoch'] + 1

    # print(f"Resuming from epoch {start_epoch}")

avg_loss_list = []
for epoch in range(start_epoch, max_epochs):
    with tqdm(dataloader, desc=f"Epoch {epoch}") as pbar:
        avg_loss = 0
        loss_list = []
        for i, batch in enumerate(dataloader):
            latents = batch[0].cuda()
            latents_norm = diffae.normalize(latents)
            # add noise
            # snr = torch.tensor(random.choices(SNR_list, k=latents.shape[0])).cuda()
            snr = random.choices(SNR_list, k=latents.shape[0])
            snr_tensor = torch.tensor(snr).cuda()
            latents_power = latents_norm.pow(2).mean(dim=1)
            noise_power = latents_power / (10 ** (snr_tensor / 10))
            noise = torch.randn_like(latents_norm) * (torch.sqrt(noise_power).unsqueeze(1))
            noisy_latents = latents_norm + noise
            # denoise
            denoised_latents = denosing_model(noisy_latents, snr)
            # denoised_latents = diffae.denormalize(denoised_latents_norm)
            # noise_hat = denosing_model(noisy_latents, snr).pred
            # loss = (noise_hat - noise).abs().mean()
            loss = (denoised_latents - latents).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            avg_loss = sum(loss_list) / len(loss_list)
            pbar.set_postfix_str(f"loss={loss.item():.4f}, avg_loss={avg_loss:.4f}")
            pbar.update(1)
        avg_loss = sum(loss_list) / len(loss_list)
        print(f"Epoch {epoch}: Average loss={avg_loss:.4f}")
        avg_loss_list.append(avg_loss)
        ckpt = {'state_dict': denosing_model.state_dict(), "epoch": epoch,
                 "loss_list": avg_loss_list, "optimizer": optimizer.state_dict()}
        torch.save(ckpt, f'checkpoints/ffhq128_denoising_trans/last.ckpt')
        if epoch % 100 == 0:
            torch.save(ckpt, f'checkpoints/ffhq128_denoising_trans/{epoch}.ckpt')