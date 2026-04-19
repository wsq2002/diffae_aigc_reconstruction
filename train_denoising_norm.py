from experiment import LitModel
import torch
import torch.nn as nn
from model.latentnet import MLPSkipNetConfig, MLPSkipNet
from choices import *
from templates import *
from templates_latent import *
from tqdm import tqdm
import random

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = nn.SiLU()
        self.linear = nn.Linear(self.in_channels, self.out_channels)
        self.ln = nn.LayerNorm(out_channels)

    def forward(self, x):
        residual = x
        out = self.linear(x)
        out = self.ln(out)
        out = self.activation(out)
        out += residual
        return self.activation(out)


# denoising_model = MLPSkipNetConfig.make_model()
# print("Done")
# class DenoisingModel(nn.Module):
#     def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=4):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.hidden_channels = hidden_channels
#         self.num_layers = num_layers
#         self.activation = nn.SiLU()
#         self.

#     def 

config = MLPSkipNetConfig(
    num_channels=512,
    skip_layers=list(range(1, 10)),
    num_hid_channels=2048,
    num_layers=10,
    num_time_emb_channels=64,
    activation=Activation.silu,
    use_norm=True,
    condition_bias=1,
    dropout=0,
    last_act=Activation.none,
    num_time_layers=2,
    time_last_act=False
)


denosing_model = MLPSkipNet(config)
conf = ffhq128_autoenc_latent()
diffae = LitModel(conf)
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
diffae.load_state_dict(state['state_dict'], strict=False)
# diffae.ema_model.eval()
# diffae = diffae.cuda()
denosing_model = denosing_model.cuda()
denosing_model = nn.DataParallel(denosing_model)
dataloader = diffae.train_dataloader()
print(f"latent shape: {diffae.conds.shape}\nmean: {diffae.conds_mean.mean()}, std: {diffae.conds_std.mean()}")
optimizer = torch.optim.Adam(denosing_model.parameters(), lr=1e-4)

SNR_list = list(range(1, 14, 3))
max_epochs = 30000

start_epoch = 0
# resume
if os.path.exists(f'checkpoints/ffhq128_denoising/norm/last.ckpt'):
    ckpt = torch.load(f'checkpoints/ffhq128_denoising/norm/last.ckpt', map_location='cpu')
    denosing_model.load_state_dict(ckpt['state_dict'])
    start_epoch = ckpt['epoch'] + 1
    print(f"Resuming from epoch {start_epoch}")

for epoch in range(start_epoch, max_epochs):
    with tqdm(dataloader, desc=f"Epoch {epoch}") as pbar:
        loss_list, loss_denorm_list = [], []
        for i, batch in enumerate(dataloader):
            latents = batch[0].cuda()
            latents_norm = diffae.normalize(latents)
            # add noise
            snr = torch.tensor(random.choices(SNR_list, k=latents.shape[0])).cuda()
            latents_power = latents_norm.pow(2).mean(dim=1)
            noise_power = latents_power / (10 ** (snr / 10))
            noise = torch.randn_like(latents_norm) * (torch.sqrt(noise_power).unsqueeze(1))
            noisy_latents = latents_norm + noise
            # denoise
            denoised_latents_norm = denosing_model(noisy_latents, snr)
            loss = (denoised_latents_norm.pred - latents_norm).abs().mean()
            denoised_latents = diffae.denormalize(denoised_latents_norm.pred)
            # noise_hat = denosing_model(noisy_latents, snr).pred
            # loss = (noise_hat - noise).abs().mean()
            loss_denorm = (denoised_latents - latents).abs().mean()
            loss_denorm_list.append(loss_denorm.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            avg_loss = sum(loss_list) / len(loss_list)
            pbar.set_postfix_str(f"loss={loss.item():.4f}, avg_loss={avg_loss:.4f}, loss_denorm={loss_denorm.item():.4f}")
            pbar.update(1)
        avg_loss = sum(loss_list) / len(loss_list)
        avg_loss_denorm = sum(loss_denorm_list) / len(loss_denorm_list)
        print(f"Epoch {epoch}: Average loss={avg_loss:.4f}, Average loss_denorm={avg_loss_denorm:.4f}")
        ckpt = {'state_dict': denosing_model.state_dict(), "epoch": epoch}
        torch.save(ckpt, f'checkpoints/ffhq128_denoising/norm/last.ckpt')





