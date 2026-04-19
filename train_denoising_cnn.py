import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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


class ResidualBlock(nn.Module):
    """
    残差块，由两个一维卷积层和ReLU激活组成，带有残差连接。
    """
    def __init__(self, num_features, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(num_features, num_features, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_features, num_features, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        out += residual
        out = self.relu(out)
        return out

class DenoiseResNet1D(nn.Module):
    """
    基于一维卷积的残差网络，用于从加噪信号中恢复原始信号。
    输入包括加噪信号和SNR，输出为恢复后的信号。
    """
    def __init__(self, num_features=64, num_blocks=5):
        super(DenoiseResNet1D, self).__init__()
        # 初始卷积层：输入为2通道（加噪信号 + SNR），输出为num_features通道
        self.initial_conv = nn.Conv1d(in_channels=2, out_channels=num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # 残差块堆叠
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )
        # 输出卷积层：将特征映射回1通道
        self.final_conv = nn.Conv1d(in_channels=num_features, out_channels=1, kernel_size=3, padding=1)
        self.embedding = nn.Embedding(6, 512)
        self.snr_dict = {1: 0, 4: 1, 7: 2, 10: 3, 13: 4}

    def forward(self, noisy_signal, snr):
        """
        前向传播
        参数：
            noisy_signal: 加噪信号，形状为 (batch_size, 1, N)
            snr: 信噪比，形状为 (batch_size, 1)
        返回：
            denoised_signal: 恢复后的信号，形状为 (batch_size, 1, N)
        """
        noisy_signal = noisy_signal.unsqueeze(1)  # (batch_size, 1, N)
        batch_size, _, N = noisy_signal.size()
        # 将SNR扩展为与信号长度相同的向量
        # snr_expanded = snr.unsqueeze(-1).expand(-1, -1, N)  # (batch_size, 1, N)
        snr = torch.tensor(list(map(self.snr_dict.get, snr))).cuda()
        snr = self.embedding(snr.long()).unsqueeze(1)  # (batch_size, 1, 512)
        # 拼接加噪信号和扩展后的SNR
        x = torch.cat([noisy_signal, snr], dim=1)  # (batch_size, 2, N)
        # 初始卷积和激活
        out = self.initial_conv(x)  # (batch_size, num_features, N)
        out = self.relu(out)
        # 通过残差块堆叠
        out = self.residual_blocks(out)  # (batch_size, num_features, N)
        # 输出卷积
        out = self.final_conv(out)  # (batch_size, 1, N)
        return out.squeeze(1)  # (batch_size, N)



denosing_model = DenoiseResNet1D(num_features=64, num_blocks=10)
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

SNR_list = list(range(1, 14, 3))
max_epochs = 3000

start_epoch = 0
# resume
if os.path.exists(f'checkpoints/ffhq128_denoising_cnn/last.ckpt'):
    ckpt = torch.load(f'checkpoints/ffhq128_denoising_cnn/last.ckpt', map_location='cpu')
    denosing_model.load_state_dict(ckpt['state_dict'])
    start_epoch = ckpt['epoch'] + 1
    print(f"Resuming from epoch {start_epoch}")

for epoch in range(start_epoch, max_epochs):
    with tqdm(dataloader, desc=f"Epoch {epoch}") as pbar:
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
            denoised_latents_norm = denosing_model(noisy_latents, snr)
            denoised_latents = diffae.denormalize(denoised_latents_norm)
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
        print(f"Epoch {epoch}: Average loss={sum(loss_list) / len(loss_list)}")
        ckpt = {'state_dict': denosing_model.state_dict(), "epoch": epoch}
        torch.save(ckpt, f'checkpoints/ffhq128_denoising_cnn/last.ckpt')
