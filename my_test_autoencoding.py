from templates import *
import matplotlib.pyplot as plt
from PIL import Image
import time

device = 'cuda:0'
conf = ffhq128_autoenc_130M()
# print(conf.name)
model = LitModel(conf)
state = torch.load(f'checkpoints/{conf.name}/last-v5.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

data = ImageDataset('imgs_align', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
batch = data[2]['img'][None]  # (1, 3, 128, 128)  [0, 1]

plt.imshow(batch[0].permute([1, 2, 0]) / 2 + 0.5)

cond = model.encode(batch.to(device))
xT = model.encode_stochastic(batch.to(device), cond, T=250) # 这里DDIM的T居然只有250步，这里的T不影响，改成2结果也一样
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ori = (batch + 1) / 2
ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
ax[0].set_title('Original Image')
# ax[1].imshow(xT[0].permute(1, 2, 0).cpu())
ax[1].set_title('Encoded Image')
# plt.savefig('output_images_celeba.png', bbox_inches='tight') 


print("Encoding finished.")


snr = 10
# Decode
seed = 4
torch.random.manual_seed(seed)  # original is 1
np.random.seed(seed)  # original is 0
torch.cuda.manual_seed(seed)  # original is 0
torch.cuda.manual_seed_all(seed)  # original is 0
xT1 = torch.randn_like(xT)  # (1, 128)
xT2 = torch.randn_like(batch)  # (1, 128)
# pred = model.render(xT1, cond, T=20)

start_time = time.time()
T = 100
pred = model.render(xT, cond, T=T)
end_time = time.time()
print(f"Time used: {end_time - start_time:.2f}s")
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ori = (batch[0] + 1) / 2
image = ori.permute(1, 2, 0).cpu().numpy() * 255
image = image.astype(np.uint8)
pred = pred[0].permute(1, 2, 0).cpu().numpy() * 255
pred = pred.astype(np.uint8)
img_ori = Image.fromarray(image)
# img_ori.save('output_images_celeba_ori_male.png')
img_pred = Image.fromarray(pred)
img_pred.save(f'results/xT_comparison/869/reconstruction_{snr}.png')
# img_pred.save(f'output_images_celeba_pred_{snr}_seed_{seed}.png')
mse = ((image - pred) ** 2).mean()
psnr = 10 * np.log10(255 ** 2 / mse)
print(f"PSNR: {psnr:.2f}")
# ax[0].imshow(image)
# ax[1].imshow(pred)
# ax[0].set_title('Original Image')
# ax[1].set_title('Decoded Image')
# plt.savefig('output_images_decoded_celeba_snr_5_random.png', bbox_inches='tight') 
print("Decoding finished.")

# img_ori = Image.fromarray(image)
# img_pred = Image.fromarray(pred)

# # 拼接图片，中间留空白
# gap = 20  # 空白宽度
# w, h = img_ori.size
# comparison = Image.new('RGB', (w * 2 + gap, h), (255, 255, 255))  # 白色背景
# comparison.paste(img_ori, (0, 0))
# comparison.paste(img_pred, (w + gap, 0))
# comparison.save('image_comparison.png')
# print("Saved image_comparison.png")
