from templates import *
from templates_latent import *

device = 'cuda:0'
conf = ffhq128_autoenc_latent()
conf.T_eval = 100
conf.latent_T_eval = 100
# print(conf.name)
model = LitModel(conf)
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
print(model.load_state_dict(state['state_dict'], strict=False))
model.to(device)

torch.manual_seed(1)
imgs = model.sample(8, device=device, T=20, T_latent=200)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 4, figsize=(4*5, 2*5))
ax = ax.flatten()
for i in range(len(imgs)):
    ax[i].imshow(imgs[i].cpu().permute([1, 2, 0]))
    ax[i].axis('off')
plt.tight_layout()
plt.savefig('sample.pdf', dpi=300)