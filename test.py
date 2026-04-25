import torch
from models.generator import MambaSEUNet
import yaml
cfg = yaml.safe_load(open('recipes/Mamba-SEUNet/Mamba-SEUNet.yaml'))
model = MambaSEUNet(cfg)
x_mag = torch.randn(2, cfg['stft_cfg']['n_fft']//2+1, 10)
x_pha = torch.randn_like(x_mag)
with torch.no_grad():
    dm, dp, dc = model(x_mag, x_pha)
print(dm.shape, dp.shape, dc.shape)
