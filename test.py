import torch
from torch.utils.data import DataLoader, TensorDataset

from models.vqvae import SignalVectorQuantizedVAE

model = SignalVectorQuantizedVAE(patch_size=100, channel_size=32)
input = torch.randn((1, 1, 1, 100))
x_tilde, z_e_x, z_q_x = model(input)
print(x_tilde.shape, z_e_x.shape, z_q_x.shape)