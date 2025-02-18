import torch
import matplotlib.pyplot as plt

from models.vqvae import SignalVectorQuantizedVAE

def vis(model_path, data_path, patch_size, fig_path="vis.png"):
    model = SignalVectorQuantizedVAE(patch_size)
    model.load_state_dict(torch.load(model_path))

    data = torch.load(data_path, weights_only=True)
    x = data[0:1, 0:1, 0:1, :patch_size]
    x_reconstruct, z_e_x, z_q_x = model(x)

    print(x_reconstruct.shape, z_e_x.shape)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(x[0, 0, 0, :].detach().numpy())
    ax2.plot(x_reconstruct[0, 0, 0, :].detach().numpy())
    plt.savefig(fig_path)
