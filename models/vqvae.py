import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)

class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        # z_e_x - (B, D, H, W)
        # emb   - (K, D)

        emb = self.embedding.weight
        dists = torch.pow(
            z_e_x.unsqueeze(1) - emb[None, :, :, None, None],
            2
        ).sum(2)

        latents = dists.min(1)[1]
        return latents

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class SignalVectorQuantizedVAE(nn.Module):
    def __init__(self, patch_size, temp_kernel=15, channel_size=16, K=2048, padding=(0, 7)):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, channel_size, (1, temp_kernel), padding=padding, bias=False),
            nn.BatchNorm2d(channel_size),
        )

        self.codebook = VQEmbedding(K, patch_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channel_size, 1, (1, temp_kernel), padding=padding, bias=False),
            nn.Tanh()
        )

        self.apply(weights_init)

    def encode(self, x):
        # x Shape = (B, 1, 1, P)
        # B = Batch Size
        # P = Patch Size
        z_e_x = self.encoder(x)
        # z_e_x Shape = (B, C, 1, P)
        # C = Channel Size
        latents = self.codebook(z_e_x.permute(0, 3, 1, 2))
        # latents Shape = (B, C, 1)
        return latents, z_e_x

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents)
        # z_q_x Shape = (B, C, 1, P)
        x_tilde = self.decoder(z_q_x)
        # x_tilde Shape = (B, 1, 1, P)
        return x_tilde, z_q_x

    def forward(self, x):
        latents, z_e_x = self.encode(x)
        x_tilde, z_q_x = self.decode(latents)
        return x_tilde, z_e_x, z_q_x

# class VectorQuantizedEEGNet(nn.Module):
#     def __init__(self, signal_vqvae, input_dim, dim, K=512):
#         super().__init__()
#         # Load Signal VQVAE and Freeze
#         self.signal_vqvae = signal_vqvae
#         for param in self.signal_vqvae.parameters():
#             param.requires_grad = False

#         self.


#     def forward(self, x):
#         x_tilde, z_e_x, z_q_x = self.signal_vqvae(x)
#         return 
