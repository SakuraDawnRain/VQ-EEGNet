import torch
from torch.utils.data import DataLoader, TensorDataset

from models.vqvae import SignalVectorQuantizedVAE
from pipeline import train, test
from vis import vis

# parameters
device="cuda"
batch_size = 1024
LR = 1e-3
patch_size = 20
epochs = 10

# Load Data
X_train = torch.load("data/X_train.pt", weights_only=True).to(device)[:, :, :, :1000]
X_test = torch.load("data/X_test.pt", weights_only=True).to(device)[:, :, :, :1000]
y_train = torch.load("data/y_train.pt", weights_only=True).to(device)
y_test = torch.load("data/y_test.pt", weights_only=True).to(device)

# Train VQVAE for Signal Reconstruction
signal_train = X_train.reshape((X_train.shape[0]*X_train.shape[2]*X_train.shape[3]//patch_size, 1, 1, patch_size))
signal_train_set = TensorDataset(signal_train, torch.zeros((X_train.shape[0]*X_train.shape[2]*X_train.shape[3]//patch_size, 1, 1, patch_size)))
signal_train_loader = DataLoader(signal_train_set, batch_size=batch_size, shuffle=True)
signal_test = X_test.reshape((X_test.shape[0]*X_test.shape[2]*X_test.shape[3]//patch_size, 1, 1, patch_size))
signal_test_set = TensorDataset(signal_test, torch.zeros((X_test.shape[0]*X_test.shape[2]*X_test.shape[3]//patch_size, 1, 1, patch_size)))
signal_test_loader = DataLoader(signal_test_set, batch_size=batch_size, shuffle=False)

model = SignalVectorQuantizedVAE(patch_size=patch_size).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=True)
for epoch in range(epochs):
    print("start training epoch", str(epoch))
    train(model, signal_train_loader, opt, device)
test(model, signal_test_loader, device)
torch.save(model.state_dict(), "signal-vqvae.pth")

# Visualize VQ result on test set
vis(model_path="signal-vqvae.pth", data_path="data/X_test.pt", patch_size=patch_size)
