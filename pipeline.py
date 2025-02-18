import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal

LAMDA = 1
PRINT_INTERVAL = 100
K = 512

def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()

def train(model, train_loader, opt, device):
    train_loss = []
    for batch_idx, (x, _) in enumerate(train_loader):
        start_time = time.time()
        x = x.to(device)

        opt.zero_grad()

        x_tilde, z_e_x, z_q_x = model(x)
        z_q_x.retain_grad()

        loss_recons = F.mse_loss(x_tilde, x)
        loss_recons.backward(retain_graph=True)

        # Straight-through estimator
        z_e_x.backward(z_q_x.grad, retain_graph=True)

        # Vector quantization objective
        model.codebook.zero_grad()
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        loss_vq.backward(retain_graph=True)

        # Commitment objective
        loss_commit = LAMDA * F.mse_loss(z_e_x, z_q_x.detach())
        loss_commit.backward()
        opt.step()

        N = x.numel()
        nll = Normal(x_tilde, torch.ones_like(x_tilde)).log_prob(x)
        log_px = nll.sum() / N + np.log(128) - np.log(K * 2)
        log_px /= np.log(2)

        train_loss.append(
            [log_px.item()] + to_scalar([loss_recons, loss_vq])
        )

        if (batch_idx + 1) % PRINT_INTERVAL == 0:
            print('\tIter [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                batch_idx * len(x), len(train_loader.dataset),
                PRINT_INTERVAL * batch_idx / len(train_loader),
                np.asarray(train_loss)[-PRINT_INTERVAL:].mean(0),
                time.time() - start_time
            ))


def test(model, test_loader, device):
    start_time = time.time()
    val_loss = []
    for batch_idx, (x, _) in enumerate(test_loader):
        x = x.to(device)
        x_tilde, z_e_x, z_q_x = model(x)
        loss_recons = F.mse_loss(x_tilde, x)
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        val_loss.append(to_scalar([loss_recons, loss_vq]))

    print('\nValidation Completed!\tLoss: {} Time: {:5.3f}'.format(
        np.asarray(val_loss).mean(0),
        time.time() - start_time
    ))
    return np.asarray(val_loss).mean(0)
