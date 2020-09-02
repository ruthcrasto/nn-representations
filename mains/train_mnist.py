import torch
import torchvision

from models.mlp import MLP
from models.cnn import CNN
from utils.dataset import Dataset
from utils.train import train

# Hyperparameters
num_iters = 20
batch_size = 512
lr = 0.001
train_size = 40000


# Load train/test data
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor()
                                        ]))

mnist_loader = torch.utils.data.DataLoader(train_data, batch_size=60000, shuffle=True)


# Pre-processing
_, (X, t) = next(enumerate(mnist_loader))
X = torch.squeeze(X.view(-1, 28 * 28))
X_mean = torch.mean(X, dim=0)
centered_data = X - X_mean

# centered_data = torch.reshape(centered_data, (-1, 1, 28, 28)) # for cnn only. TODO: add model selection logic.

train_loader = torch.utils.data.DataLoader(Dataset(centered_data[:train_size], t[:train_size]), batch_size=batch_size, shuffle=False)
valid_loader = torch.utils.data.DataLoader(Dataset(centered_data[train_size:], t[train_size:]), batch_size=batch_size, shuffle=False)


# Training
model = MLP()
train(model, train_loader, valid_loader, {
    "lr": lr,
    "batch_size": batch_size,
    "num_iters": num_iters
})
