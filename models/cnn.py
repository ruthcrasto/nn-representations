import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, kernel_size):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
          nn.Conv2d(1, 16, kernel_size, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(16, 32, kernel_size, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.softmax = nn.Sequential(
          nn.Linear(7 * 7 * 32, 10),
          nn.Softmax(dim=1)
        )

    def forward(self, x):
      cnn = self.cnn(x)
      return self.softmax(torch.reshape(cnn, (cnn.shape[0], -1)))

    def get_representation(self, x):
      return self.cnn(x)

    def reshape_data(self, X):
        return X

