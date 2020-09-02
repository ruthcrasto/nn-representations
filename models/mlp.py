import torch.nn as nn

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
          nn.Linear(784, 512),
          nn.ReLU(),
          nn.Linear(512, 256),
          nn.ReLU(),
          nn.Linear(256, 128),
          nn.ReLU(),
        )
        self.softmax = nn.Sequential(
          nn.Linear(128, 10),
          nn.Softmax(dim=1)
        )

    def forward(self, x):
      return self.softmax(self.mlp(x))

    def get_representation(self, x):
      return self.mlp(x)

    def reshape_data(self, X):
        return X
