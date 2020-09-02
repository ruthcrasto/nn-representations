import torch

class Dataset(torch.utils.data.Dataset):
  def __init__(self, data, labels):
        self.labels = labels
        self.data = data

  def __len__(self):
        return len(self.data)

  def __getitem__(self, index):
        X = self.data[index]
        y = self.labels[index]

        return X, y
