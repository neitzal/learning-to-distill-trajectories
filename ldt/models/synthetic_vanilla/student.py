import torch
from torch import nn as nn


class StudentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, 128, bias=True)
        self.fc3 = nn.Linear(128, 1, bias=True)

    def forward(self, x):
        hidden = self.fc1(x)
        y = self.fc2(hidden)
        y = torch.relu(y)
        y = self.fc3(y)
        y = y.squeeze(1)
        return y, hidden