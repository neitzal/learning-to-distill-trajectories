import torch
from torch import nn as nn


class StudentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, n_classes, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        logits = self.fc3(x)

        # Supervise the logits only
        activations = logits

        return logits, activations