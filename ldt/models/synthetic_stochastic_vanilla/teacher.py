import torch
from torch import nn as nn


class Teacher(nn.Module):
    def __init__(self, teacher_hidden_dim, privileged_dim, supervision_dim):
        super().__init__()
        self.fc1 = nn.Linear(privileged_dim, teacher_hidden_dim)
        self.fc2 = nn.Linear(teacher_hidden_dim, teacher_hidden_dim)
        self.fc3 = nn.Linear(teacher_hidden_dim, supervision_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


