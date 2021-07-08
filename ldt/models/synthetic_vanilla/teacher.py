import torch
from torch import nn as nn


class Teacher(nn.Module):
    def __init__(self, privileged_dim, student_hidden_dim,
                 teacher_hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(privileged_dim, teacher_hidden_dim)
        self.fc2 = nn.Linear(teacher_hidden_dim, teacher_hidden_dim)
        self.fc3 = nn.Linear(teacher_hidden_dim, student_hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


class PerfectTeacher(nn.Module):
    def __init__(self, privileged_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(privileged_dim, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, hidden_dim)

        with torch.no_grad():
            self.fc1.weight.set_()


    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x