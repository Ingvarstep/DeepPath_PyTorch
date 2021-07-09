import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Add regularization

class PolicyNN(nn.Module):

    def __init__(self, state_dim, action_dim, initializer=None):
        super(PolicyNN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512, bias=True)
        self.fc2 = nn.Linear(512, 1024, bias=True)
        self.fc3 = nn.Linear(1024, action_dim, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        y = F.relu(self.fc1(state))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        action_probs = self.softmax(y)
        return action_probs

class ValueNN(nn.Module):

    def __init__(self, state_dim, initializer=None):
        super(ValueNN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64, bias=True)
        self.fc2 = nn.Linear(64, 1, bias=True)

    def forward(self, state):
        y = F.relu(self.fc1(state))
        value_estimated = self.fc2(y)
        return torch.squeeze(value_estimated)

class QNetwork(nn.Module):

    def __init__(self, state_dim, action_space):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_space)

    def forward(self, state):
        y = F.relu(self.fc1(state))
        y = F.relu(self.fc2(y))
        action_values = self.fc3(y)
        return action_values
