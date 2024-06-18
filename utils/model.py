import torch
import torch.nn as nn

class DummyNetwork(torch.nn.Module):
    def __init__(self, input_dim=4, output_dim=2):
        super(DummyNetwork, self).__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, fc1_units=128, fc2_units=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
