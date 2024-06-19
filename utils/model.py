import torch
import torch.nn as nn

class DummyNetwork(torch.nn.Module):
    def __init__(self, input_dim=4, output_dim=2):
        super(DummyNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QNetwork(4, 2).to(device)
    x = torch.rand(10, 4).to(device)
    out = model(x)
    print(out.shape)
