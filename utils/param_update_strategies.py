
def soft_update(local_model, target_model, tau=1):
    """
    θ_target = τ * θ_local + (1 - τ) * θ_target
    When tau is equal to 1, the soft_update function will behave as hard update

    Params
    ======
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter 
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

if __name__ == "__main__":
    import torch.nn as nn

    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(2, 2)

        def forward(self, x):
            x = self.fc1(x)
            return x

    local_model = SimpleNet()
    target_model = SimpleNet()

    print("Initial weights of target model:")
    for param in target_model.parameters():
        print(param.data)
    print("-" * 50)

    soft_update(local_model, target_model, tau=1)

    print("Weights of local model:")
    for param in local_model.parameters():
        print(param.data)
    print("-" * 50)

    print("Updated weights of target model after soft update:")
    for param in target_model.parameters():
        print(param.data)
