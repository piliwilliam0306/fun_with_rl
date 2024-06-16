
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
    from model import DummyNetwork
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_model = DummyNetwork().to(device)
    target_model = DummyNetwork().to(device)

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
