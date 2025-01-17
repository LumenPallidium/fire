import torch

class LinearMapLearner(torch.nn.Module):
    """
    Implementing the cognitive map learner from:
    https://www.nature.com/articles/s41467-024-46586-0
    """
    def __init__(self, state_dim, action_dim, hidden_dim = 128, bias = False):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.state_embed = torch.nn.Linear(state_dim, hidden_dim,
                                           bias = bias)
        self.action_embed = torch.nn.Linear(action_dim, hidden_dim,
                                            bias = bias)

    def forward(self, state, action):
        state_embed = self.state_embed(state)
        action_embed = self.action_embed(action)
        return state_embed + action_embed
    
    def manual_update(self, obs, action, next_obs, lr = 0.01):
        assert self.bias == False, "Bias is not supported for manual update"
        with torch.no_grad():
            next_state_hat = self.forward(obs, action)
            pred_error = next_state_hat - next_obs

            dV = torch.einsum("bi,bj->bij",
                              -pred_error,
                              action).mean(dim = 0)
            dQ = torch.einsum("bi,bj->bij",
                              pred_error,
                              obs).mean(dim = 0)
            self.state_embed.weight += lr * dV
            self.action_embed.weight += lr * dQ

            # normalize V
            self.state_embed.weight /= torch.norm(self.state_embed.weight,
                                                  dim = 1).unsqueeze(1)