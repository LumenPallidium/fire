import torch
from gymnasium import make
from utils import ReplayBuffer, NormedSparseMLP
from copy import deepcopy
from tqdm import tqdm

class DiscreteSkillSpace(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def sample(self, n_samples = 1):
        x = torch.randint(0, self.dim, (n_samples,))
        if n_samples == 1:
            x = x.squeeze(0)
        return x
    
    def __len__(self):
        return self.dim
    
    def __repr__(self):
        return f"DiscreteSkillSpace(dim={self.dim})"
    
class ContinuousSkillSpace(torch.nn.Module):
    def __init__(self, dim, von_mises = True):
        super().__init__()
        self.dim = dim
        self.von_mises = von_mises

    def sample(self, n_samples = 1):
        x = torch.randn(n_samples, self.dim, dtype = torch.float32)
        if self.von_mises:
            # assumes kappa is 0
            x /= x.norm(p = 2)
        if n_samples == 1:
            x = x.squeeze(0)
        return x
    
    def __len__(self):
        return self.dim
    
    def __repr__(self):
        return f"ContinuousSkillSpace(dim={self.dim})"
    
class StateRepresenter(torch.nn.Module):
    def __init__(self,
                 obs_dim,
                 skill_dim,
                 dims = [256, 128]):
        super().__init__()
        self.obs_dim = obs_dim
        self.skill_dim = skill_dim

        self.encoder = NormedSparseMLP([obs_dim] + dims + [skill_dim])

    def forward(self, obs):
        return self.encoder(obs)
    
class SkillConditionedPolicy(torch.nn.Module):
    def __init__(self,
                 obs_dim,
                 skill_dim,
                 action_dim,
                 discrete = False,
                 dims = [256, 128]):
        super().__init__()
        self.obs_dim = obs_dim
        self.skill_dim = skill_dim
        self.action_dim = action_dim
        self.discrete = discrete

        self.actor = NormedSparseMLP([obs_dim + skill_dim] + dims + [action_dim])
        self.softmax = torch.nn.Softmax(dim = -1)

    def forward(self, obs, skill):
        # paper doesn't say how both inputs layout, assume concat
        x = torch.cat([obs, skill], dim = -1)
        if self.discrete:
            return self.softmax(self.actor(x))
        return self.actor(x)
    
class SuccessorEncoder(torch.nn.Module):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 skill_dim,
                 dims = [256, 128]):
        super().__init__()
        self.obs_dim = obs_dim
        self.skill_dim = skill_dim

        self.encoder = NormedSparseMLP([obs_dim + action_dim + skill_dim] + dims + [skill_dim])

    def forward(self, obs, action_onehot, skill):
        x = torch.cat([obs, action_onehot, skill], dim = -1)
        return self.encoder(x)
    
    def ema(self, other, alpha = 0.99):
        for p, q in zip(self.parameters(), other.parameters()):
            p.data = alpha * p.data + (1 - alpha) * q.data

def main(n_epochs = 100,
         steps_per_epoch = 512,
         batch_size = 256,
         n_counterfactual_skills = 256,
         gamma = 0.99,
         skill_dim = 128):
    env = make("Ant-v5")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    skill_space = ContinuousSkillSpace(skill_dim).to(device)
    state_representer = StateRepresenter(obs_dim, skill_dim).to(device)
    policy = SkillConditionedPolicy(obs_dim, skill_dim, action_dim).to(device)
    successor_encoder = SuccessorEncoder(obs_dim, action_dim, skill_dim).to(device)
    successor_encoder_ema = deepcopy(successor_encoder)
    successor_encoder_ema.requires_grad_(False)

    optimizer_policy = torch.optim.Adam(policy.parameters(), lr = 1e-4)
    optimizer_successor = torch.optim.Adam(successor_encoder.parameters(), lr = 1e-4)
    optimizer_encoder = torch.optim.Adam(state_representer.parameters(), lr = 1e-4)

    buffer = ReplayBuffer(obs_dim,
                          action_dim,
                          special_buffer_dim = skill_dim,
                          capacity = steps_per_epoch * batch_size)
    
    pbar = tqdm(range(n_epochs * steps_per_epoch))
    losses = []

    for epoch in range(n_epochs):
        # fill buffer
        obs, _ = env.reset()
        for buffer_step in range(buffer.capacity):
            skill = skill_space.sample().to(device)
            action = policy(torch.tensor(obs,
                                         device = device,
                                         dtype = torch.float32),
                                  skill)
            next_obs, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy())
            done = terminated or truncated

            buffer.push(obs, action, next_obs, done, reward = reward, special = skill)
            obs = next_obs
            if done:
                obs, _ = env.reset()
            pbar.set_description(f"Filling buffer {buffer_step}/{buffer.capacity}")
            

        for _ in range(steps_per_epoch):
            optimizer_encoder.zero_grad()
            optimizer_successor.zero_grad()
            optimizer_policy.zero_grad()

            states, actions, rewards, next_states, dones, skills = buffer.sample(batch_size,
                                                                                 device = device)
            
            # new skills and actions
            counterfactual_skills = skill_space.sample(n_samples = n_counterfactual_skills)
            counterfactual_skills = counterfactual_skills.repeat(batch_size, 1, 1).to(device)
            new_actions = policy(states, skills)

            state_reps = state_representer(states)
            next_state_reps = state_representer(next_states)
            successors = successor_encoder(states, actions, skills)
            successors_emas = successor_encoder_ema(next_states,
                                                    new_actions,
                                                    skills)

            state_diffs = next_state_reps - state_reps

            encoder_loss = -torch.einsum("bi,bi->b",
                                         state_diffs,
                                         skills).mean()
            nce_term = torch.einsum("bi,bmi->bm",
                                    state_diffs,
                                    counterfactual_skills)
            nce_term = torch.logsumexp(nce_term, dim = 1)
            encoder_loss += nce_term.mean()

            successor_td = state_diffs + gamma * successors_emas

            successor_loss = torch.nn.functional.mse_loss(successors,
                                                          successor_td.detach())
            policy_loss = -torch.einsum("bi,bi->b",
                                        successors,
                                        skills).mean()
            
            encoder_loss.backward()
            successor_loss.backward(retain_graph = True)
            policy_loss.backward()

            optimizer_encoder.step()
            optimizer_successor.step()
            optimizer_policy.step()

            successor_encoder_ema.ema(successor_encoder)
            losses.append((encoder_loss.item(),
                           successor_loss.item(),
                           policy_loss.item()))

            pbar.update(1)
            pbar.set_description(f"Epoch {epoch} | Loss {encoder_loss.item() + successor_loss.item() + policy_loss.item():.2f}")
    return losses

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    losses = main(n_epochs = 100,
                  steps_per_epoch = 8,
                  batch_size = 256)

    fig, ax = plt.subplots(1, 1, figsize = (12, 6))
    ax.plot(losses)