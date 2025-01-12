import torch
import numpy as np
from gymnasium import make
from utils import ReplayBuffer, NormedSparseMLP, SACWrapper
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
                 dims = [256, 128],
                 sparsity = 0.2):
        super().__init__()
        self.obs_dim = obs_dim
        self.skill_dim = skill_dim

        self.encoder = NormedSparseMLP([obs_dim] + dims + [skill_dim],
                                       sparsity = sparsity)

    def forward(self, obs):
        return self.encoder(obs)
    
class SkillConditionedPolicy(torch.nn.Module):
    def __init__(self,
                 obs_dim,
                 skill_dim,
                 action_dim,
                 discrete = False,
                 variance = False,
                 dims = [256, 128],
                 sparsity = 0.2):
        super().__init__()
        self.obs_dim = obs_dim
        self.skill_dim = skill_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.variance = variance

        self.actor_embed = NormedSparseMLP([obs_dim + skill_dim] + dims,
                                           sparsity = sparsity)
        self.actor = torch.nn.Linear(dims[-1], action_dim)
        if self.variance:
            self.logstd = torch.nn.Linear(dims[-1], action_dim)

        self.softmax = torch.nn.Softmax(dim = -1)

    def forward(self, obs, skill):
        # paper doesn't say how both inputs layout, assume concat
        x = torch.cat([obs, skill], dim = -1)
        x = self.actor_embed(x)
        if self.discrete:
            return self.softmax(self.actor(x))
        elif self.variance:
            return self.actor(x), self.logstd(x)
        return self.actor(x)
    
    def stochastic_sample(self, obs, skill):
        assert self.variance
        mu, logstd = self.forward(obs, skill)
        std = torch.exp(logstd)

        normal = torch.distributions.Normal(mu, std)
        eps = normal.sample()
        log_prob = normal.log_prob(eps).sum(dim = -1)
        return mu + std * eps, log_prob
    
class SuccessorEncoder(torch.nn.Module):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 skill_dim,
                 dims = [256, 128],
                 sparsity = 0.2):
        super().__init__()
        self.obs_dim = obs_dim
        self.skill_dim = skill_dim

        self.encoder = NormedSparseMLP([obs_dim + action_dim + skill_dim] + dims + [skill_dim],
                                       sparsity = sparsity)

    def forward(self, obs, action_onehot, skill):
        x = torch.cat([obs, action_onehot, skill], dim = -1)
        return self.encoder(x)
    
    def ema(self, other, alpha = 0.99):
        for p, q in zip(self.parameters(), other.parameters()):
            p.data = alpha * p.data + (1 - alpha) * q.data

def fill_buffer(env, policy, device, skill_space, buffer, pbar, counter):
    done = False
    # fill buffer
    obs, _ = env.reset()
    skill = skill_space.sample().to(device)
    ep_traj = []
    while not done:
        action, action_std = policy(torch.tensor(obs,
                                    device = device,
                                    dtype = torch.float32),
                        skill)
        action = action + action_std.exp() * torch.randn_like(action)
        next_obs, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
        done = terminated or truncated

        buffer.push(obs, action, next_obs, done, reward = reward, special = skill)
        obs = next_obs
        counter += 1
        pbar.update(0)
        pbar.set_description(f"Filling buffer {counter}/{buffer.capacity}")
        ep_traj.append([info["x_position"], info["y_position"]])
    return ep_traj, counter


def main(n_epochs = 100,
         steps_per_epoch = 512,
         batch_size = 256,
         buffer_capacity = int(1e4),
         episodes_per_epoch = 4,
         gamma = 0.99,
         skill_dim = 2,
         traj_snapshot_every = 10,
         save_model_every = 50):
    env = make("Ant-v5")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    skill_space = ContinuousSkillSpace(skill_dim).to(device)
    state_representer = StateRepresenter(obs_dim, skill_dim).to(device)
    policy = SkillConditionedPolicy(obs_dim, skill_dim, action_dim,
                                    variance = True).to(device)
    sac_model = SACWrapper(obs_dim, action_dim,
                           #TODO : don't hardcode reward dim
                           reward_dim = 256).to(device)
    successor_encoder = SuccessorEncoder(obs_dim, action_dim, skill_dim).to(device)
    successor_encoder_ema = deepcopy(successor_encoder)
    successor_encoder_ema.requires_grad_(False)

    optimizer_policy = torch.optim.Adam(policy.parameters(), lr = 1e-4)
    optimizer_successor = torch.optim.Adam(successor_encoder.parameters(), lr = 1e-4)
    optimizer_encoder = torch.optim.Adam(state_representer.parameters(), lr = 1e-4)

    buffer = ReplayBuffer(obs_dim,
                          action_dim,
                          special_buffer_dim = skill_dim,
                          capacity = buffer_capacity)
    
    pbar = tqdm(range(n_epochs * steps_per_epoch))
    losses = []

    xy_trajs = []

    for epoch in range(n_epochs):
        counter = 0
        for ep in range(episodes_per_epoch):
            ep_traj, counter = fill_buffer(env, policy, device,
                                           skill_space, buffer, pbar,
                                           counter)
            # record all episodes for given epoch
            if epoch % traj_snapshot_every == 0:
                xy_trajs.append(ep_traj)
            
        for _ in range(steps_per_epoch):
            optimizer_encoder.zero_grad()
            optimizer_successor.zero_grad()
            optimizer_policy.zero_grad()

            states, actions, rewards, next_states, dones, skills = buffer.sample(batch_size,
                                                                                 device = device)
            
            # new actions
            new_actions, new_actions_std = policy(states, skills)
            new_next_actions, next_actions_std = policy(next_states, skills)

            new_actions = new_actions + new_actions_std.exp() * torch.randn_like(new_actions)
            new_next_actions = new_next_actions + next_actions_std.exp() * torch.randn_like(new_next_actions)
            
            state_reps = state_representer(states)
            next_state_reps = state_representer(next_states)
            successors = successor_encoder(states, new_actions, skills)

            state_diffs = next_state_reps - state_reps
            counterfactual_skills = skill_space.sample(n_samples = batch_size).to(device)

            encoder_loss = -(state_diffs * skills).sum(dim=-1).mean()
            nce_term = torch.einsum("bi,mi->bm",
                                    state_diffs,
                                    counterfactual_skills)
            nce_term = torch.exp(nce_term).mean(dim = -1).log()
            encoder_loss += nce_term.mean()

            with torch.no_grad():
                successors_emas = successor_encoder_ema(next_states,
                                                        new_next_actions,
                                                        skills)
                successor_td = state_diffs + gamma * successors_emas

            successor_loss = torch.nn.functional.mse_loss(successors,
                                                          successor_td.detach())
            policy_reward = -(successors * skills).sum(dim=-1).mean()
            policy_loss, _ = sac_model([states, actions, rewards, next_states, dones, skills],
                                       policy,
                                       reward = policy_reward)
            
            encoder_loss.backward()
            successor_loss.backward(retain_graph = True)
            policy_loss.backward()

            optimizer_encoder.step()
            optimizer_successor.step()
            sac_model.step()
            optimizer_policy.step()

            successor_encoder_ema.ema(successor_encoder)
            losses.append([encoder_loss.item(),
                           successor_loss.item(),
                           policy_loss.item()])

            pbar.update(1)
            pbar.set_description(f"Epoch {epoch} | Loss {encoder_loss.item():.2f} | {successor_loss.item():.2f} | {policy_loss.item():.2f}")
        if epoch % save_model_every == 0:
            # save models
            torch.save(state_representer.state_dict(),
                    f"tmp/state_representer_{epoch}.pt")
            torch.save(policy.state_dict(),
                    f"tmp/policy_{epoch}.pt")
            torch.save(successor_encoder.state_dict(),
                    f"tmp/successor_encoder_{epoch}.pt")
            torch.save(sac_model.state_dict(),
                       f"tmp/sac_model_{epoch}.pt")
    return np.array(losses), xy_trajs

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    os.makedirs("tmp", exist_ok = True)
    n_epochs = 50
    episodes_per_epoch = 8
    traj_snapshot_every = n_epochs // 10
    smoothing_num = max(1, int(n_epochs / 10))
    smooth_kernel = np.ones(smoothing_num) / smoothing_num

    losses, trajs = main(n_epochs = n_epochs,
                         steps_per_epoch = 512,
                         buffer_capacity = int(1e6),
                         episodes_per_epoch = episodes_per_epoch,
                         traj_snapshot_every = traj_snapshot_every,
                         batch_size = 256)

    fig, ax = plt.subplots(1, 3, figsize = (12, 6))
    # plot the losses
    for i, label in enumerate(["Encoder", "Successor", "Policy"]):
        smooth_losses = np.convolve(losses[:, i],
                                    smooth_kernel,
                                    mode = "valid")
        smooth_losses /= smooth_losses.max() - smooth_losses.min()
        ax[0].plot(smooth_losses, label = label)
    ax[0].legend()
    dists_time = []
    # plots some of the trajectories
    for i in range(len(trajs)):
        traj = np.array(trajs[i])
        alpha = (i / len(trajs))
        ax[1].plot(traj[:, 0], traj[:, 1], alpha = alpha)

        # lastly get mean distance travelled over time
        dists = np.linalg.norm(traj[0, :] - traj[-1, :], axis = -1)
        dists_time.append(dists)
    ax[2].plot(dists_time)

    fig.savefig("tmp/skill_results.png", dpi = 300)