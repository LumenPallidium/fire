import torch
import numpy as np
from gymnasium import make
from utils import ReplayBuffer, NormedSparseMLP, SACWrapper, SkillConditionedPolicy
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
        action, _ = policy.stochastic_sample(torch.tensor(obs,
                                             device = device,
                                             dtype = torch.float32),
                                             skill = skill)
        next_obs, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
        done = (terminated or truncated) and (buffer.min_capacity < buffer.total)

        buffer.push(obs, action, next_obs, done, reward = reward, special = skill)
        obs = next_obs
        counter += 1
        pbar.update(0)
        pbar.set_description(f"Filling buffer {counter}/{buffer.capacity}")
        ep_traj.append([info["x_position"], info["y_position"]])
    return ep_traj, counter

def continuous_metra(n_epochs = 100,
                     steps_per_epoch = 512,
                     batch_size = 256,
                     buffer_capacity = int(1e4),
                     episodes_per_epoch = 4,
                     gamma = 0.99,
                     lam = 30,
                     eps = 1e-3,
                     lr = 1e-4,
                     skill_dim = 2,
                     env_name = "Ant-v5",
                     traj_snapshot_every = 10,
                     save_model_every = 50):
    env = make(env_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    target_entropy = -action_dim

    skill_space = ContinuousSkillSpace(skill_dim).to(device)
    state_representer = StateRepresenter(obs_dim, skill_dim).to(device)
    policy = SkillConditionedPolicy(obs_dim, action_dim, skill_dim = skill_dim,
                                    variance = True).to(device)
    sac_model = SACWrapper(obs_dim, action_dim,
                           target_entropy=target_entropy,
                           gamma = gamma).to(device)
    lam = torch.tensor(lam, device = device, requires_grad = False)

    optimizer_policy = torch.optim.Adam(policy.parameters(), lr = lr)
    optimizer_encoder = torch.optim.Adam(state_representer.parameters(), lr = lr)

    buffer = ReplayBuffer(obs_dim,
                          action_dim,
                          special_buffer_dim = skill_dim,
                          capacity = buffer_capacity)
    
    pbar = tqdm(range(n_epochs * steps_per_epoch))
    losses = []

    xy_trajs = []
    dists = []

    for epoch in range(n_epochs):
        counter = 0
        for ep in range(episodes_per_epoch):
            ep_traj, counter = fill_buffer(env, policy, device,
                                           skill_space, buffer, pbar,
                                           counter)
            # record all episodes for given epoch
            if epoch % traj_snapshot_every == 0:
                xy_trajs.append(ep_traj)
            # always record dists
            dist = np.linalg.norm(np.array(ep_traj[0]) - np.array(ep_traj[-1]))
            dists.append(dist)
            
        for _ in range(steps_per_epoch):
            optimizer_encoder.zero_grad()
            optimizer_policy.zero_grad()

            states, actions, rewards, next_states, dones, skills = buffer.sample(batch_size,
                                                                                 device = device)
            
            state_reps = state_representer(states)
            next_state_reps = state_representer(next_states)

            state_diffs = next_state_reps - state_reps

            encoder_loss = -(state_diffs * skills).sum(dim=-1).mean()
            lambda_term = torch.clamp(1 - state_diffs.norm(p = 2, dim = -1), max = eps)
            encoder_loss -= lam * lambda_term.mean()

            policy_reward = (state_diffs * skills).sum(dim=-1, keepdim = True)
            critic_loss, policy_loss = sac_model([states, actions, rewards, next_states, dones, skills],
                                                 policy,
                                                 reward = policy_reward)
            
            encoder_loss.backward()
            policy_loss.backward()

            optimizer_encoder.step()
            with torch.no_grad():
                # gradient descent manually
                lam = lam - 0.1 * lambda_term.mean()
            optimizer_policy.step()

            sac_model.update_targets()

            losses.append([encoder_loss.item(),
                           policy_loss.item()])

            pbar.update(1)
            pbar.set_description(f"Epoch {epoch} | Loss {encoder_loss.item():.2f} | {policy_loss.item():.2f} | lam {lam.item():.2f}")
        if epoch % save_model_every == 0:
            # save models
            torch.save(state_representer.state_dict(),
                    f"tmp/state_representer_{epoch}.pt")
            torch.save(policy.state_dict(),
                    f"tmp/policy_{epoch}.pt")
            torch.save(sac_model.state_dict(),
                       f"tmp/sac_model_{epoch}.pt")
    return np.array(losses), xy_trajs, dists

def continuous_csf(n_epochs = 100,
                   steps_per_epoch = 512,
                   batch_size = 256,
                   buffer_capacity = int(1e4),
                   episodes_per_epoch = 4,
                   gamma = 0.99,
                   skill_dim = 2,
                   env_name = "Ant-v5",
                   traj_snapshot_every = 10,
                   save_model_every = 50):
    """
    Continuous version of the method from this paper:
    https://arxiv.org/abs/2412.08021
    """
    env = make(env_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    skill_space = ContinuousSkillSpace(skill_dim).to(device)
    state_representer = StateRepresenter(obs_dim, skill_dim).to(device)
    policy = SkillConditionedPolicy(obs_dim, action_dim, skill_dim = skill_dim,
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
            new_actions, new_actions_std = policy(states, skill = skills)
            new_next_actions, next_actions_std = policy(next_states, skill = skills)

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
            policy_reward = (successors * skills).sum(dim=-1)
            _, policy_loss = sac_model([states, actions, rewards, next_states, dones, skills],
                                       policy,
                                       reward = policy_reward)
            
            encoder_loss.backward()
            successor_loss.backward(retain_graph = True)
            policy_loss.backward()

            optimizer_encoder.step()
            optimizer_successor.step()
            sac_model.step()
            optimizer_policy.step()
            sac_model.update_targets()

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
    n_epochs = 400
    episodes_per_epoch = 8
    steps_per_epoch = 200
    traj_snapshot_every = n_epochs // 10
    smoothing_num = max(1, int(n_epochs / 10))
    smooth_kernel = np.ones(smoothing_num) / smoothing_num

    losses, trajs, dists = continuous_metra(n_epochs = n_epochs,
                                            steps_per_epoch = steps_per_epoch,
                                            buffer_capacity = int(1e6),
                                            episodes_per_epoch = episodes_per_epoch,
                                            traj_snapshot_every = traj_snapshot_every,
                                            batch_size = 256)

    fig, ax = plt.subplots(1, 3, figsize = (12, 6))
    # plot the losses
    for i, label in enumerate(["Encoder", "Policy"]):
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

    # smooth
    dists_time_np = np.array(dists)
    dists_time_np = np.convolve(dists_time_np,
                                smooth_kernel,
                                mode = "valid")
    ax[2].plot(dists_time_np)

    fig.savefig("tmp/skill_results.png", dpi = 300)