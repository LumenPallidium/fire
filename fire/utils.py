import torch
import math
import numpy as np
from copy import deepcopy

class ReplayBuffer:
    """
    A basic replay buffer.
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 special_buffer_dim = None,
                 capacity = int(1e4),
                 min_capacity = 1000):
        self.capacity = capacity
        if min_capacity < 1:
            min_capacity = int(capacity * min_capacity)
        self.min_capacity = min_capacity

        self.state_buffer = np.zeros((capacity, state_dim), dtype = np.float32)
        self.action_buffer = np.zeros((capacity, action_dim), dtype = np.float32)
        self.reward_buffer = np.zeros((capacity, 1), dtype = np.float32)
        self.next_state_buffer = np.zeros((capacity, state_dim), dtype = np.float32)
        self.done_buffer = np.zeros((capacity, 1), dtype = np.float32)

        self.special_buffer = None
        if special_buffer_dim is not None:
            self.special_buffer = np.zeros((capacity, special_buffer_dim), dtype = np.float32)

        self.pos = 0
        self.total = 0

    def push(self, state, action, next_state, done, reward = torch.nan, special = None):
        # reset the position, making this fifo
        if self.pos >= self.capacity:
            self.pos = 0

        self.state_buffer[self.pos] = state
        self.action_buffer[self.pos] = action.detach().cpu().numpy()
        self.reward_buffer[self.pos] = reward
        self.next_state_buffer[self.pos] = next_state
        self.done_buffer[self.pos] = done

        if (self.special_buffer is not None) and (special is not None):
            self.special_buffer[self.pos] = special.detach().cpu().numpy()

        self.pos += 1
        self.total += 1

    def sample(self, batch_size = 256, device = None):
        if self.total < batch_size:
            batch_size = self.total
        if self.total < self.capacity:
            idx = np.random.randint(0, self.total, batch_size)
        else:
            idx = np.random.randint(0, self.capacity, batch_size)
        return (torch.tensor(self.state_buffer[idx], device = device),
                torch.tensor(self.action_buffer[idx], device = device),
                torch.tensor(self.reward_buffer[idx], device = device),
                torch.tensor(self.next_state_buffer[idx], device = device),
                torch.tensor(self.done_buffer[idx], device = device),
                torch.tensor(self.special_buffer[idx], device = device) if self.special_buffer is not None else None)
    
class ConvEncoder(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 activation = torch.nn.LeakyReLU()):
        super(ConvEncoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = torch.functional.F.layer_norm(x, x.shape[1:])
        x = self.activation(self.conv2(x))
        x = torch.functional.F.layer_norm(x, x.shape[1:])
        x = self.activation(self.conv3(x))
        return x.view(x.size(0), -1)

def entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

def custom_sparse_init(tensor, sparsity = 0.1):
    with torch.no_grad():
        fan_in = tensor.size(1)
        scale = 1 / math.sqrt(fan_in)
        tensor.data.uniform_(-1, 1).mul_(scale)
        mask = torch.rand_like(tensor) > sparsity
        tensor.data *= mask

class SparseLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, sparsity = 0.9):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, dtype = torch.float32))
        self.bias = torch.nn.Parameter(torch.empty(out_features, dtype = torch.float32))
        self.reset_parameters(sparsity)

    def reset_parameters(self, sparsity):
        custom_sparse_init(self.weight, sparsity)
        self.bias.data.zero_()

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)
    
class NormedSparseMLP(torch.nn.Module):
    def __init__(self,
                 dims,
                 sparsity = 0.9,
                 activation = torch.nn.LeakyReLU()):
        super().__init__()
        self.in_dim = dims[0]
        self.out_dim = dims[-1]
        self.sparsity = sparsity

        self.layers = []
        for i in range(1, len(dims)):
            self.layers.append(SparseLinear(dims[i-1], dims[i], sparsity))
            self.layers.append(torch.nn.LayerNorm(dims[i], elementwise_affine = False))
            if i < len(dims) - 1:
                self.layers.append(activation)

        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class SkillConditionedPolicy(torch.nn.Module):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 skill_dim = None,
                 discrete = False,
                 variance = False,
                 dims = [256, 128],
                 sparsity = 0.2):
        super().__init__()
        self.obs_dim = obs_dim
        if skill_dim is None:
            skill_dim = 0
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

    def forward(self, x, skill = None):
        # paper doesn't say how both inputs layout, assume concat
        if self.skill_dim != 0:
            x = torch.cat([x, skill], dim = -1)
        x = self.actor_embed(x)
        if self.discrete:
            return self.softmax(self.actor(x))
        elif self.variance:
            return self.actor(x), self.logstd(x)
        return self.actor(x)
    
    def stochastic_sample(self, obs, skill = None):
        assert self.variance
        mu, logstd = self.forward(obs, skill = skill)
        std = torch.exp(logstd)

        normal = torch.distributions.Normal(mu, std)
        eps = normal.sample()
        log_prob = normal.log_prob(eps).sum(dim = -1)
        return mu + std * eps, log_prob

class DoubleQNetwork(torch.nn.Module):
    """
    A pair of Q-networks. Putting them in a single module because it makes
    things clean and they are inseparable anyway.
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dims = [256, 256],
                 reward_dim = 1,
                 activation = torch.nn.LeakyReLU()):
        super(DoubleQNetwork, self).__init__()
        self.q1 = NormedSparseMLP([state_dim + action_dim] + hidden_dims + [reward_dim],
                                  activation=activation)
        self.q2 = NormedSparseMLP([state_dim + action_dim] + hidden_dims + [reward_dim],
                                  activation=activation)

    def forward(self, state, action):
        x = torch.cat([state, action], dim = -1)
        return self.q1(x), self.q2(x)
    
    def ema(self, other, alpha = 0.995):
        for p, q in zip(self.parameters(), other.parameters()):
            p.data = alpha * p.data + (1 - alpha) * q.data
    
def sac_step(sample, policy, critics,
             gamma = 0.99, alpha = 0.01,
             override_rewards = None):
    """
    A single step of Soft Actor Critic. Minimalist version based on CleanRL.
    """
    states, actions, rewards, next_states, dones, skills = sample

    if override_rewards is not None:
        rewards = override_rewards.detach()

    # get the q values
    with torch.no_grad():
        next_action, next_log_pi = policy.stochastic_sample(next_states, skills)
        target_q1, target_q2 = critics(next_states, next_action)
        target_q = torch.min(target_q1, target_q2)
        target_q = rewards + gamma * (1 - dones) * (target_q - alpha * next_log_pi[:, None])

    q1, q2 = critics(states, actions)
    critic_loss = torch.nn.functional.mse_loss(q1, target_q) + torch.nn.functional.mse_loss(q2, target_q)

    # update policy
    new_action, log_pi = policy.stochastic_sample(states, skills)
    q1_new, q2_new = critics(states, new_action)
    q_new = torch.min(q1_new, q2_new)

    policy_loss = (alpha * log_pi - q_new).mean()

    return critic_loss, policy_loss

class SACWrapper(torch.nn.Module):
    """
    Convenience wrapper so we can do most SAC steps in a single call and
    not worry about the critics and targers.
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 critic_dims = [256, 256],
                 reward_dim = 1,
                 activation = torch.nn.LeakyReLU(),
                 lr = 1e-4,
                 gamma = 0.99,
                 alpha = 0.995):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.alpha = alpha

        self.critics = DoubleQNetwork(state_dim, action_dim,
                                      hidden_dims = critic_dims,
                                      reward_dim = reward_dim,
                                      activation = activation)
        self.targets = deepcopy(self.critics).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.critics.parameters(), lr = lr)

    def forward(self, sample, policy, reward = None, train = True):
        critic_loss, policy_loss = sac_step(sample, policy, self.critics,
                                            override_rewards = reward,
                                            gamma = self.gamma, alpha = self.alpha)
        if train:
            # backward here cause i don't want to worry about it otherwise
            critic_loss.backward(retain_graph = True)

        return critic_loss, policy_loss
    
    def step(self):
        self.optimizer.step()
        self.targets.ema(self.critics)
                                      

if __name__ == "__main__":
    # testing SAC here
    import gym
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    env_name = "LunarLanderContinuous-v2"
    n_epochs = 1000
    steps_per_epoch = 16
    batch_size = 256
    episodes_per_epoch = 4

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = SkillConditionedPolicy(state_dim, action_dim, variance = True).to(device)
    critics = SACWrapper(state_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr = 1e-4)

    buffer = ReplayBuffer(state_dim, action_dim, capacity = 100000)

    pbar = tqdm(total = n_epochs * steps_per_epoch + episodes_per_epoch * n_epochs)
    losses = []

    for epoch in range(n_epochs):
        counter = 0
        for ep in range(episodes_per_epoch):
            state, _ = env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    action, action_std = policy.stochastic_sample(torch.tensor(state, dtype = torch.float32, device = device))
                action += action_std.exp() * torch.randn_like(action)
                next_state, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
                buffer.push(state, action, next_state, done, reward)
                state = next_state
                counter += 1
                done = terminated or truncated
                pbar.set_description(f"Epoch {epoch} | B{ep} | E{counter}")
            pbar.update(1)
        for _ in range(steps_per_epoch):
            optimizer.zero_grad()

            states, actions, rewards, next_states, dones, skills = buffer.sample(batch_size,
                                                                                 device = device)
            
            critic_loss, policy_loss = critics((states, actions, rewards, next_states, dones, skills),
                                               policy, train = True)
            
            critic_loss.backward(retain_graph = True)
            policy_loss.backward()

            critics.step()
            optimizer.step()

            losses.append([critic_loss.item(), policy_loss.item(), rewards.mean().item()])

            pbar.update(1)
            pbar.set_description(f"Epoch {epoch} | {critic_loss.item():.2f} | {policy_loss.item():.2f}")
    
    pbar.close()
    losses = np.array(losses)
    fig, ax = plt.subplots(3, 1, figsize = (12, 8))
    smooth_losses = np.apply_along_axis(lambda x: np.convolve(x, np.ones(5) / 5, mode = "valid"),
                                        axis = 0, arr = losses)
    ax[0].plot(smooth_losses[:, 0], label = "Critic Loss")
    ax[1].plot(smooth_losses[:, 1], label = "Policy Loss")
    ax[2].plot(smooth_losses[:, 2], label = "Reward")