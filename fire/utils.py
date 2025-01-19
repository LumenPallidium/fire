import torch
import math
import numpy as np
from copy import deepcopy

def sample_geometric(gamma, n = 1, max_len = 100):
    dist = torch.distributions.Geometric(probs = 1 - gamma)
    sample = dist.sample((n,))
    sample = torch.clamp(sample, 0, max_len)
    if n == 1:
        sample = sample.item()
    return sample

def find_episode_boundaries(episodes):
    """
    Find the boundaries of episodes in a batch of episodes, which has
    shape (batch, trajectory length).
    """
    # get differences between episode indices
    diff = episodes[:, 1:] != episodes[:, :-1]
    first_diff = diff.argmax(axis=1)
    
    # spots where there may be a change
    has_transition = diff.any(axis=1)
    # for each element in the batch, the episode length
    valid_lengths = np.where(has_transition, first_diff + 1, episodes.shape[1])
    
    return valid_lengths

class ReplayBuffer:
    """
    A basic replay buffer.
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 special_buffer_dim = None,
                 return_trajectory = False,
                 capacity = int(1e6),
                 min_capacity = 1000):
        self.capacity = capacity
        if min_capacity < 1:
            min_capacity = int(capacity * min_capacity)
        self.min_capacity = min_capacity
        self.return_trajectory = return_trajectory

        self.state_buffer = np.zeros((capacity, state_dim), dtype = np.float32)
        self.action_buffer = np.zeros((capacity, action_dim), dtype = np.float32)
        self.reward_buffer = np.zeros((capacity, 1), dtype = np.float32)
        self.next_state_buffer = np.zeros((capacity, state_dim), dtype = np.float32)
        self.done_buffer = np.zeros((capacity, 1), dtype = np.float32)
        self.episode_buffer = np.zeros((capacity, 1), dtype = np.int32)

        self.special_buffer = None
        if special_buffer_dim is not None:
            self.special_buffer = np.zeros((capacity, special_buffer_dim), dtype = np.float32)

        self.pos = 0
        self.total = 0

    def push(self, state, action, next_state, done,
             episode = None, reward = torch.nan, special = None):
        # reset the position, making this fifo
        if self.pos >= self.capacity:
            self.pos = 0

        self.state_buffer[self.pos] = state
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        self.action_buffer[self.pos] = action
        self.reward_buffer[self.pos] = reward
        self.next_state_buffer[self.pos] = next_state
        self.done_buffer[self.pos] = done

        if episode is not None:
            self.episode_buffer[self.pos] = episode

        if (self.special_buffer is not None) and (special is not None):
            self.special_buffer[self.pos] = special.detach().cpu().numpy()

        self.pos += 1
        self.total += 1

    def sample(self,
               gamma = 0.99,
               max_traj_length = 100,
               batch_size = 256, device = None):
        if self.return_trajectory:
            traj_length = sample_geometric(gamma,
                                           max_len = max_traj_length)
            traj_length = max(int(traj_length), 1)
        else:
            traj_length = 0
        if self.total < batch_size:
            batch_size = self.total
        if self.total < self.capacity:
            idx = np.random.randint(0, self.total - traj_length,
                                    batch_size)
        else:
            idx = np.random.randint(0, self.capacity - traj_length,
                                    batch_size)
        if self.return_trajectory:
            # if our sample extends multiple episodes, trim it
            if traj_length != 1:
                idx_tmp = idx[:, None] + np.arange(traj_length)
                # clip index if it spans two episodes
                episodes = self.episode_buffer[idx_tmp, 0]
                traj_length = np.min(find_episode_boundaries(episodes))
            idx = idx[:, None] + np.arange(traj_length)
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
    
class SparseMLP(torch.nn.Module):
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
            if i < len(dims) - 1:
                self.layers.append(activation)

        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

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
            self.layers.append(torch.nn.LayerNorm(dims[i - 1], elementwise_affine = False))
            self.layers.append(SparseLinear(dims[i-1], dims[i], sparsity))
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
                 log_std_min_max = (-5, 2),
                 sparsity = 0.2):
        super().__init__()
        self.obs_dim = obs_dim
        if skill_dim is None:
            skill_dim = 0
        self.skill_dim = skill_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.variance = variance
        self.log_std_min_max = log_std_min_max

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
            act = self.actor(x)
            log_std = self.logstd(x)
            log_std = torch.clamp(log_std,
                                  self.log_std_min_max[0],
                                  self.log_std_min_max[1])
            return act, log_std
        return self.actor(x)
    
    def stochastic_sample(self, obs, skill = None):
        assert self.variance
        mu, logstd = self.forward(obs, skill = skill)
        std = torch.exp(logstd)

        normal = torch.distributions.Normal(mu, std)
        action = normal.rsample()

        action_tanh = torch.tanh(action)
        log_prob = normal.log_prob(action)
        log_prob -= torch.log((1 - action_tanh.pow(2)) + 1e-6)
        return action_tanh, log_prob.sum(dim = -1)

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
                 activation = torch.nn.LeakyReLU(),
                 sparsity = 0.2):
        super(DoubleQNetwork, self).__init__()
        self.q1 = NormedSparseMLP([state_dim + action_dim] + hidden_dims + [reward_dim],
                                  sparsity=sparsity,
                                  activation=activation)
        self.q2 = NormedSparseMLP([state_dim + action_dim] + hidden_dims + [reward_dim],
                                  sparsity=sparsity,
                                  activation=activation)

    def forward(self, state, action):
        x = torch.cat([state, action], dim = -1)
        return self.q1(x), self.q2(x)
    
    def ema(self, other, alpha = 0.995):
        # doing the networks separately to be safe
        for p, q in zip(self.q1.parameters(), other.q1.parameters()):
            p.data = alpha * p.data + (1 - alpha) * q.data
        for p, q in zip(self.q2.parameters(), other.q2.parameters()):
            p.data = alpha * p.data + (1 - alpha) * q.data


class SACWrapper(torch.nn.Module):
    """
    Convenience wrapper so we can do most SAC steps in a single call and
    not worry about the critics and targets.
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 critic_dims = [256, 256],
                 reward_dim = 1,
                 activation = torch.nn.LeakyReLU(),
                 lr = 1e-4,
                 gamma = 0.99,
                 alpha = 0.995,
                 log_entropy_alpha = torch.tensor(0),
                 target_entropy = None):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.alpha = alpha
        self.log_entropy_alpha = log_entropy_alpha
        self.entropy_alpha = log_entropy_alpha.exp()
        self.target_entropy = target_entropy
        self.last_log_pi = None

        self.critics = DoubleQNetwork(state_dim, action_dim,
                                      hidden_dims = critic_dims,
                                      reward_dim = reward_dim,
                                      activation = activation,)
        self.targets = deepcopy(self.critics).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.critics.parameters(), lr = lr)

        if target_entropy is not None:
            self.log_entropy_alpha = torch.nn.Parameter(torch.tensor(0.0))
            self.entropy_optimizer = torch.optim.Adam([self.log_entropy_alpha], lr = lr)

    def sac_step(self, sample, policy,
                override_rewards = None,
                train_critic = True):
        """
        A single step of Soft Actor Critic. Minimalist version based on CleanRL.
        """
        states, actions, rewards, next_states, dones, skills = sample

        if override_rewards is not None:
            rewards = override_rewards.detach()

        # get the q values
        with torch.no_grad():
            next_action, next_log_pi = policy.stochastic_sample(next_states,
                                                                skill = skills)
            target_q1, target_q2 = self.critics(next_states, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * (1 - dones) * (target_q - self.entropy_alpha * next_log_pi[:, None])

        q1, q2 = self.critics(states, actions)
        critic_loss = torch.nn.functional.mse_loss(q1, target_q) + torch.nn.functional.mse_loss(q2, target_q)
        self.last_log_pi = next_log_pi.detach().mean()
        if train_critic:
            self.optimizer.zero_grad()
            critic_loss.backward()
            self.step()
            
        # update policy
        new_action, log_pi = policy.stochastic_sample(states, skills)
        q1_new, q2_new = self.critics(states, new_action)
        q_new = torch.min(q1_new, q2_new)

        policy_loss = (self.entropy_alpha * log_pi - q_new).mean()

        return critic_loss, policy_loss

    def forward(self, sample, policy, reward = None, train = True):

        critic_loss, policy_loss = self.sac_step(sample, policy, 
                                                 override_rewards = reward,
                                                 train_critic = train)

        return critic_loss, policy_loss
    
    def step(self):
        self.optimizer.step()
        if self.target_entropy is not None:
            self.entropy_optimizer.zero_grad()
            self.entropy_alpha = self.log_entropy_alpha.detach().exp()
            entropy_loss = -(self.log_entropy_alpha * (self.last_log_pi + self.target_entropy).detach()).mean()
            entropy_loss.backward()
            self.entropy_optimizer.step()
    
    def update_targets(self):
        self.targets.ema(self.critics, alpha = self.alpha)

class IntScheduler:
    def __init__(self, start, end, n_steps = 100):
        self.start = start
        self.end = end
        self.n_steps = n_steps
        self.slope = (end - start) / n_steps

    def __call__(self, step):
        if step >= self.n_steps:
            return self.end
        val = self.start + self.slope * step
        return int(val)
                                      

if __name__ == "__main__":
    # testing SAC here
    import gym
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    env_name = "LunarLanderContinuous-v2"
    n_epochs = 1500 # 3000 epochs at 8x8 steps/episodes per epoch ~~ 1.5hrs on RTX 3090
    steps_per_epoch = 1
    batch_size = 256
    episodes_per_epoch = 1
    train_start_steps = 10000
    time_penalty = -2e-2
    penalty_start = 200

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    targt_entropy = -action_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = SkillConditionedPolicy(state_dim, action_dim,
                                    variance = True).to(device)
    critics = SACWrapper(state_dim, action_dim,
                         target_entropy = targt_entropy).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr = 1e-4)

    buffer = ReplayBuffer(state_dim, action_dim, capacity = 1000000)

    total_steps = 0
    pbar = tqdm(total = n_epochs * steps_per_epoch + episodes_per_epoch * n_epochs)
    losses = []
    total_rewards = []
    running_reward = 0

    for epoch in range(n_epochs):
        counter = 0
        for ep in range(episodes_per_epoch):
            policy.eval()
            state, _ = env.reset()
            done = False
            total_reward = 0
            step = 0
            while not done:
                with torch.no_grad():
                    action, action_log_probs = policy.stochastic_sample(torch.tensor(state,
                                                                                     dtype = torch.float32,
                                                                                     device = device))
                next_state, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
                if step > penalty_start:
                    reward += time_penalty
                buffer.push(state, action, next_state, done, reward)
                total_reward += reward
                state = next_state

                counter += 1
                total_steps += 1
                step += 1

                done = (terminated or truncated)
                pbar.set_description(f"Epoch {epoch} | R{running_reward:.2f} | S{total_steps}")
            total_rewards.append(total_reward)
            running_reward = 0.01 * total_reward + (1 - 0.01) * running_reward
            if total_steps > train_start_steps:
                pbar.update(1)
        if total_steps > train_start_steps:
            policy.train()
            for _ in range(steps_per_epoch):
                states, actions, rewards, next_states, dones, skills = buffer.sample(batch_size,
                                                                                    device = device)
                
                critic_loss, policy_loss = critics((states, actions, rewards, next_states, dones, skills),
                                                policy, train = True)
                
                optimizer.zero_grad()    
                policy_loss.backward()        
                optimizer.step()

                critics.update_targets()

                losses.append([critic_loss.item(), policy_loss.item()])

                pbar.update(1)
                pbar.set_description(f"Epoch {epoch} |R {running_reward:.2f}| {critic_loss.item():.2f} | {policy_loss.item():.2f}")
    
    pbar.close()
    losses = np.array(losses)

    fig, ax = plt.subplots(3, 1, figsize = (12, 8))
    smooth_losses = np.apply_along_axis(lambda x: np.convolve(x, np.ones(500) / 500, mode = "valid"),
                                        axis = 0, arr = losses)
    ax[0].plot(smooth_losses[:, 0])
    ax[0].set_title("Critic Loss")
    ax[1].plot(smooth_losses[:, 1])
    ax[1].set_title("Policy Loss")

    total_rewards = np.array(total_rewards)
    smooth_rewards = np.convolve(total_rewards.squeeze(),
                                 np.ones(100) / 100, mode = "valid")

    ax[2].plot(total_rewards)
    ax[2].set_title("Reward")
    plt.tight_layout()