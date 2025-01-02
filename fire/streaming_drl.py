import os
import torch
from torch.optim import Optimizer
from typing import List, Optional
from torch import Tensor
import math
import gymnasium as gym
from tqdm import tqdm
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from gymnasium.wrappers.record_video import RecordVideo

class ConvEncoder(torch.nn.Module):
    def __init__(self, input_channels):
        super(ConvEncoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return x.view(x.size(0), -1)

def entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

def custom_sparse_init(tensor, sparsity = 0.1):
    with torch.no_grad():
        fan_in = tensor.size(1)
        scale = 1 / fan_in
        tensor.data.uniform_(-1, 1).mul_(scale)
        mask = torch.rand_like(tensor) > sparsity
        tensor.data *= mask
        tensor.data /= math.sqrt(fan_in)

class SparseLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, sparsity = 0.9):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.empty(out_features))
        self.reset_parameters(sparsity)

    def reset_parameters(self, sparsity):
        custom_sparse_init(self.weight, sparsity)
        self.bias.data.zero_()

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)

def _running_mean_var(x, mean, p, n, add_batch = True):
    """
    Based on Welford's online algorithm. p is an unscaled
    variance estimator.
    """
    if len(x.shape) == 1 and add_batch:
        x = x.unsqueeze(0)
    n += 1
    mean_bar = mean + (x - mean) / n
    #TODO : could make this a matrix and get covariance
    p += ((x - mean) * (x - mean_bar)).mean(dim = 0)
    var = p / (n-1) if n > 1 else torch.tensor(1)
    return mean_bar, var, n

class RewardScaler(torch.nn.Module):
    def __init__(self, decay):
        super().__init__()
        self.decay = decay
        self.register_buffer("u", torch.zeros(1))
        self.register_buffer("p", torch.zeros(1))
        self.register_buffer("n", torch.zeros(1))

    def forward(self, reward, terminal = 0):
        """
        Scale the reward. terminal is a binary flag indicating
        whether the episode is over.
        """
        with torch.no_grad():
            u = reward + self.decay * (1 - terminal) * self.u
            _, var, self.n = _running_mean_var(u, 0, self.p, self.n, add_batch = False)
            self.u = u
        return reward / torch.sqrt(var + 1e-8)
    
class ObsScaler(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("p", torch.zeros(dim))
        self.register_buffer("n", torch.zeros(1))

    def forward(self, obs, update = True):
        obs_shape = obs.shape
        obs = obs.view(obs_shape[0], -1)
        with torch.no_grad():
            mean, var, n = _running_mean_var(obs, self.mean, self.p, self.n)
        if update:
            self.mean = mean
            self.var = var
            self.n = n
        scale_obs = (obs - mean) / torch.sqrt(var + 1e-8)
        return scale_obs.view(obs_shape)
    

def _obgd(
    params: List[Tensor],
    eligbility_traces : List[Tensor],
    grads: List[Tensor],
    error: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    momentum: float,
    lr: float,
    scaling_factor : float,
    discount_factor: float,
    eligibility_trace_decay: float,
    maximize: bool,
):
    """
    Single step of the (adaptive) Overshooting-bounded Gradient Descent (ObGD) optimizer.
    """
    assert grad_scale is None and found_inf is None
    with torch.no_grad():
        for i, param in enumerate(params):
            grad = grads[i] if maximize else -grads[i]
            eligibility_scaler = eligibility_trace_decay * discount_factor
            eligbility_traces[i].mul_(eligibility_scaler).add_(grad)

            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = ((error * eligbility_traces[i])**2).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_((error * eligbility_traces[i])**2,
                                            alpha = 1 - momentum)
                    
                sqrt_momentum = torch.sqrt(buf + 1e-8)
            else:
                sqrt_momentum = 1

            scaled_error = torch.maximum(error.abs(), torch.tensor(1)).mean()
            et_scaled = eligbility_traces[i] / sqrt_momentum
            #TODO : might not be sum here - maybe keep batch
            momentum_scale = scaling_factor * scaled_error * et_scaled.abs().sum()
            lr_ = min(lr, 1 / (momentum_scale.item() + 1))

            param.data.add_(et_scaled, alpha=lr_ * error.mean().item())

class ObGD(Optimizer):
    def __init__(self,
                 params,
                 lr=1,
                 momentum=0,
                 maximize=False,
                 scaling_factor = 2,
                 discount_factor = 0.99,
                 eligibility_trace_decay = 0.8,):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if scaling_factor <= 1:
            raise ValueError(f"Invalid scaling factor: {scaling_factor}")
        if discount_factor < 0 or discount_factor > 1:
            raise ValueError(f"Invalid discount factor: {discount_factor}")
        if eligibility_trace_decay < 0 or eligibility_trace_decay > 1:
            raise ValueError(f"Invalid eligibility trace decay: {eligibility_trace_decay}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            maximize=maximize,
        )
        super().__init__(params, defaults)
        self.scaling_factor = scaling_factor
        self.discount_factor = discount_factor
        self.eligibility_trace_decay = eligibility_trace_decay

    def __setstate__(self, state):
        super().__setstate__(state)

    # mostly following official PyTorch SGD implementation
    def step(self, td_error, reset_eligibility = False, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            eligbility_traces: List[Tensor] = []
            momentum_buffer_list: List[Optional[Tensor]] = []

            for p in group["params"]:
                if p.grad is not None:
                    params.append(p)
                    grads.append(p.grad)
                    if reset_eligibility:
                        eligbility_traces.append(torch.zeros_like(p))
                    else:
                        eligbility_traces.append(self.state[p].get("eligibility_trace",
                                                                   torch.zeros_like(p)))
                    momentum_buffer_list.append(self.state[p].get("momentum_buffer", None))

            _obgd(
                params,
                eligbility_traces,
                grads,
                td_error,
                momentum_buffer_list,
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                momentum = group["momentum"],
                lr=group["lr"],
                scaling_factor = self.scaling_factor,
                discount_factor = self.discount_factor,
                eligibility_trace_decay = self.eligibility_trace_decay,
                maximize=group["maximize"],
            )

            for i, p in enumerate(group["params"]):
                self.state[p]["eligibility_trace"] = eligbility_traces[i]
                self.state[p]["momentum_buffer"] = momentum_buffer_list[i]
    
def create_atari_env(video_dir, episode, name = "AssaultNoFrameskip-v4"):
    env = gym.make(name, render_mode="rgb_array")
    env = AtariPreprocessing(env, scale_obs=False, grayscale_obs=True, frame_skip=1)
    env = FrameStack(env, 4)
    env = RecordVideo(env,
                      video_folder=video_dir,
                      episode_trigger=lambda x: True,
                      name_prefix=f"episode_{episode}",
                      disable_logger=True)
    return env

def create_classic_env(video_dir, episode, name = "CartPole-v1"):
    env = gym.make(name, render_mode="rgb_array")
    env = RecordVideo(env,
                      video_folder=video_dir,
                      episode_trigger=lambda x: True,
                      name_prefix=f"episode_{episode}",
                      disable_logger=True)
    return env

def create_env(video_dir, episode, name):
    if "NoFrameskip" in name:
        return create_atari_env(video_dir, episode, name)
    else:
        return create_classic_env(video_dir, episode, name)

def preprocess_state(state):
    return torch.FloatTensor(np.array(state)).unsqueeze(0) / 255.0

def stream_ac(env,
              network,
              epsilon,
              pbar,
              counter,
              lr = 1,
              entropy_coef = 0.01,
              steps_per_step = 1,
              **optimizer_kwargs):
    state, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0

    optimizer_a = ObGD(list(network.actor.parameters()) + list(network.encoder.parameters()),
                       lr=lr,
                       maximize=True,
                       **optimizer_kwargs)
    optimizer_c = ObGD(list(network.critic.parameters()),
                       lr=lr,
                       maximize=True,
                       **optimizer_kwargs)
    
    gamma = optimizer_a.discount_factor

    while not (done or truncated):
        optimizer_a.zero_grad()
        optimizer_c.zero_grad()

        state_tensor = preprocess_state(state).requires_grad_()
    
        inputs = network.encoder(state_tensor)

        state_value = network.critic(inputs.detach().clone().requires_grad_())
        # only letting encoder grad go through actor
        q_values = network.act(inputs)
        action = torch.multinomial(q_values, 1)

        reward = 0
        for _ in range(steps_per_step):
            next_state, reward_i, done, truncated, _ = env.step(action)
            reward += reward_i

        terminal = 1 if done or truncated else 0
        next_state_tensor = preprocess_state(next_state)
        next_inputs = network.encoder(next_state_tensor)

        reward_normalized = network.reward_scaler(reward, terminal)
        with torch.no_grad():
            delta = reward_normalized + (1 - terminal) * gamma * network.critic(next_inputs)
            delta -= state_value
        delta_sgn = torch.sign(delta).detach()
        
        # note since we are using eligibility traces in optimizer,
        # loss grad is combined with delta later
        loss_a = torch.log(q_values[0, action] + 1e-8).sum()
        loss_a += (delta_sgn * entropy_coef * entropy(q_values)).sum()

        loss_c = state_value.sum()

        loss_a.backward()
        loss_c.backward()

        optimizer_a.step(delta)
        optimizer_c.step(delta)
        
        state = next_state
        total_reward += reward
        counter += steps_per_step
        pbar.set_description(f"Ep:{pbar.n} |S:{counter} |TD:{delta.mean().item():.2f} |R:{total_reward:.2f}")

    pbar.update(1)

    return total_reward, counter

def stream_q(env,
             network,
             epsilon,
             pbar,
             counter,
             lr = 1,
             steps_per_step = 1,
             **optimizer_kwargs):
    state, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0

    optimizer = ObGD(list(network.q.parameters()),
                     lr=lr,
                     maximize=True,
                     **optimizer_kwargs)
    gamma = optimizer.discount_factor

    while not (done or truncated):
        optimizer.zero_grad()

        state_tensor = preprocess_state(state).requires_grad_()
    
        inputs = network.encoder(state_tensor)

        state_action_values = network(inputs)
        greedy = np.random.random() < epsilon
        if greedy:
            action = torch.randint(0, state_action_values.shape[-1], (1,)).item()
        else:
            action = torch.argmax(state_action_values).item()

        reward = 0
        for _ in range(steps_per_step):
            next_state, reward_i, done, truncated, _ = env.step(action)
            reward += reward_i

        with torch.no_grad():
            terminal = 1 if done or truncated else 0

            next_state_tensor = preprocess_state(next_state)
            next_inputs = network.encoder(next_state_tensor)
            next_state_action_values = network(next_inputs) * (1 - terminal)

            reward_normalized = network.reward_scaler(reward, terminal)
            delta = reward_normalized + gamma * torch.max(next_state_action_values, dim=-1).values
            delta -= state_action_values[:, action]

        loss = state_action_values.sum()

        loss.backward()
        optimizer.step(delta, reset_eligibility = (not greedy))

        state = next_state
        total_reward += reward
        counter += steps_per_step
        pbar.set_description(f"Ep:{pbar.n} |S:{counter}\n|TD:{delta.mean().item():.2f} |R:{total_reward:.2f}\n|E:{epsilon:.2f}")

    pbar.update(1)

    return total_reward, counter

def train(network,
          num_episodes,
          env_name,
          episode_function,
          video_dir = "tmp/",
          epsilon_start=1.0,
          epsilon_final=0.01,
          epsilon_decay=0.99,):
    os.makedirs(video_dir, exist_ok=True)
    epsilon = epsilon_start
    rewards = []
    pbar = tqdm(range(num_episodes))
    counter = 0
    for episode in range(num_episodes):
        env = create_env(video_dir, episode, env_name)
        total_reward, counter = episode_function(env, network, epsilon, pbar, counter)
        epsilon = max(epsilon_final, epsilon * epsilon_decay)
        env.close()
        rewards.append(total_reward)
    return rewards

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
            self.layers.append(activation)

        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ActorCritic(torch.nn.Module):
    def __init__(self,
                 encoder_out_size,
                 action_dim,
                 encoder_type = "conv",
                 actor_dims = [256, 128],
                 critic_dims = [256, 128],):
        super().__init__()
        if encoder_type == "conv":
            self.encoder = torch.nn.Sequential(
                ConvEncoder(4),
                ObsScaler(encoder_out_size))
        else:
            self.encoder = ObsScaler(encoder_out_size)
        self.actor = NormedSparseMLP([encoder_out_size] + actor_dims + [action_dim])
        self.critic = NormedSparseMLP([encoder_out_size] + critic_dims + [1])
        self.reward_scaler = RewardScaler(0.99)

    def act(self, x):
        return torch.nn.functional.softmax(self.actor(x), dim=-1)
    
    def forward(self, x):
        inputs = self.encoder(x)
        return self.actor(inputs), self.critic(inputs)
    
class QNet(torch.nn.Module):
    def __init__(self,
                 encoder_out_size,
                 action_dim,
                 encoder_type = "conv",
                 q_dims = [256, 128],):
        super().__init__()
        if encoder_type == "conv":
            self.encoder = torch.nn.Sequential(
                ConvEncoder(4),
                ObsScaler(encoder_out_size))
        else:
            self.encoder = ObsScaler(encoder_out_size)
        self.q = NormedSparseMLP([encoder_out_size] + q_dims + [action_dim])
        self.reward_scaler = RewardScaler(0.99)

    def forward(self, x):
        return self.q(x)

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    # encoder_out_size = 64 * 7 * 7
    # env_name = "AssaultNoFrameskip-v4"
    encoder_out_size = 4
    q_dims = [16, 32]
    env_name = "CartPole-v1"
    encoder_type = "classic"
    num_episodes = 100
    env = create_env("tmp/", 0, env_name)
    action_num = env.action_space.n

    # net = ActorCritic(encoder_out_size, action_num, encoder_type)

    # # do a dummy pass to initialize the network
    # state = preprocess_state(env.reset()[0])
    # inputs = net.encoder(state)
    # net.act(inputs)
    # net.critic(inputs)

    # rewards = train(net, num_episodes, env_name, stream_ac)

    net = QNet(encoder_out_size, action_num, encoder_type, q_dims = q_dims)

    # do a dummy pass to initialize the network
    state = preprocess_state(env.reset()[0])
    inputs = net.encoder(state)
    net(inputs)

    rewards = train(net, num_episodes, env_name, stream_q)

    smooth_rewards = np.convolve(rewards, np.ones(10) / 10, mode="valid")
    plt.plot(smooth_rewards)