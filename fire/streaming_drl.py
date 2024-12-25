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
        with torch.no_grad():
            mean, var, n = _running_mean_var(obs, self.mean, self.p, self.n)
        if update:
            self.mean = mean
            self.var = var
            self.n = n
        return (obs - mean) / torch.sqrt(var + 1e-8)
    

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

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        eligibility_trace = eligbility_traces[i]
        if eligibility_trace is None:
            eligibility_trace = torch.clone(grad).detach().requires_grad_(False)
            eligbility_traces[i] = eligibility_trace
        else:
            eligibility_scaler = eligibility_trace_decay * discount_factor
            eligibility_trace.mul_(eligibility_scaler).add_(grad)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(eligibility_trace).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(eligibility_trace,
                                        alpha = 1 - momentum)

        scaled_error = torch.maximum(error, torch.tensor(1)).mean()
        et_scaled = eligibility_trace / torch.sqrt(torch.tensor(momentum) + 1e-8)
        #TODO : might not be sum here - maybe keep batch
        momentum_scale = scaling_factor * scaled_error * et_scaled.abs().sum()
        lr_ = min(lr, 1 / (momentum_scale.item() + 1e-8))

        param.data.add_(et_scaled, alpha=-lr_ * error.mean().item())


class ObGD(Optimizer):
    def __init__(self,
                 params,
                 lr=1e-3,
                 momentum=0,
                 maximize=False,
                 scaling_factor = 2,
                 discount_factor = 0.99,
                 eligibility_trace_decay = 0.9,):
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
    def step(self, td_error, closure=None):
        loss = None
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
                    eligbility_traces.append(self.state[p].get("eligibility_trace", None))
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

        return loss
    
def create_env(video_dir, episode, name = "AssaultNoFrameskip-v4"):
    env = gym.make(name, render_mode="rgb_array")
    env = AtariPreprocessing(env, scale_obs=False, grayscale_obs=True, frame_skip=1)
    env = FrameStack(env, 4)
    env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: True, name_prefix=f"episode_{episode}")
    return env

def preprocess_state(state):
    return torch.FloatTensor(np.array(state)).unsqueeze(0) / 255.0

def stream_ac(env,
              network,
              epsilon,
              pbar,
              counter,
              lr = 1e-3,
              entropy_coef = 0.01,
              **optimizer_kwargs):
    state, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0

    optimizer_a = ObGD(list(network.actor.parameters()),
                       lr=lr,
                       maximize=True,
                       **optimizer_kwargs)
    optimizer_c = ObGD(list(network.critic.parameters()) + list(network.encoder.parameters()),
                       lr=lr,
                       maximize=True,
                       **optimizer_kwargs)
    
    gamma = optimizer_a.discount_factor

    while not (done or truncated):
        optimizer_a.zero_grad()
        optimizer_c.zero_grad()

        state_tensor = preprocess_state(state)
    
        inputs = network.encoder(state_tensor)

        state_value = network.critic(inputs)
        # only letting encoder grad go through critic
        q_values = network.act(inputs.detach().requires_grad_())
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax(q_values).item()

        next_state, reward, done, truncated, _ = env.step(action)
        next_state_tensor = preprocess_state(next_state)
        next_inputs = network.encoder(next_state_tensor)

        terminal = 1 if done or truncated else 0
        reward_normalized = network.reward_scaler(reward, terminal)
        delta = reward_normalized + gamma * network.critic(next_inputs)
        delta -= state_value
        delta_sgn = torch.sign(delta).detach()
        
        # note since we are using eligibility traces in optimizer,
        # loss grad is combined with delta later
        loss_a = torch.log(q_values[0, action]).sum()
        loss_a += (delta_sgn * entropy_coef * entropy(q_values)).sum()

        loss_c = state_value.sum()

        loss_a.backward()
        loss_c.backward()

        optimizer_a.step(delta)
        optimizer_c.step(delta)
        
        state = next_state
        total_reward += reward
        counter += 1
        pbar.set_description(f"Steps: {counter}, Reward: {total_reward}")

    pbar.update(1)

    return total_reward, counter

def train(network,
          num_episodes,
          episode_function,
          video_dir = "tmp/",
          epsilon_start=1.0,
          epsilon_final=0.01,
          epsilon_decay=0.95,):
    os.makedirs(video_dir, exist_ok=True)
    epsilon = epsilon_start
    rewards = []
    pbar = tqdm(range(num_episodes))
    counter = 0
    for episode in range(num_episodes):
        env = create_env(video_dir, episode)
        total_reward, counter = episode_function(env, network, epsilon, pbar, counter)
        epsilon = max(epsilon_final, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
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
                 actor_dims = [256, 128],
                 critic_dims = [256, 128],):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            ConvEncoder(4),
            torch.nn.LayerNorm(encoder_out_size))
        self.actor = NormedSparseMLP([encoder_out_size] + actor_dims + [action_dim])
        self.critic = NormedSparseMLP([encoder_out_size] + critic_dims + [1])
        self.reward_scaler = RewardScaler(0.99)

    def act(self, x):
        return torch.nn.functional.softmax(self.actor(x), dim=-1)
    
    def forward(self, x):
        inputs = self.encoder(x)
        return self.actor(inputs), self.critic(inputs)


if __name__ == "__main__":
    encoder_out_size = 64 * 7 * 7
    env_name = "AssaultNoFrameskip-v4"
    num_episodes = 50
    env = create_env("tmp/", 0, env_name)
    action_num = env.action_space.n

    net = ActorCritic(encoder_out_size, action_num)

    # do a dummy pass to initialize the network
    state = preprocess_state(env.reset()[0])
    inputs = net.encoder(state)
    net.act(inputs)
    net.critic(inputs)

    rewards = train(net, num_episodes, stream_ac)