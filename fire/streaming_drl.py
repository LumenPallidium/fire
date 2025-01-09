import os
import torch
from torch.optim import Optimizer
from typing import List, Optional
from torch import Tensor
import gym
import ale_py
from tqdm import tqdm
import numpy as np
from gym.wrappers import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.record_video import RecordVideo
from copy import deepcopy

from utils import NormedSparseMLP, ConvEncoder, entropy

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
    return mean_bar, var, p, n

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
            _, var, self.p, self.n = _running_mean_var(u, 0, self.p, self.n, add_batch = False)
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
            mean, var, self.p, n = _running_mean_var(obs, self.mean, self.p, self.n)
        if update:
            self.mean = mean
            self.var = var
            self.n = n
        scale_obs = (obs - mean) / torch.sqrt(var + 1e-8)
        return scale_obs.view(obs_shape)
    


class ObGD(Optimizer):
    def __init__(self,
                 params,
                 lr=1,
                 momentum=0.2,
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

    def _update_et_momentum(self, momentum, td_error):
        et_sum = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad if group["maximize"] else -p.grad
                eligbility_trace = self.state[p].get("eligibility_trace",
                                                     None)
                if eligbility_trace is None:
                    eligbility_trace = torch.zeros_like(p)
                    self.state[p]["eligibility_trace"] = eligbility_trace
                eligibility_scaler = self.eligibility_trace_decay * self.discount_factor
                eligbility_trace.mul_(eligibility_scaler).add_(grad)
                if momentum != 0:
                    buf = self.state[p].get("momentum_buffer", None)
        
                    if buf is None:
                        buf = ((td_error * eligbility_trace)**2).clone().detach()
                        self.state[p]["momentum_buffer"] = buf
                    else:
                        buf.mul_(momentum).add_((td_error * eligbility_trace)**2,
                                                alpha = 1 - momentum)
                    sqrt_momentum = torch.sqrt(buf + 1e-8)
                else:
                    sqrt_momentum = 1

                et_sum += (eligbility_trace / sqrt_momentum).abs().sum()
        return et_sum


    # mostly following official PyTorch SGD implementation
    def step(self, td_error, reset_eligibility = False):
        grad_scale = getattr(self, "grad_scale", None)
        found_inf = getattr(self, "found_inf", None)
        assert grad_scale is None and found_inf is None
        with torch.no_grad():
            et_sum = self._update_et_momentum(self.defaults["momentum"], td_error)
            for group in self.param_groups:
                lr = group["lr"]
                momentum = group["momentum"]
                for i, p in enumerate(group["params"]):
                    if p.grad is None:
                        continue
                    eligbility_trace = self.state[p].get("eligibility_trace",
                                                         torch.zeros_like(p))
                    if momentum != 0:
                        buf = self.state[p]["momentum_buffer"]
                        sqrt_momentum = torch.sqrt(buf + 1e-8)
                    else:
                        sqrt_momentum = 1

                    et_scaled = eligbility_trace / sqrt_momentum

                    scaled_error = torch.maximum(td_error.abs(), torch.tensor(1)).mean()
                    #TODO : might not be sum here - maybe keep batch
                    momentum_scale = self.scaling_factor * scaled_error * et_sum
                    lr_ = min(lr, lr / (momentum_scale.item() + 1))

                    p.data.add_(et_scaled, alpha=lr_ * td_error.mean().item())
                    if reset_eligibility:
                        self.state[p]["eligibility_trace"] = torch.zeros_like(p)

def create_atari_env(video_dir, episode, name = "ALE/AssaultNoFrameskip-v4"):
    env = gym.make(name, render_mode="rgb_array")
    env = AtariPreprocessing(env, scale_obs=False, grayscale_obs=True, frame_skip=1)
    env = FrameStack(env, 4)
    env = RecordVideo(env,
                      video_folder=video_dir,
                      episode_trigger=lambda x: True,
                      name_prefix=f"episode_{episode}")
    return env

def create_classic_env(video_dir, episode, name = "CartPole-v1"):
    env = gym.make(name, render_mode="rgb_array")
    env = RecordVideo(env,
                      video_folder=video_dir,
                      episode_trigger=lambda x: True,
                      name_prefix=f"episode_{episode}")
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

    optimizer_a = ObGD(network.actor.parameters(),
                       lr=lr,
                       maximize=True,
                       **optimizer_kwargs)
    optimizer_c = ObGD(network.critic.parameters(),
                       lr=lr,
                       maximize=True,
                       **optimizer_kwargs)
    
    gamma = optimizer_a.discount_factor

    while not (done or truncated):
        optimizer_a.zero_grad()
        optimizer_c.zero_grad()

        state_tensor = preprocess_state(state).requires_grad_()
    
        state_value = network.critic(state_tensor.detach().clone().requires_grad_())
        # only letting encoder grad go through actor
        q_values = network.act(state_tensor)
        action = torch.multinomial(q_values, 1).item()

        reward = 0
        for _ in range(steps_per_step):
            next_state, reward_i, done, truncated, _ = env.step(action)
            reward += reward_i

        terminal = 1 if done or truncated else 0
        next_state_tensor = preprocess_state(next_state)

        with torch.no_grad():
            reward_normalized = network.reward_scaler(reward, terminal)
            delta = reward_normalized + (1 - terminal) * gamma * network.critic(next_state_tensor)
            delta -= state_value
            delta_sgn = torch.sign(delta).detach()
        
        # note since we are using eligibility traces in optimizer,
        # loss grad is combined with delta later
        loss_a = torch.log(q_values[0, action] + 1e-8).sum()
        loss_a += (delta_sgn * entropy_coef * entropy(q_values)).sum()

        loss_c = state_value.sum()

        loss_a.backward()
        loss_c.backward()

        optimizer_c.step(delta.mean(), reset_eligibility = terminal)
        optimizer_a.step(delta.mean(), reset_eligibility = terminal)
        
        state = next_state
        total_reward += reward
        counter += steps_per_step
        pbar.set_description(f"Ep:{pbar.n} |S:{counter} |R:{total_reward:.2f}")

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

    optimizer = ObGD(list(network.parameters()),
                     lr=lr,
                     maximize=True,
                     **optimizer_kwargs)
    gamma = optimizer.discount_factor

    while not (done or truncated):
        optimizer.zero_grad()

        state_tensor = preprocess_state(state).requires_grad_()
        inputs = network.encoder(state_tensor)

        state_action_values = network(inputs)
        default_action = torch.argmax(state_action_values).item()
        greedy = np.random.random() < epsilon
        if greedy:
            action = torch.randint(0, state_action_values.shape[-1], (1,)).item()
            if action == default_action:
                greedy = False
        else:
            action = default_action

        reward = 0
        for _ in range(steps_per_step):
            next_state, reward_i, done, truncated, _ = env.step(action)
            reward += reward_i
            terminal = 1 if done or truncated else 0
            if terminal:
                break

        with torch.no_grad():
            next_state_tensor = preprocess_state(next_state)
            next_inputs = network.encoder(next_state_tensor)
            next_state_action_values = network(next_inputs) * (1 - terminal)

            reward_normalized = network.reward_scaler(reward, terminal)
            delta = reward_normalized + gamma * torch.max(next_state_action_values, dim=-1).values
            delta -= state_action_values[:, action]

        loss = state_action_values[:, action].sum()

        loss.backward()
        optimizer.step(delta.mean(),
                       reset_eligibility = (terminal or not greedy))

        state = next_state
        total_reward += reward
        counter += steps_per_step
        pbar.set_description(f"Ep:{pbar.n} |S:{counter} |R:{total_reward:.2f} |E:{epsilon:.2f}")

    pbar.update(1)

    return total_reward, counter

def train(network,
          num_episodes,
          env_name,
          episode_function,
          video_dir = "tmp/",
          epsilon_start=1.0,
          epsilon_final=0.01,
          epsilon_decay=0.9,
          **episode_fn_kwargs):
    os.makedirs(video_dir, exist_ok=True)
    epsilon = epsilon_start
    rewards = []
    pbar = tqdm(range(num_episodes))
    counter = 0
    for episode in range(num_episodes):
        env = create_env(video_dir, episode, env_name)
        total_reward, counter = episode_function(env, network, epsilon,
                                                 pbar, counter,
                                                 **episode_fn_kwargs)
        epsilon = max(epsilon_final, epsilon * epsilon_decay)
        env.close()
        rewards.append(total_reward)
    return rewards


class ActorCritic(torch.nn.Module):
    def __init__(self,
                 encoder_out_size,
                 action_dim,
                 encoder_type = "conv",
                 actor_dims = [256, 128],
                 critic_dims = [256, 128],):
        super().__init__()
        if encoder_type == "conv":
            encoder_a = torch.nn.Sequential(
                ConvEncoder(4),
                ObsScaler(encoder_out_size))
            encoder_c = deepcopy(encoder_a)
        else:
            encoder_a = ObsScaler(encoder_out_size)
            encoder_c = deepcopy(encoder_a)
        self.actor = torch.nn.Sequential(
            encoder_a,
            NormedSparseMLP([encoder_out_size] + actor_dims + [action_dim])
        )
        self.critic = torch.nn.Sequential(
            encoder_c,
            NormedSparseMLP([encoder_out_size] + critic_dims + [1]))
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
    encoder_out_size = 64 * 7 * 7
    env_name = "AssaultNoFrameskip-v4"
    encoder_type = "conv"
    q_dims = [256, 128]
    steps_per_step = 4
    # encoder_out_size = 4
    # env_name = "CartPole-v1"
    # encoder_type = "classic"
    # steps_per_step = 1
    num_episodes = 50
    env = create_env("tmp/", 0, env_name)
    action_num = env.action_space.n

    net = ActorCritic(encoder_out_size, action_num, encoder_type,
    actor_dims = q_dims, critic_dims = q_dims)

    # do a dummy pass to initialize the network
    with torch.no_grad():
        state = preprocess_state(env.reset()[0])
        net.act(state)
        net.critic(state)

    rewards = train(net, num_episodes, env_name, stream_ac,
                    steps_per_step = steps_per_step)

    # net = QNet(encoder_out_size, action_num, encoder_type, q_dims = q_dims)

    # # do a dummy pass to initialize the network
    # state = preprocess_state(env.reset()[0])
    # inputs = net.encoder(state)
    # net(inputs)

    # rewards = train(net, num_episodes, env_name, stream_q)

    smooth_rewards = np.convolve(rewards, np.ones(10) / 10, mode="valid")
    plt.plot(smooth_rewards)