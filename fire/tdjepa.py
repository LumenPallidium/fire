import torch
from utils import ConvEncoder
from copy import deepcopy

def ortho_loss(x):
    b = x.shape[0]
    xx = torch.einsum("...i,...j->...ij",
                      x, x)
    xx -= torch.eye(xx.shape[-1], device = xx.device)
    xx /= (2 * b * (b - 1))
    return (xx**2).sum() - (1/b) * (x**2).sum()

class MLPEmbed(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        self.ln = torch.nn.LayerNorm(hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, out_dim)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln(x)
        x = self.tanh(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class TDJEncoders(torch.nn.Module):
    def __init__(self,
                 in_size,
                 exp_conv_out_size = None,
                 hidden_dim = 256):
        super().__init__()
        #TODO: make this less sloppy
        if exp_conv_out_size is None:
            first_stage = torch.nn.Linear(in_size, hidden_dim)
        else:
            first_stage = torch.nn.Sequential(
                ConvEncoder(input_channels = in_size),
                torch.nn.Linear(exp_conv_out_size, hidden_dim))
        state_encoder = torch.nn.Sequential(
            first_stage,
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim))
        if exp_conv_out_size is None:
            first_stage_task = torch.nn.Linear(in_size, hidden_dim)
        else:
            first_stage_task = torch.nn.Sequential(
                ConvEncoder(input_channels = in_size),
                torch.nn.Linear(exp_conv_out_size, hidden_dim))
        task_encoder = torch.nn.Sequential(
            first_stage_task,
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim))
        self.state_encoder = state_encoder
        self.task_encoder = task_encoder

    def forward(self, state):
        state_emb = self.state_encoder(state)
        task_emb = self.task_encoder(state)
        return state_emb, task_emb

    def ema(self, other, alpha = 0.995):
        for p, q in zip(self.state_encoder.parameters(), other.state_encoder.parameters()):
            p.data = alpha * p.data + (1 - alpha) * q.data
        for p, q in zip(self.task_encoder.parameters(), other.task_encoder.parameters()):
            p.data = alpha * p.data + (1 - alpha) * q.data

class TDJPredictor(torch.nn.Module):
    def __init__(self,
                 task_dim,
                 action_dim,
                 out_dim = None,
                 hidden_dim = 256):
        super().__init__()
        out_dim = out_dim if out_dim is not None else hidden_dim
        self.task_embed = MLPEmbed(task_dim,
                                   2 * hidden_dim,
                                   hidden_dim)
        self.action_embed = MLPEmbed(action_dim,
                                     2 * hidden_dim,
                                     hidden_dim)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim))

    def forward(self, context, task, action):
        #TODO : check context is done right
        task_e = self.task_embed(task * context)
        # TODO paper says this should be action*context, but that doesn't make sense dimensionally
        action_e = self.action_embed(action)# * context)
        x = task_e * action_e
        pred = self.predictor(x)
        return pred
    
    def ema(self, other, alpha = 0.995):
        for p, q in zip(self.task_embed.parameters(), other.task_embed.parameters()):
            p.data = alpha * p.data + (1 - alpha) * q.data
        for p, q in zip(self.action_embed.parameters(), other.action_embed.parameters()):
            p.data = alpha * p.data + (1 - alpha) * q.data
        for p, q in zip(self.predictor.parameters(), other.predictor.parameters()):
            p.data = alpha * p.data + (1 - alpha) * q.data

class TDJActor(torch.nn.Module):
    def __init__(self,
                 task_dim,
                 state_dim,
                 hidden_dim = 256,
                 out_dim = 10):
        super().__init__()
        self.task_embed = MLPEmbed(task_dim, 2 * hidden_dim, hidden_dim)
        self.state_embed = MLPEmbed(state_dim, 2 * hidden_dim, hidden_dim)
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),)
        
    def forward(self, task, state):
        task_e = self.task_embed(task * state)
        state_e = self.state_embed(state)
        x = task_e * state_e
        act = self.actor(x)
        return act

class TDJEPA(torch.nn.Module):
    def __init__(self,
                 in_size,
                 exp_conv_out_size = None,
                 task_dim = 256,
                 state_dim = 256,
                 action_dim = 256,
                 hidden_dim = 256,
                 gamma = 0.99):
        super().__init__()

        self.gamma = gamma
        self.encoders = TDJEncoders(in_size,
                                    hidden_dim = hidden_dim,
                                    exp_conv_out_size = exp_conv_out_size,)
        self.predictor = TDJPredictor(task_dim,
                                      action_dim = action_dim,
                                      hidden_dim = hidden_dim,
                                      out_dim = task_dim)
        self.predictor_task = TDJPredictor(state_dim,
                                           action_dim = action_dim,
                                           hidden_dim = hidden_dim,
                                           out_dim = state_dim)
        self.actor = TDJActor(task_dim,
                              state_dim,
                              out_dim = action_dim,
                              hidden_dim = hidden_dim)
        self.ema_encoders = deepcopy(self.encoders).requires_grad_(False)
        self.ema_predictor = deepcopy(self.predictor).requires_grad_(False)
        self.ema_predictor_task = deepcopy(self.predictor_task).requires_grad_(False)

        self.optimizer = torch.optim.Adam(list(self.encoders.parameters()) +
                                          list(self.predictor.parameters()) +
                                          list(self.predictor_task.parameters()),
                                          lr = 1e-4)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr = 1e-4)

    def forward(self, state, task):
        state_emb, _ = self.encoders(state = state)
        action = self.actor(task = task,
                            state = state_emb)
        return action
    
    def get_losses(self, state, task, action, next_state, reg_lambda = 1,):
        with torch.no_grad():
            state_emb_tgt, task_emb_tgt = self.ema_encoders(state = next_state)
            alt_action = self.actor(task = task,
                                    state = state_emb_tgt)
            pred_state_target = self.ema_predictor(context = state_emb_tgt,
                                                     task = task,
                                                     action = alt_action)
            pred_task_target = self.ema_predictor_task(context = task_emb_tgt,
                                                       task = task,
                                                       action = alt_action)
        state_emb, task_emb = self.encoders(state = state)
        pred = self.predictor(
            context = state_emb,
            task = task,
            action = action
        )
        pred_task = self.predictor_task(
            context = task_emb,
            task = task,
            action = action
        )
        phi_loss = torch.nn.functional.mse_loss(pred,
                                                self.gamma * pred_state_target + task_emb_tgt)
        psi_loss = torch.nn.functional.mse_loss(pred_task,
                                                self.gamma * pred_task_target + state_emb_tgt)
        loss = 0.5 * (phi_loss + psi_loss)

        loss_reg = reg_lambda * (ortho_loss(state_emb) +
                                 ortho_loss(task_emb))
        loss += loss_reg
        # actor loss
        action_new = self.actor(
            task = task,
            state = state_emb
        )
        pred_new_action = self.predictor(
            context = state_emb,
            task = task,
            action = action_new
        )
        actor_loss = -(pred_new_action * task).mean()
        return loss, actor_loss
    
    def optimizer_steps(self, loss, actor_loss):
        self.optimizer.zero_grad()
        loss.backward(retain_graph = True)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()
        self.actor_optimizer.step()

    def update_ema(self, alpha = 0.995):
        self.ema_encoders.ema(self.encoders, alpha = alpha)
        self.ema_predictor.ema(self.predictor, alpha = alpha)
        self.ema_predictor_task.ema(self.predictor_task, alpha = alpha)

if __name__ == "__main__":
    import gymnasium as gym
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import numpy as np
    from utils import ReplayBuffer
    env_name = "Ant-v5"
    n_epochs = 1500 # 3000 epochs at 8x8 steps/episodes per epoch ~~ 1.5hrs on RTX 3090
    steps_per_epoch = 32
    batch_size = 256
    episodes_per_epoch = 1
    train_start_steps = 10000

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tdjepa = TDJEPA(obs_dim,
                    task_dim = 256,
                    state_dim = 256,
                    action_dim = action_dim,
                    hidden_dim = 256,
                    gamma = 0.999).to(device)

    buffer = ReplayBuffer(obs_dim, action_dim, capacity = int(1e7))

    total_steps = 0
    pbar = tqdm(total = n_epochs * steps_per_epoch + episodes_per_epoch * n_epochs)
    losses = []
    total_rewards = []
    running_reward = 0

    for epoch in range(n_epochs):
        counter = 0
        policy = tdjepa.actor
        for ep in range(episodes_per_epoch):
            policy.eval()
            state, _ = env.reset()
            task = torch.randn(1, 256).to(device)
            done = False
            total_reward = 0
            step = 0
            while not done:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    state_emb, _ = tdjepa.encoders(state = state_tensor)
                    action = tdjepa.actor(task = task,
                                          state = state_emb)
                next_state, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy()[0])
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
        # update steps_per_epoch based on episodes collected
        buffer_size = len(buffer)
        # seems like a nice heuristic
        steps_per_epoch = min(300, max(32, buffer_size // (10 * batch_size)))
        if total_steps > train_start_steps:
            policy.train()
            for _ in range(steps_per_epoch):
                states, actions, rewards, next_states, dones, _ = buffer.sample(batch_size,
                                                                                    device = device)
                #TODO : paper is unclear - should this be sampled based on obs?
                task = torch.randn(batch_size, 256).to(device)
                loss, actor_loss = tdjepa.get_losses(state = states,
                                                     next_state = next_states,
                                                     task = task,
                                                     action = actions)
                tdjepa.optimizer_steps(loss, actor_loss)
                tdjepa.update_ema()

                losses.append([loss.item(), actor_loss.item()])

                pbar.update(1)
                pbar.set_description(f"Epoch {epoch} |R {running_reward:.2f}| {loss.item():.2f} | {actor_loss.item():.2f}")
    
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
    fig.savefig("tmp/sac.png")