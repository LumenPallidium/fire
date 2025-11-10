import torch
from utils import ConvEncoder
from copy import deepcopy

def ortho_loss(x):
    b = x.shape[0]
    xx = torch.einsum("...i,...j->...ij",
                      x, x)
    # set diagonal to zero
    diag = torch.eye(xx.shape[-1], device=xx.device).unsqueeze(0)
    xx = (xx**2) * (1 - diag)
    xx /= (2 * b * (b - 1))
    return xx.sum() - (1/b) * (x**2).sum()

class SphereNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=self.dim, keepdim=True)
        return x / (norm + 1e-8)

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
                 state_dim,
                 task_dim,
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
            torch.nn.Linear(hidden_dim, state_dim),
            SphereNorm(dim = 1))
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
            torch.nn.Linear(hidden_dim, task_dim),
            SphereNorm(dim=1))
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
                 state_dim,
                 action_dim,
                 out_dim = None,
                 hidden_dim = 256):
        super().__init__()
        out_dim = out_dim if out_dim is not None else hidden_dim
        self.task_embed = MLPEmbed(task_dim + state_dim,
                                   hidden_dim,
                                   hidden_dim)
        self.action_embed = MLPEmbed(action_dim + state_dim,
                                     hidden_dim,
                                     hidden_dim)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim))

    def forward(self, context, task, action):
        task_context = torch.cat([task, context], dim = -1)
        task_e = self.task_embed(task_context)
        action_context = torch.cat([action, context], dim = -1)
        action_e = self.action_embed(action_context)
        x = torch.cat([task_e, action_e], dim = -1)
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
                 out_dim = 10,
                 noise_scale = 0.2,
                 out_activation = torch.nn.Tanh()):
        super().__init__()
        self.noise_scale = noise_scale
        self.task_embed = MLPEmbed(task_dim + state_dim,
                                   hidden_dim, hidden_dim)
        self.state_embed = MLPEmbed(state_dim,
                                    hidden_dim, hidden_dim)
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim))
        self.out_activation = out_activation
        
    def forward(self, task, state):
        task_state = torch.cat([task, state], dim = -1)
        task_e = self.task_embed(task_state)
        state_e = self.state_embed(state)
        x = torch.cat([task_e, state_e], dim = -1)
        act = self.actor(x)
        if self.noise_scale > 0:
            noise = torch.randn_like(act) * self.noise_scale
            act = act + noise
        act = self.out_activation(act)
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
                                    state_dim = state_dim,
                                    task_dim = task_dim,
                                    hidden_dim = hidden_dim,
                                    exp_conv_out_size = exp_conv_out_size,)
        self.predictor = TDJPredictor(task_dim,
                                      state_dim = state_dim,
                                      action_dim = action_dim,
                                      hidden_dim = hidden_dim,
                                      out_dim = task_dim)
        self.predictor_task = TDJPredictor(task_dim,
                                           state_dim = task_dim,
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

        self.optimizer_enc = torch.optim.Adam(self.encoders.parameters(),
                                              lr = 1e-4)
        self.optimizer_pred = torch.optim.Adam(
                                          list(self.predictor.parameters()) +
                                          list(self.predictor_task.parameters()),
                                          lr = 2e-4)
        self.optimizer_act = torch.optim.Adam(self.actor.parameters(),
                                                lr = 1e-4)

    def forward(self, state, task):
        state_emb, _ = self.encoders(state = state)
        action = self.actor(task = task,
                            state = state_emb)
        return action
    
    def get_losses(self, state, task, action, next_state, reg_lambda = 1,):
        with torch.no_grad():
            state_emb_target, task_emb_target = self.ema_encoders(state = next_state)
            alt_action = self.actor(task = task,
                                    state = state_emb_target)
            pred_state_target = self.ema_predictor(context = state_emb_target,
                                                   task = task,
                                                   action = alt_action)
            pred_task_target = self.ema_predictor_task(context = task_emb_target,
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
                                                self.gamma * pred_state_target + task_emb_target)
        psi_loss = torch.nn.functional.mse_loss(pred_task,
                                                self.gamma * pred_task_target + state_emb_target)
        loss = 0.5 * (phi_loss + psi_loss)

        loss_reg = reg_lambda * (ortho_loss(state_emb) +
                                 ortho_loss(task_emb))
        loss += loss_reg
        # actor loss
        action_new = self.actor(
            task = task,
            state = state_emb.detach()
        )
        pred_new_action = self.predictor(
            context = state_emb.detach(),
            task = task,
            action = action_new
        )
        actor_loss = -(pred_new_action * task).sum(dim = -1).mean()
        return loss, actor_loss
    
    def optimizer_steps(self, loss, actor_loss,
                        clip_grad = 1.0):
        self.optimizer_enc.zero_grad()
        self.optimizer_pred.zero_grad()
        self.optimizer_act.zero_grad()
        loss.backward(retain_graph = True)
        actor_loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.encoders.parameters(),
                                           clip_grad)
            torch.nn.utils.clip_grad_norm_(list(self.predictor.parameters()) +
                                           list(self.predictor_task.parameters()),
                                           clip_grad)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                           clip_grad)
        self.optimizer_enc.step()
        self.optimizer_pred.step()
        self.optimizer_act.step()

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

    torch.manual_seed(333)
    np.random.seed(333)

    env_name = "Ant-v5"
    n_epochs = 500 # 3000 epochs at 8x8 steps/episodes per epoch ~~ 1.5hrs on RTX 3090
    steps_per_epoch = 4
    batch_size = 256
    task_dim = 50
    episodes_per_epoch = 1
    train_start_steps = 10000
    gamma = 0.5

    env = gym.make(env_name,
                   forward_reward_weight = 0.0,)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #TODO : see how they did task dim given multiplication
    tdjepa = TDJEPA(obs_dim,
                    task_dim = task_dim,
                    state_dim = 256,
                    action_dim = action_dim,
                    hidden_dim = 256,
                    gamma = gamma).to(device)

    buffer = ReplayBuffer(obs_dim, action_dim,
                          special_buffer_dim=task_dim,
                          capacity = int(1e7))

    total_steps = 0
    #TODO : this is not right
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

            task = torch.randn(1, task_dim).to(device)
            task = task / (torch.norm(task, p=2, dim=1, keepdim=True) + 1e-8)

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
                buffer.push(state, action, next_state, done, reward,
                            special = task)
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
        steps_per_epoch = min(100, max(4, buffer_size // (100 * batch_size)))
        if total_steps > train_start_steps:
            policy.train()
            for _ in range(steps_per_epoch):
                states, actions, rewards, next_states, dones, task = buffer.sample(batch_size,
                                                                                    device = device)
                rand_task = torch.randn(batch_size, task_dim).to(device)
                rand_task = rand_task / (torch.norm(rand_task, p=2, dim=1, keepdim=True) + 1e-8)
                mask = torch.rand(batch_size, 1).to(device) < 0.5
                task = torch.where(mask, task, rand_task)

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