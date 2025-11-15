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

class OjaLinear(torch.nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias = True,
                 activation = torch.nn.Identity(),):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn(in_features, out_features) * (1.0 / in_features**0.5))

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, compute_oja_loss = False):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        out = self.activation(out)
        if compute_oja_loss:
            oja_loss = (x - out @ self.weight.t())**2
            oja_loss = oja_loss.mean()
            return out, oja_loss
        else:
            return out

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
    
class TDJOjaEncoder(torch.nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 n_middle_layers = 3,
                 exp_conv_out_size = None,
                 hidden_dim = 256):
        super().__init__()

        self.conv = ConvEncoder(input_channels = in_size) if exp_conv_out_size is not None else torch.nn.Identity()

        middle_layers = []
        start_dim = exp_conv_out_size if exp_conv_out_size is not None else in_size
        for _ in range(n_middle_layers - 1):
            middle_layers.append(OjaLinear(start_dim, hidden_dim,
                                           activation = torch.nn.ReLU()))
            start_dim = hidden_dim
        middle_layers.append(OjaLinear(start_dim, out_size))

        self.encoder = torch.nn.ModuleList(middle_layers)
        
        self.out_norm = SphereNorm(dim = 1)
    
    def forward(self, state):
        x = self.conv(state)
        total_oja_loss = 0.0
        for layer in self.encoder:
            x, l = layer(x, compute_oja_loss = True)
            total_oja_loss = total_oja_loss + l

        x = self.out_norm(x)
        return x, total_oja_loss
        
class TDJEncoderPairOja(torch.nn.Module):
    def __init__(self,
                 in_size,
                 state_dim,
                 task_dim,
                 state_encoder_extra_depth = 3,
                 exp_conv_out_size = None,
                 hidden_dim = 256):
        super().__init__()
        self.state_encoder = TDJOjaEncoder(in_size,
                                           state_dim,
                                           n_middle_layers = state_encoder_extra_depth,
                                           exp_conv_out_size = exp_conv_out_size,
                                           hidden_dim = hidden_dim)
        self.task_encoder = TDJOjaEncoder(in_size,
                                          task_dim,
                                          n_middle_layers = 2,
                                          exp_conv_out_size = exp_conv_out_size,
                                          hidden_dim = hidden_dim)

    def forward(self, state):
        state_emb, state_oja_loss = self.state_encoder(state)
        task_emb, task_oja_loss = self.task_encoder(state)
        total_oja_loss = state_oja_loss + task_oja_loss

        return state_emb, task_emb, total_oja_loss
    
    def ema(self, other, alpha = 0.999):
        for p, q in zip(self.state_encoder.parameters(), other.state_encoder.parameters()):
            p.data = alpha * p.data + (1 - alpha) * q.data
        for p, q in zip(self.task_encoder.parameters(), other.task_encoder.parameters()):
            p.data = alpha * p.data + (1 - alpha) * q.data

class TDJEncoderPair(torch.nn.Module):
    def __init__(self,
                 in_size,
                 state_dim,
                 task_dim,
                 state_encoder_extra_depth = 3,
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
        if state_encoder_extra_depth is not None:
            layers = []
            for _ in range(state_encoder_extra_depth):
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            extra_depth = torch.nn.Sequential(*layers)
        else:
            extra_depth = torch.nn.Identity()
        state_encoder = torch.nn.Sequential(
            first_stage,
            torch.nn.ReLU(),
            extra_depth,
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
        return state_emb, task_emb, None

    def ema(self, other, alpha = 0.999):
        for p, q in zip(self.state_encoder.parameters(), other.state_encoder.parameters()):
            p.data = alpha * p.data + (1 - alpha) * q.data
        for p, q in zip(self.task_encoder.parameters(), other.task_encoder.parameters()):
            p.data = alpha * p.data + (1 - alpha) * q.data

class TDJPredictor(torch.nn.Module):
    def __init__(self,
                 task_dim,
                 state_dim,
                 action_dim,
                 n_predictors = 2,
                 use_oja = False,
                 out_dim = None,
                 final_activation = torch.nn.Tanh(),
                 hidden_dim = 256):
        super().__init__()
        self.use_oja = use_oja
        out_dim = out_dim if out_dim is not None else hidden_dim
        self.task_embed = MLPEmbed(task_dim + state_dim,
                                   hidden_dim,
                                   hidden_dim)
        self.action_embed = MLPEmbed(action_dim + state_dim,
                                     hidden_dim,
                                     hidden_dim)
        predictors = []
        for _ in range(n_predictors):
            if use_oja:
                predictor = torch.nn.ModuleList([
                    OjaLinear(2 * hidden_dim, hidden_dim,
                            activation = torch.nn.ReLU()),
                    OjaLinear(hidden_dim, out_dim)
                ])
            else:
                predictor = torch.nn.Sequential(
                    torch.nn.Linear(2 * hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, out_dim))
            predictors.append(predictor)
        self.predictors = torch.nn.ModuleList(predictors)
        self.final_activation = final_activation

    def forward(self, context, task, action):
        task_context = torch.cat([task, context], dim = -1)
        task_e = self.task_embed(task_context)
        action_context = torch.cat([action, context], dim = -1)
        action_e = self.action_embed(action_context)
        x = torch.cat([task_e, action_e], dim = -1)
        preds = []
        total_oja_loss = 0.0
        for predictor in self.predictors:
            y = x
            if self.use_oja:
                for layer in predictor:
                    y, l = layer(y, compute_oja_loss = True)
                    total_oja_loss = total_oja_loss + l
                preds.append(y)
            else:
                pred = predictor(y)
                preds.append(pred)
        pred = torch.stack(preds, dim = 0).mean(dim = 0)
        pred = self.final_activation(pred)
        return pred, total_oja_loss
    
    def ema(self, other, alpha = 0.999):
        for p, q in zip(self.task_embed.parameters(), other.task_embed.parameters()):
            p.data = alpha * p.data + (1 - alpha) * q.data
        for p, q in zip(self.action_embed.parameters(), other.action_embed.parameters()):
            p.data = alpha * p.data + (1 - alpha) * q.data
        for p, q in zip(self.predictors.parameters(), other.predictors.parameters()):
            p.data = alpha * p.data + (1 - alpha) * q.data

class TDJActor(torch.nn.Module):

    def __init__(self,
                 task_dim,
                 state_dim,
                 hidden_dim = 256,
                 out_dim = 10,
                 noise_scale = 0.2,):
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
        
    def forward(self, task, state):
        task_state = torch.cat([task, state], dim = -1)
        task_e = self.task_embed(task_state)
        state_e = self.state_embed(state)
        x = torch.cat([task_e, state_e], dim = -1)
        act = self.actor(x)
        if self.noise_scale > 0:
            noise = torch.randn_like(act) * self.noise_scale
            act = act + noise
        return act

class TDJEPA(torch.nn.Module):
    def __init__(self,
                 in_size,
                 exp_conv_out_size = None,
                 use_oja = False,
                 task_dim = 256,
                 state_dim = 256,
                 action_dim = 256,
                 hidden_dim = 256,
                 gamma = 0.99):
        super().__init__()

        self.gamma = gamma
        self.use_oja = use_oja
        if use_oja:
            self.encoders = TDJEncoderPairOja(in_size,
                                              state_dim = state_dim,
                                              task_dim = task_dim,
                                              exp_conv_out_size = exp_conv_out_size,
                                              hidden_dim = hidden_dim)
        else:
            self.encoders = TDJEncoderPair(in_size,
                                        state_dim = state_dim,
                                        task_dim = task_dim,
                                        hidden_dim = hidden_dim,
                                        exp_conv_out_size = exp_conv_out_size,)
        self.predictor = TDJPredictor(task_dim,
                                      state_dim = state_dim,
                                      action_dim = action_dim,
                                      hidden_dim = hidden_dim,
                                      out_dim = task_dim,
                                      use_oja = use_oja)
        self.predictor_task = TDJPredictor(task_dim,
                                           state_dim = task_dim,
                                           action_dim = action_dim,
                                           hidden_dim = hidden_dim,
                                           out_dim = state_dim,
                                           use_oja = use_oja)
        self.actor = TDJActor(task_dim,
                              state_dim,
                              out_dim = action_dim,
                              hidden_dim = hidden_dim)
        self.ema_encoders = deepcopy(self.encoders).requires_grad_(False)
        self.ema_predictor = deepcopy(self.predictor).requires_grad_(False)
        self.ema_predictor_task = deepcopy(self.predictor_task).requires_grad_(False)

        self.optimizer = torch.optim.Adam([{'params': self.predictor.parameters(), 'lr': 1e-3},
                                           {'params': self.predictor_task.parameters(), 'lr': 1e-3},
                                           {'params': self.encoders.parameters()},
                                           {'params': self.actor.parameters(), 'lr': 1e-5}],
                                          lr = 1e-4)

    def forward(self, state, task):
        state_emb, _, _ = self.encoders(state = state)
        action = self.actor(task = task,
                            state = state_emb)
        return action
    
    def get_losses(self, state, task, action, next_state,
                   reg_lambda = 0,
                   reg_oja = 1):
        with torch.no_grad():
            state_emb_target, task_emb_target, _ = self.ema_encoders(state = next_state,)
            alt_action = self.actor(task = task,
                                    state = state_emb_target)
            pred_state_target, _ = self.ema_predictor(context = state_emb_target,
                                                   task = task,
                                                   action = alt_action)
            pred_task_target, _ = self.ema_predictor_task(context = task_emb_target,
                                                       task = task,
                                                       action = alt_action)
        state_emb, task_emb, oja_loss_enc = self.encoders(state = state)
        pred, oja_loss_pred = self.predictor(
            context = state_emb,
            task = task,
            action = action
        )
        pred_task, oja_loss_task = self.predictor_task(
            context = task_emb,
            task = task,
            action = action
        )
        phi_loss = torch.nn.functional.mse_loss(pred,
                                                self.gamma * pred_state_target + task_emb_target)
        psi_loss = torch.nn.functional.mse_loss(pred_task,
                                                self.gamma * pred_task_target + state_emb_target)
        loss = phi_loss + psi_loss
        loss_reg = torch.tensor(0.0, device=state_emb.device)
        if self.use_oja and reg_oja != 0:
            oja_loss = oja_loss_enc + oja_loss_pred + oja_loss_task
            loss_reg += reg_oja * oja_loss

        if reg_lambda != 0:
            loss_reg += reg_lambda * (ortho_loss(state_emb) +
                                    ortho_loss(task_emb))
            
        if loss > 1e3:
            pass
        # actor loss
        action_new = self.actor(
            task = task,
            state = state_emb.detach()
        )
        pred_new_action, _ = self.predictor(
            context = state_emb.detach(),
            task = task,
            action = action_new
        )
        actor_loss = -(pred_new_action * task).sum(dim = -1).mean()
        return loss, loss_reg, actor_loss
    
    def optimizer_steps(self, loss,
                        clip_grad = 1.0):
        self.optimizer.zero_grad()
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(),
                                           clip_grad)
        self.optimizer.step()
        
    def update_ema(self, alpha = 0.999):
        self.ema_encoders.ema(self.encoders, alpha = alpha)
        self.ema_predictor.ema(self.predictor, alpha = alpha)
        self.ema_predictor_task.ema(self.predictor_task, alpha = alpha)


if __name__ == "__main__":
    import os
    import gymnasium as gym
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import numpy as np
    from utils import ReplayBuffer

    torch.manual_seed(333)
    np.random.seed(333)

    env_name = "Ant-v5"
    n_epochs = 100 # 3000 epochs at 8x8 steps/episodes per epoch ~~ 1.5hrs on RTX 3090
    batch_size = 512
    task_dim = 50
    episodes_per_epoch = 10
    buffer_size = int(1e6)
    train_start_steps = int(0.1 * buffer_size)
    steps_per_epoch = train_start_steps // batch_size
    gamma = 0.95
    use_oja = True
    eval_every = 5


    def skill_video(tdjepa,
                    checkpoint_dir = "tmp/",
                    env_name = "Ant-v5",
                    policy_prefix = "",
                    skill_dim = 50,
                    num_skills = 3):
        from gymnasium.wrappers import RecordVideo
        from skill_rl import ContinuousSkillSpace
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        env = gym.make(env_name,
                render_mode="rgb_array",
                max_episode_steps=5000)

        env = RecordVideo(env,
                          video_folder=checkpoint_dir,
                          episode_trigger=lambda x: True,
                          name_prefix=policy_prefix)

        skill_space = ContinuousSkillSpace(skill_dim).to(device)
        skills = skill_space.enumerate().to(device)

        for skill_i in tqdm(range(num_skills)):
            skill = skills[skill_i]
            obs, _ = env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    action = tdjepa(state = state_tensor,
                                    task = skill.unsqueeze(0))
                    # clamp
                    action = torch.clamp(action, -env.action_space.high[0], env.action_space.high[0])
                    action = action.detach().cpu().numpy()[0]
                obs, reward, terminated, truncated, info = env.step(action)
                done = (terminated or truncated)
        env.close()

    env = gym.make(env_name,
                   forward_reward_weight = 0.0,)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tdjepa = TDJEPA(obs_dim,
                    task_dim = task_dim,
                    state_dim = 256,
                    action_dim = action_dim,
                    hidden_dim = 256,
                    use_oja = use_oja,
                    gamma = gamma).to(device)

    buffer = ReplayBuffer(obs_dim, action_dim,
                          special_buffer_dim=task_dim,
                          capacity = buffer_size)

    total_steps = 0
    pbar = tqdm(total = n_epochs)
    losses = []
    total_rewards = []
    running_reward = 0

    for epoch in range(n_epochs):
        counter = 0
        policy = tdjepa.actor
        while (total_steps < train_start_steps) or (counter < episodes_per_epoch):
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
                        action = tdjepa(state = state_tensor,
                                        task = task)
                        # clamp
                        action = torch.clamp(action, -env.action_space.high[0], env.action_space.high[0])
                    next_state, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy()[0])
                    buffer.push(state, action, next_state, done, reward,
                                special = task)
                    total_reward += reward
                    state = next_state

                    counter += 1
                    total_steps += 1
                    step += 1

                    done = (terminated or truncated)
                    pbar.set_description(f"Epoch {epoch} | R {running_reward:.2f} | S{total_steps}")
                total_rewards.append(total_reward)
                running_reward = 0.01 * total_reward + (1 - 0.01) * running_reward


        if total_steps > train_start_steps:
            policy.train()
            for _ in range(steps_per_epoch):
                states, actions, rewards, next_states, dones, task = buffer.sample(batch_size = batch_size,
                                                                                   device = device)
                rand_task = torch.randn(batch_size, task_dim).to(device)
                rand_task = rand_task / (torch.norm(rand_task, p=2, dim=1, keepdim=True) + 1e-8)
                mask = torch.rand(batch_size, 1).to(device) < 0.5
                task = torch.where(mask, task, rand_task)

                loss_rep, loss_reg, actor_loss = tdjepa.get_losses(state = states,
                                        next_state = next_states,
                                        task = task,
                                        action = actions)
                loss = loss_rep + loss_reg + actor_loss
                tdjepa.optimizer_steps(loss)
                tdjepa.update_ema()

                losses.append([loss_rep.item(), actor_loss.item(), loss_reg.item()])
                pbar.set_description(f"Epoch {epoch} |R {running_reward:.2f}| {loss_rep.item():.2f} | {actor_loss.item():.2f} | {loss_reg.item():.2f}")

            if epoch % eval_every == 0:
                with torch.no_grad():
                    emb_states, emb_states_task, _ = tdjepa.encoders(state = states)
                    corr = torch.einsum("bi,ci->bc", emb_states, emb_states)
                    # get mean of off-diagonal
                    off_diag = corr - torch.diag(torch.diag(corr))
                    mean_off_diag = off_diag.mean().item()
                    print(f"Eval Epoch {epoch} | Mean off-diagonal correlation: {mean_off_diag:.4f}")
                skill_video(tdjepa,
                            policy_prefix="tdjepa_",)
        pbar.update(1)
    
    pbar.close()
    losses = np.array(losses)

    fig, ax = plt.subplots(4, 1, figsize = (12, 8))
    smooth_losses = np.apply_along_axis(lambda x: np.convolve(x, np.ones(500) / 500, mode = "valid"),
                                        axis = 0, arr = losses)
    ax[0].plot(smooth_losses[1000:, 0])
    ax[0].set_title("SF Loss")
    ax[1].plot(smooth_losses[1000:, 1])
    ax[1].set_title("Policy Loss")
    ax[2].plot(smooth_losses[1000:, 2])
    ax[2].set_title("Regularization Loss")

    total_rewards = np.array(total_rewards)
    smooth_rewards = np.convolve(total_rewards.squeeze(),
                                 np.ones(100) / 100, mode = "valid")

    ax[3].plot(total_rewards)
    ax[3].set_title("Reward")
    plt.tight_layout()
    fig.savefig("tmp/sac.png")