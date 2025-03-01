import torch
import numpy as np
from torch.nn.functional import mse_loss, huber_loss, relu
from utils import SparseMLP, SymLog, log_barrier_loss, ortho_loss


def volume_loss(x, y, target_volume):
    vol_action = torch.stack([x, y], dim = -1)
    grammian = torch.einsum("bid,bjd->bij",
                            vol_action,
                            vol_action).det()
    vol_loss = log_barrier_loss(grammian.abs().sqrt(), target_volume).mean()
    return vol_loss

class LinearMapLearner(torch.nn.Module):
    """
    Implementing the cognitive map learner from:
    https://www.nature.com/articles/s41467-024-46586-0
    """
    def __init__(self, state_dim, action_dim, hidden_dim = 512, bias = False):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.state_embed = torch.nn.Linear(state_dim, hidden_dim,
                                           bias = bias)
        self.action_embed = torch.nn.Linear(action_dim, hidden_dim,
                                            bias = bias)

    def forward(self, state, action):
        state_embed = self.state_embed(state)
        action_embed = self.action_embed(action)
        return state_embed + action_embed
    
    def manual_update(self, obs, action, next_obs,
                      lr_v = 0.01, lr_q = 0.01):
        assert self.bias == False, "Bias is not supported for manual update"
        with torch.no_grad():
            next_state_hat = self.forward(obs, action)
            next_state = self.state_embed(next_obs)
            pred_error = next_state_hat - next_state

            dV = torch.einsum("bi,bj->bij",
                              -pred_error,
                              action).mean(dim = 0)
            dQ = torch.einsum("bi,bj->bij",
                              pred_error,
                              next_obs).mean(dim = 0)
            self.action_embed.weight += lr_v * dV
            self.state_embed.weight += lr_q * dQ

            # normalize the weights
            self.action_embed.weight /= torch.norm(self.action_embed.weight,
                                                  dim = 0).unsqueeze(0)
        return pred_error
    
    def step_to_goal(self, curr_obs, goal_obs, **kwargs):
        curr_embed = self.state_embed(curr_obs)
        goal_embed = self.state_embed(goal_obs)

        diff = goal_embed - curr_embed
        # this is essential a dual to the action maximizing the state diff
        action = torch.einsum("ij,i->j",
                              self.action_embed.weight,
                              diff)

        return action.clamp(-1, 1)
    
class HierarchicalActionComposer(torch.nn.Module):
    def __init__(self,
                 dim,
                 activation = torch.nn.Tanh(),
                 level_transform = True):
        super().__init__()
        self.dim = dim
        self.activation = activation

        self.pair_weight = torch.nn.Parameter(torch.randn(2 * dim, dim))

        if level_transform:
            self.level_transform = torch.nn.Linear(dim, dim)
        else:
            self.level_transform = torch.nn.Identity()

    def forward(self, actions):
        """
        Takes in an arbitrary number of actions and composes them
        into a single resulting action. Assumes shape is (B, N, D)
        """
        composed_action = actions.clone()
        while composed_action.shape[1] != 1:
            # apply activation beforehand
            composed_action = self.activation(composed_action)
            odd = composed_action.shape[1] % 2
            if odd:
                # randomly extract one action
                idx = torch.randint(0, composed_action.shape[1] - 1, (1,))
                action = composed_action[:, idx]
                action = self.level_transform(action)

                composed_action = torch.cat([composed_action[:, :idx],
                                            composed_action[:, idx + 1:]],
                                            dim = 1)
            composed_action = composed_action.view(-1,
                                                   composed_action.shape[1] // 2,
                                                   2 * self.dim)
            composed_action = torch.einsum("bnd,dm->bnm",
                                           composed_action,
                                           self.pair_weight)
            composed_action = self.level_transform(composed_action)
            if odd:
                # reinsert the action
                idx = 1 + idx // 2
                composed_action = torch.cat([composed_action[:, :idx],
                                            action,
                                            composed_action[:, idx:]],
                                            dim = 1)
        return composed_action.squeeze(1)
    
class DeepMapLearner(torch.nn.Module):
    def __init__(self, state_dim, action_dim,
                 hidden_dim_obs = 512,
                 hidden_dim_act = 512,
                 activation = torch.nn.Tanh(),
                 autoencode = True,
                 lambda_v = 1,
                 lambda_area = 0.5,
                 lambda_t = 1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        if isinstance(hidden_dim_obs, int):
            hidden_dim_obs = [hidden_dim_obs]
        if isinstance(hidden_dim_act, int):
            hidden_dim_act = [hidden_dim_act]
        self.hidden_dim_obs = hidden_dim_obs
        self.hidden_dim_act = hidden_dim_act
        assert hidden_dim_act[-1] == hidden_dim_obs[-1], "Last hidden dim must match"
        self.embed_dim = hidden_dim_obs[-1]
        self.lambda_v = lambda_v
        self.lambda_area = lambda_area
        self.lambda_t = lambda_t
        self.autoencode = autoencode

        self.state_embed = SparseMLP([state_dim] + hidden_dim_obs,
                                     activation=activation)
        self.action_embed = SparseMLP([action_dim] + hidden_dim_act,
                                      activation=activation)
        self.tanh = torch.nn.Identity()
        if self.autoencode:
            hidden_dim_act_inv = hidden_dim_act[::-1]
            self.action_decoder = SparseMLP(hidden_dim_act_inv + [action_dim],
                                            activation=activation)
        
        # for tracking stats
        self.state_vars = []

    def forward(self, state, action):
        state_embed = self.state_embed(state)
        action_embed = self.action_embed(action)
        return state_embed + action_embed
    
    def get_loss(self, obs, action, next_obs):
        state_embed = self.state_embed(obs)
        action_embed = self.action_embed(action)
        with torch.no_grad():
            next_state = self.state_embed(next_obs)

        state_vars = state_embed.var(dim = (0, 1))
        self.state_vars.append(state_vars.mean().item())

        # multistep prediction
        state_embed_start = state_embed[:, 0, :]
        next_state_end = next_state[:, -1, :]
        cumulative_action = action_embed.clone()
        # can't mix up the gradients - treat targets as constant
        loss_state = mse_loss(state_embed_start,
                              (next_state_end - cumulative_action).detach())
        loss_action = mse_loss(cumulative_action,
                               (next_state_end - state_embed_start).detach())

        if self.autoencode:
            action_embed_ = action_embed.clone().detach()
            action_hat = self.action_decoder(action_embed_)
            action_hat = self.tanh(action_hat)
            # weight magnitude and alignment separately
            magnitude_hat, magnitude = action_hat.norm(dim = -1), action_embed_.norm(dim = -1)
            mag_loss = mse_loss(magnitude_hat, magnitude)
            full_magnitude = (magnitude_hat * magnitude)[:, None]
            align_loss = -(action_hat * action).sum(dim = -1) / full_magnitude.detach()
            loss_decode = mag_loss + 2 * align_loss.mean()
        else:
            loss_decode = torch.tensor(0.0,
                                       device = obs.device)
        return loss_state, loss_action, loss_decode
    
    def _gradient_ascent_action(self, curr_obs, diff, desired_loss = 0.9, max_steps = 200):
        # perform gradient descent on the action embedding
        action = torch.randn(self.action_dim,
                             device = curr_obs.device)
        action.requires_grad_(True)
        loss = -1
        step = 0
        while (loss < desired_loss) and (step < max_steps):
            action_embed = self.action_embed(action)
            loss = torch.cosine_similarity(action_embed, diff.detach(),
                                           dim = 0)
            loss.backward()
            with torch.no_grad():
                # maximize the cosine similarity
                action += 0.05 * action.grad
                action.grad.zero_()
                self.action_embed.zero_grad()
            step += 1
            
        action = action.detach().clone()
        return action
    
    def step_to_goal(self, curr_obs, goal_obs, **kwargs):
        with torch.no_grad():
            curr_embed = self.state_embed(curr_obs)
            goal_embed = self.state_embed(goal_obs)

            diff = self.projector(torch.cat([curr_embed, goal_embed], dim = -1))
        if self.autoencode:
            with torch.no_grad():
                action = self.action_decoder(diff)
                action = self.tanh(action)
        else:
            action = self._gradient_ascent_action(curr_obs,
                                                  diff,
                                                  **kwargs)
            action = action.clamp(-1, 1)

        return action
    
class DeepMapLearnerComplex(DeepMapLearner):
    def __init__(self, state_dim, action_dim,
                 hidden_dim_obs = 512,
                 hidden_dim_act = 512,
                 activation = torch.nn.Tanh(),
                 autoencode = True,
                 lambda_v = 1,
                 lambda_area = 0.5,
                 lambda_t = 1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        if isinstance(hidden_dim_obs, int):
            hidden_dim_obs = [hidden_dim_obs]
        if isinstance(hidden_dim_act, int):
            hidden_dim_act = [hidden_dim_act]
        self.hidden_dim_obs = hidden_dim_obs
        self.hidden_dim_act = hidden_dim_act
        assert hidden_dim_act[-1] == hidden_dim_obs[-1], "Last hidden dim must match"
        self.embed_dim = hidden_dim_obs[-1]
        self.lambda_v = lambda_v
        self.lambda_area = lambda_area
        self.lambda_t = lambda_t
        self.autoencode = autoencode

        self.state_embed = SparseMLP([state_dim] + hidden_dim_obs,
                                     activation=activation)
        self.action_embed = SparseMLP([action_dim] + hidden_dim_act,
                                      activation=activation)
        self.projector = SparseMLP([2 * self.embed_dim, 2 * self.embed_dim, self.embed_dim],
                                   activation = activation)
        self.composer = HierarchicalActionComposer(self.embed_dim,
                                                   activation = activation)
        self.tanh = torch.nn.Identity()
        if self.autoencode:
            hidden_dim_act_inv = hidden_dim_act[::-1]
            self.action_decoder = SparseMLP(hidden_dim_act_inv + [action_dim],
                                            activation=activation)
        
        # for tracking stats
        self.state_vars = []
        

    def forward(self, state, action):
        state_embed = self.state_embed(state)
        action_embed = self.action_embed(action)
        return state_embed + action_embed
    
    def get_loss(self, obs, action, next_obs):
        state_embed = self.state_embed(obs)
        action_embed = self.action_embed(action)
        with torch.no_grad():
            next_state = self.state_embed(next_obs)

        state_vars = state_embed.var(dim = (0, 1))
        self.state_vars.append(state_vars.mean().item())

        # multistep prediction
        cumulative_action = self.composer(action_embed)
        state_embed_start = state_embed[:, 0, :]
        next_state_end = next_state[:, -1, :]
        # can't mix up the gradients - treat targets as constant
        loss_state = mse_loss(state_embed_start,
                              (next_state_end - cumulative_action).detach())
        loss_action = mse_loss(cumulative_action,
                               (next_state_end - state_embed_start).detach())
        
        loss_state += self.lambda_area * (1 / (state_vars + 1e-6)).mean()
        self.lambda_area *= 0.999

        # for all steps to the final state, see how the projection gets the first action
        next_state_ends = next_state_end.unsqueeze(1).repeat(1, state_embed.shape[1], 1)
        state_end_pairs = torch.cat([state_embed,
                                     next_state_ends],
                                    dim = -1)
        action_hat = self.projector(state_end_pairs.clone().detach().requires_grad_(True))
        loss_state += mse_loss(action_embed.detach(), action_hat)

        # make sure projector and cumulative action agree
        state_end_pairs = torch.cat([state_embed_start,
                                     state_embed_start + cumulative_action],
                                    dim = -1)
        action_hat = self.projector(state_end_pairs.clone().detach().requires_grad_(True))
        # it should project onto the first step/action
        loss_state += mse_loss(action_hat, action_embed[:, 0, :].detach())

        if self.autoencode:
            action_embed_ = action_embed.clone().detach()
            action_hat = self.action_decoder(action_embed_)
            action_hat = self.tanh(action_hat)
            # weight magnitude and alignment separately
            magnitude_hat, magnitude = action_hat.norm(dim = -1), action_embed_.norm(dim = -1)
            mag_loss = mse_loss(magnitude_hat, magnitude)
            full_magnitude = (magnitude_hat * magnitude)[:, None]
            align_loss = -(action_hat * action).sum(dim = -1) / full_magnitude.detach()
            loss_decode = mag_loss + 2 * align_loss.mean()
        else:
            loss_decode = torch.tensor(0.0,
                                       device = obs.device)
        return loss_state, loss_action, loss_decode
    
    
class IntrinsicMapLearner(torch.nn.Module):
    def __init__(self, state_dim, action_dim,
                 hidden_dim_obs = 512,
                 hidden_dim_act = 512,
                 activation = SymLog(),
                 action_std = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        if isinstance(hidden_dim_obs, int):
            hidden_dim_obs = [hidden_dim_obs]
        if isinstance(hidden_dim_act, int):
            hidden_dim_act = [hidden_dim_act]
        self.hidden_dim_obs = hidden_dim_obs
        self.hidden_dim_act = hidden_dim_act
        assert hidden_dim_act[0] == hidden_dim_obs[-1], "Last hidden dim must match"
        self.embed_dim = hidden_dim_obs[-1]
        self.action_std = action_std

        self.state_embed = SparseMLP([state_dim] + hidden_dim_obs,
                                     activation=activation)

        self.action_decoder = SparseMLP(hidden_dim_act + [action_dim],
                                        activation=activation)
        self.tanh = torch.nn.Tanh()
        self.state_vars = []
        
    def forward(self, action_embed):
        return self.tanh(self.action_decoder(action_embed))
    
    def sample_actions(self, batch_size, device = None):
        actions_embed = torch.randn(batch_size, self.embed_dim,
                                    device = device) * self.action_std
        actions = self.forward(actions_embed)
        return actions, actions_embed
    
    def get_loss(self, obs, action_embed, next_obs):
        state_embed = self.state_embed(obs)
        with torch.no_grad():
            next_state = self.state_embed(next_obs)

        state_vars = state_embed.var(dim = 0)
        self.state_vars.append(state_vars.mean().item())

        loss = mse_loss(next_state - state_embed, action_embed.detach())

        action = self.tanh(self.action_decoder(action_embed))
        # measure of magnitude agreement between state changes and action
        scale = np.sqrt(self.embed_dim)
        action_scale = np.sqrt(self.action_dim)
        with torch.no_grad():
            inner_alignment = (next_state - state_embed).norm(dim = -1) / scale
        loss_a = mse_loss(action.norm(dim = -1) / action_scale,
                          inner_alignment)

        return loss, loss_a
    
    def step_to_goal(self, curr_obs, goal_obs, **kwargs):
        with torch.no_grad():
            curr_embed = self.state_embed(curr_obs)
            goal_embed = self.state_embed(goal_obs)

            diff = goal_embed - curr_embed
            diff = (diff / diff.norm(dim = -1).unsqueeze(-1)) * self.action_std

            action = self.action_decoder(diff)
            action = self.tanh(action)

        return action
            
if __name__ == "__main__":
    from gymnasium import make
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from gymnasium.wrappers import RecordVideo
    from utils import ReplayBuffer
    env_name = "Ant-v5"
    env = make(env_name,
               exclude_current_positions_from_observation=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    target_entropy = -action_dim

    map_learner = LinearMapLearner(obs_dim + 1, action_dim).to(device)
    deep_map_learner = DeepMapLearner(obs_dim + 1, action_dim,
                                     [256, 128, 64, 32], [256, 128, 64, 32],
                                     activation = SymLog()).to(device)
    # deep_map_learner = IntrinsicMapLearner(obs_dim + 1, action_dim,
    #                                        hidden_dim_act = [32, 128],
    #                                        hidden_dim_obs = [256, 128, 64, 32],)
    deep_map_learner = deep_map_learner.to(device)

    n_epochs = 150
    eps_per_epoch = 4
    steps_per_epoch = 4 * eps_per_epoch
    batch_size = 256
    episode_count = 0

    errors = []
    errors_deep = []
    pbar = tqdm(range(n_epochs * eps_per_epoch + n_epochs * steps_per_epoch))
    buffer = ReplayBuffer(obs_dim, action_dim,
                          return_trajectory = True,
                          special_buffer_dim = deep_map_learner.embed_dim,)

    optimizer_state = torch.optim.Adam(deep_map_learner.state_embed.parameters(),
                                 lr = 0.00001)
    if not isinstance(deep_map_learner, IntrinsicMapLearner):
        optimizer_action = torch.optim.Adam(deep_map_learner.action_embed.parameters(),
                                            lr = 0.0001)
    optimizer_decoder = torch.optim.Adam(deep_map_learner.action_decoder.parameters(),
                                         lr = 0.001)

    for epoch in range(n_epochs):
        for _ in range(eps_per_epoch):
            obs, _ = env.reset()
            done = False
            counter = 0
            while not done:
                if isinstance(deep_map_learner, IntrinsicMapLearner):
                    with torch.no_grad():
                        action, a_embeds = deep_map_learner.sample_actions(1,
                                                                           device = device)
                        # conver to numpy
                        action = action[0].cpu().numpy()
                        a_embeds = a_embeds[0]
                else:
                    action = env.action_space.sample()
                    a_embeds = None
                next_obs, reward, terminated, truncated, info = env.step(action)

                done = (terminated or truncated)
                buffer.push(obs, action, next_obs, done, special = a_embeds, episode = episode_count, reward = reward)
                obs = next_obs
                counter += 1

                pbar.set_description(f"Epoch: {epoch}| Counter: {counter}")
            
            episode_count += 1
            pbar.update(1)
        
        for _ in range(steps_per_epoch):
            states, actions, rewards, next_states, dones, a_embeds = buffer.sample(batch_size = batch_size,
                                                                                 device = device)
            traj_length = states.shape[1]
            # concat states and dones
            states = torch.cat([states, dones], dim = -1)
            next_states = torch.cat([next_states, dones], dim = -1)

            #TODO change? ignore trajectories for manual update
            pred_error = map_learner.manual_update(states[:, 0, :],
                                                   actions[:, 0, :],
                                                   next_states[:, 0, :])
            if isinstance(deep_map_learner, IntrinsicMapLearner):
                loss_state, loss_decode = deep_map_learner.get_loss(states[:, 0, :],
                                                                    a_embeds[:, 0, :],
                                                                    next_states[:, 0, :])
            else:
                loss_state, loss_action, loss_decode = deep_map_learner.get_loss(states,
                                                                                 actions,
                                                                                 next_states)

            optimizer_state.zero_grad()
            optimizer_decoder.zero_grad()
            loss_state.backward(retain_graph = True)
            loss_decode.backward()

            if not isinstance(deep_map_learner, IntrinsicMapLearner):
                optimizer_action.zero_grad()
                loss_action.backward()
                optimizer_action.step()
            optimizer_state.step()
            optimizer_decoder.step()

            errors.append(pred_error.norm(dim = -1).mean().item())
            errors_deep.append(loss_state.item() + loss_decode.item())

            pbar.set_description(f"Epoch: {epoch}| Error: {errors[-1]:.2f}| AE: {loss_decode.item():.2f}")
            pbar.update(1)

    pbar.close()

    # get body to x,y position of 10, 10
    goal_xy = [10, 10]
    goal_obs = torch.zeros(obs_dim + 1).to(device)
    goal_obs[0] = goal_xy[0]
    goal_obs[1] = goal_xy[1]
    goal_obs[2] = 0.5
    
    env = make(env_name,
               render_mode="rgb_array",
               exclude_current_positions_from_observation = False,
               max_episode_steps=5000)

    all_xys = []

    for i, map_learner_i in enumerate([map_learner, deep_map_learner]):
        env = RecordVideo(env,
                    video_folder="tmp/",
                    episode_trigger=lambda x: True,
                    name_prefix=f"final_test_{i}")

        obs, info = env.reset()
        done = False
        xys = []
        pbar = tqdm(total = 5000)
        while not done:
            x, y = info["x_position"], info["y_position"]
            # vector pointing from current position to goal
            vec_d = torch.tensor([goal_xy[0] - x, goal_xy[1] - y],
                                dtype = torch.float32,
                                device = device)
            if vec_d.norm() < 0.5:
                break

            obs = torch.tensor(obs, dtype = torch.float32, device = device)
            obs = torch.cat([obs,
                            torch.tensor([done],
                                        dtype = torch.float32,
                                        device = device)])

            action = map_learner_i.step_to_goal(obs, goal_obs)
            next_obs, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
            done = (terminated or truncated)
            obs = next_obs
            pbar.update(1)
            xys.append([info["x_position"], info["y_position"]])

        pbar.close()
        all_xys.append(xys)
    env.close()

    errors = np.array(errors)
    errors_deep = np.array(errors_deep)
    smooth_errors = np.convolve(errors,
                                np.ones(100)/100,
                                mode = "valid")
    smooth_errors_deep = np.convolve(errors_deep,
                                     np.ones(100)/100,
                                     mode = "valid")
    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    ax[0].plot(smooth_errors, label = "Error")
    ax[0].plot(smooth_errors_deep, label = "Deep Error")

    for name, cmap, trajs in zip(["Linear", "Deep"],
                                ["viridis", "plasma"],
                                all_xys):
        trajs = np.array(trajs)
        # make a color array and use LineCollection
        colors = np.linspace(0, 1, len(trajs))
        ax[1].scatter(trajs[:, 0], trajs[:, 1],
                    marker = ",", s = 1,
                    label = name,
                    c = colors, cmap = cmap)
    ax[1].scatter(goal_xy[0], goal_xy[1], label = "Goal")
    ax[0].legend()
    plt.tight_layout()
    fig.savefig("tmp/map_learner.png", dpi = 300)
