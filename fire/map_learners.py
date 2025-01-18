import torch
from torch.nn.functional import mse_loss
from utils import SparseMLP

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
    
    def step_to_goal(self, curr_obs, goal_obs):
        curr_embed = self.state_embed(curr_obs)
        goal_embed = self.state_embed(goal_obs)

        diff = goal_embed - curr_embed
        # this is essential a dual to the action maximizing the state diff
        action = torch.einsum("ij,i->j",
                              self.action_embed.weight,
                              diff)

        return action.clamp(-1, 1)
    
class DeepMapLeaner(torch.nn.Module):
    def __init__(self, state_dim, action_dim,
                 hidden_dim_obs = 512,
                 hidden_dim_act = 512,
                 activation = torch.nn.LeakyReLU(),
                 lambda_v = 0.01,
                 eps = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        if isinstance(hidden_dim_obs, int):
            hidden_dim_obs = [hidden_dim_obs]
        if isinstance(hidden_dim_act, int):
            hidden_dim_act = [hidden_dim_act]
        self.hidden_dim_obs = hidden_dim_obs
        self.hidden_dim_act = hidden_dim_act
        self.lambda_v = lambda_v
        self.eps = eps

        self.state_embed = SparseMLP([state_dim] + hidden_dim_obs,
                                     activation=activation)
        self.action_embed = SparseMLP([action_dim] + hidden_dim_act,
                                      activation=activation)

    def forward(self, state, action):
        state_embed = self.state_embed(state)
        action_embed = self.action_embed(action)
        return state_embed + action_embed
    
    def get_loss(self, obs, action, next_obs):
        state_embed = self.state_embed(obs)
        action_embed = self.action_embed(action)
        with torch.no_grad():
            next_state = self.state_embed(next_obs)
        # can't mix up the gradients - treat targets as constant
        loss_state = mse_loss(state_embed, (next_state - action_embed).detach())
        loss_action = mse_loss(action_embed, (next_state - state_embed).detach())
        # this regularizes the action embedding
        loss_action -= self.lambda_v * torch.clamp(1 - torch.norm(action_embed),
                                                   max = self.eps)
        return loss_state, loss_action
            

if __name__ == "__main__":
    from gymnasium import make
    from tqdm import tqdm
    import numpy as np
    import matplotlib.pyplot as plt
    from utils import ReplayBuffer
    env_name = "Ant-v5"
    env = make(env_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    target_entropy = -action_dim

    map_learner = LinearMapLearner(obs_dim + 1, action_dim).to(device)
    deep_map_learner = DeepMapLeaner(obs_dim + 1, action_dim,
                                     [256], [256],
                                     activation = torch.nn.Identity()).to(device)

    n_epochs = 10
    eps_per_epoch = 30
    steps_per_epoch = 8 * eps_per_epoch
    batch_size = 256
    errors = []
    errors_deep = []
    pbar = tqdm(range(n_epochs * eps_per_epoch + n_epochs * steps_per_epoch))
    buffer = ReplayBuffer(obs_dim, action_dim)

    optimizer_state = torch.optim.Adam(deep_map_learner.state_embed.parameters(),
                                 lr = 0.01)
    optimizer_action = torch.optim.Adam(deep_map_learner.action_embed.parameters(),
                                    lr = 0.001)

    for epoch in range(n_epochs):
        for _ in range(eps_per_epoch):
            obs, _ = env.reset()
            done = False
            counter = 0
            while not done:
                action = env.action_space.sample()
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = (terminated or truncated)

                done = (terminated or truncated)
                buffer.push(obs, action, next_obs, done, reward = reward)
                counter += 1

                pbar.set_description(f"Epoch: {epoch}| Counter: {counter}")

            pbar.update(1)
        
        for _ in range(steps_per_epoch):
            states, actions, rewards, next_states, dones, skills = buffer.sample(batch_size,
                                                                                 device = device)
            # concat states and dones
            states = torch.cat([states, dones], dim = 1)
            next_states = torch.cat([next_states, dones], dim = 1)

            pred_error = map_learner.manual_update(states,
                                                   actions,
                                                   next_states)
            loss_state, loss_action = deep_map_learner.get_loss(states, actions, next_states)

            optimizer_state.zero_grad()
            optimizer_action.zero_grad()
            loss_state.backward(retain_graph = True)
            loss_action.backward()
            optimizer_state.step()
            optimizer_action.step()

            errors.append(pred_error.norm().item())
            errors_deep.append(loss_state.item() + loss_action.item())

            pbar.set_description(f"Epoch: {epoch}| Error: {errors[-1]:.2f}")
            pbar.update(1)

    pbar.close()

    # get body to x,y position of 10, 10
    goal_obs = torch.zeros(obs_dim + 1).to(device)
    goal_obs[0] = 0.5
    goal_xy = [10, 10]

    env = make(env_name,
               max_episode_steps=5000)

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
        goal_angle = torch.atan2(vec_d[1], vec_d[0])
        goal_obs[2] = torch.cos(goal_angle)
        goal_obs[3] = torch.sin(goal_angle)

        obs = torch.tensor(obs, dtype = torch.float32, device = device)
        obs = torch.cat([obs,
                         torch.tensor([done],
                                      dtype = torch.float32,
                                      device = device)])

        action = map_learner.step_to_goal(obs, goal_obs)
        next_obs, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
        done = (terminated or truncated)
        obs = next_obs
        pbar.update(1)
        xys.append([info["x_position"], info["y_position"]])

    pbar.close()

    errors = np.array(errors)
    errors_deep = np.array(errors_deep)
    smooth_errors = np.convolve(errors,
                                np.ones(100)/100,
                                mode = "valid")
    smooth_errors_deep = np.convolve(errors_deep,
                                     np.ones(100)/100,
                                     mode = "valid")
    trajs = np.array(xys)
    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    ax[0].plot(smooth_errors, label = "Error")
    ax[0].plot(smooth_errors_deep, label = "Deep Error")

    # make a color array and use LineCollection
    colors = np.linspace(0, 1, len(trajs))
    ax[1].scatter(trajs[:, 0], trajs[:, 1],
                  marker = ",", s = 1,
                  c = colors, cmap = "viridis")
    ax[1].scatter(goal_xy[0], goal_xy[1], label = "Goal")
