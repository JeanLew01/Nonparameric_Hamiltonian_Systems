import numpy as np
from typing import Callable, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

import gymnasium as gym

def collect_expert_trajectories(
    env: gym.Env,
    expert_policy: Callable[[np.ndarray], np.ndarray],
    num_episodes: int,
    max_steps_per_episode: Optional[int] = None,
    render: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    env : gym.Env: dynamics

    expert_policy : callable, (sth have sealed the do_mpc in this place)

    num_episodes : int, how much episodes do you need
    max_steps_per_episode : int or None: maximum number of interation steps

    Returns
    states : np.ndarray, shape (T, obs_dim)
    actions : np.ndarray, shape (T, act_dim)
    rewards : np.ndarray, shape (T,)
    episode_lengths : np.ndarray, shape (num_episodes,)
    """
    all_states = []
    all_actions = []
    all_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_states = []
        ep_actions = []
        ep_rewards = []

        if max_steps_per_episode is None:
            max_steps = getattr(env, "horizon_steps", 1000)
        else:
            max_steps = max_steps_per_episode

        for t in range(max_steps):
            if render:
                env.render()

            ep_states.append(obs.copy())

            #experts give actions
            action = expert_policy(obs)
            action = np.array(action, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            ep_actions.append(action.copy())
            ep_rewards.append(float(reward))

            done = bool(terminated) or bool(truncated)
            if done:
                break

        episode_lengths.append(len(ep_states))
        all_states.extend(ep_states)
        all_actions.extend(ep_actions)
        all_rewards.extend(ep_rewards)

        print(f"[BC] Collected episode {ep+1}/{num_episodes}, length = {len(ep_states)}")

    states = np.array(all_states, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    rewards = np.array(all_rewards, dtype=np.float32)
    episode_lengths = np.array(episode_lengths, dtype=np.int32)

    return states, actions, rewards, episode_lengths


class BehaviorCloning:
    """
    Behavior Cloning Trainer:
        1. given a policy class
        2. using the expert data to do supervised learning
    """

    def __init__(
        self,
        policy_class: Callable[[int, int], nn.Module],
        obs_dim: int,
        act_dim: int,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        """
        Parameters
        ----------
        policy_class : nn.Module class: class used to construct the Neural Network

        obs_dim : states dimension
        act_dim : action's dimension
        lr : float learning rate
    
        device : torch.device or None
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.policy = policy_class(obs_dim, act_dim).to(self.device)
        self.optimizer = Adam(
            self.policy.parameters(),
            lr=lr,
            weight_decay=float(weight_decay),
        )

        self.loss_fn = nn.MSELoss()

    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        batch_size: int = 256,
        epochs: int = 50,
        shuffle: bool = True,
        verbose: bool = True,
    ):

        #Use the expert data (states, actions) to train the Neural Network
        assert states.shape[0] == actions.shape[0]

        dataset = TensorDataset(
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False
        )

        self.policy.train()
        loss_history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_states, batch_actions in dataloader:
                batch_states = batch_states.to(self.device)
                batch_actions = batch_actions.to(self.device)

                self.optimizer.zero_grad()
                pred_actions = self.policy(batch_states)
                loss = self.loss_fn(pred_actions, batch_actions)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(1, num_batches)
            loss_history.append(float(avg_loss))
            if verbose:
                print(f"[BC] Epoch {epoch+1}/{epochs}, loss = {avg_loss:.6f}")
        self.last_loss_history = loss_history
        return loss_history

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:

        #use the tarined policy
        self.policy.eval()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action = self.policy(obs_tensor)
        return action.cpu().numpy()
