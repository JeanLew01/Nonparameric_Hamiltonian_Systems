import numpy as np
import torch
import matplotlib.pyplot as plt

def rollout_deterministic(env, agent, max_steps=None, x0=None):
    if x0 is not None:
        obs, info = env.reset(options={"x0": np.array(x0, dtype=np.float32)})
    else:
        obs, info = env.reset()

    states = []
    actions = []
    rewards = []

    if max_steps is None:
        max_steps = env.horizon_steps

    for k in range(max_steps):
        states.append(obs.copy())

        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            mean_action = agent.actor(obs_tensor)
        action = mean_action.cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action)

        actions.append(action.copy())
        rewards.append(reward)

        if terminated or truncated:
            break

    return np.array(states), np.array(actions), np.array(rewards)