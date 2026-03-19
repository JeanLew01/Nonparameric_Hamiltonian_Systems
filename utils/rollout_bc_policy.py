import numpy as np

__all__ = ["rollout_bc_policy"]


def rollout_bc_policy(env, bc_trainer, max_steps=None, x0=None):
    """Roll out a behavior-cloned policy in any Gymnasium-style environment."""
    if x0 is not None:
        obs, _ = env.reset(options={"x0": np.array(x0, dtype=np.float32)})
    else:
        obs, _ = env.reset()

    states = []
    actions = []
    rewards = []

    if max_steps is None:
        max_steps = env.horizon_steps

    for k in range(max_steps):
        states.append(obs.copy())
        action = bc_trainer.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action.copy())
        rewards.append(reward)

        if terminated or truncated:
            break

    return np.array(states), np.array(actions), np.array(rewards)
