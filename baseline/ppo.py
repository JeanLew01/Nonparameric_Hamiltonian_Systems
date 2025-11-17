# utilities/ppo.py

import time
import numpy as np

import gymnasium as gym

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal


class PPO:
    """
    PPO agent for continuous-action Gymnasium environments.

    - 传入 policy_class：一个 PyTorch nn.Module 类，构造方式为 policy_class(obs_dim, act_dim)
      * actor:  policy_class(obs_dim, act_dim)
      * critic: policy_class(obs_dim, 1)
    """

    def __init__(self, policy_class, env, **hyperparameters):
        """
        Parameters
        ----------
        policy_class : nn.Module class
            将被用来实例化 actor / critic 网络，要求构造函数签名为 (obs_dim, out_dim)。
        env : gym.Env
            任意连续动作空间环境，这里你会用 DoublePendulumEnv。
        hyperparameters : dict
            其它 PPO 超参数（可选），如 timesteps_per_batch, max_timesteps_per_episode,
            n_updates_per_iteration, lr, gamma, clip, render, save_freq, seed 等。
        """
        # 确认环境是连续空间
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)

        self._init_hyperparameters(hyperparameters)

        # 环境信息
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # 初始化 actor / critic
        self.actor = policy_class(self.obs_dim, self.act_dim)   # ALG STEP 1
        self.critic = policy_class(self.obs_dim, 1)

        # 优化器
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # 动作分布协方差（固定对角矩阵）
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # 日志
        self.logger = {
            "delta_t": time.time_ns(),
            "t_so_far": 0,
            "i_so_far": 0,
            "batch_lens": [],
            "batch_rews": [],
            "actor_losses": [],
        }

    # ------------------------------------------------------------------
    # 训练主循环
    # ------------------------------------------------------------------
    def learn(self, total_timesteps: int):
        """
        训练 PPO。

        Parameters
        ----------
        total_timesteps : int
            总交互步数上限。
        """
        print(
            f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ",
            end="",
        )
        print(
            f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps"
        )

        t_so_far = 0
        i_so_far = 0

        while t_so_far < total_timesteps:  # ALG STEP 2
            # 采样一批 trajectory
            (
                batch_obs,
                batch_acts,
                batch_log_probs,
                batch_rtgs,
                batch_lens,
            ) = self.rollout()  # ALG STEP 3

            # 当前 batch 的总步数
            t_so_far += np.sum(batch_lens)
            i_so_far += 1

            self.logger["t_so_far"] = t_so_far
            self.logger["i_so_far"] = i_so_far

            # 计算优势：A_k = G_t - V(s_t)
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()  # ALG STEP 5

            # 归一化优势
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # 多个 epoch 的 PPO 更新
            for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # ratio = pi_theta(a|s) / pi_theta_old(a|s)
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # clipped surrogate objective
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # 更新 actor
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # 更新 critic
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                self.logger["actor_losses"].append(actor_loss.detach())

            # 打印日志
            self._log_summary()

            # 定期保存模型
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), "./ppo_actor.pth")
                torch.save(self.critic.state_dict(), "./ppo_critic.pth")

    # ------------------------------------------------------------------
    # 采样一批数据
    # ------------------------------------------------------------------
    def rollout(self):
        """
        与环境交互，采样一个 batch。

        Returns
        -------
        batch_obs : torch.Tensor, shape (T, obs_dim)
        batch_acts : torch.Tensor, shape (T, act_dim)
        batch_log_probs : torch.Tensor, shape (T,)
        batch_rtgs : torch.Tensor, shape (T,)
        batch_lens : list[int]
        """
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []

        t = 0

        # 一直采样直到步数达到 timesteps_per_batch
        while t < self.timesteps_per_batch:
            ep_rews = []

            obs, _ = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                # 如果开启渲染，这里会调 env.render()
                if (
                    self.render
                    and (self.logger["i_so_far"] % self.render_every_i == 0)
                    and len(batch_lens) == 0
                ):
                    # 注意：DoublePendulumEnv 目前没有实现 render，可以在使用时把 render 设为 False
                    self.env.render()

                t += 1
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, terminated, truncated, _ = self.env.step(action)

                done = bool(terminated) or bool(truncated)

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        batch_obs = torch.tensor(batch_obs, dtype=torch.float32)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float32)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32)
        batch_rtgs = self.compute_rtgs(batch_rews)  # ALG STEP 4

        self.logger["batch_rews"] = batch_rews
        self.logger["batch_lens"] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    # ------------------------------------------------------------------
    # 计算 Reward-To-Go
    # ------------------------------------------------------------------
    def compute_rtgs(self, batch_rews):
        """
        计算每个时间步的 Reward-To-Go。

        Parameters
        ----------
        batch_rews : list[list[float]]
            每个 episode 的 reward 序列。

        Returns
        -------
        batch_rtgs : torch.Tensor, shape (T,)
        """
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0.0
            for rew in reversed(ep_rews):
                discounted_reward = rew + self.gamma * discounted_reward
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float32)
        return batch_rtgs

    # ------------------------------------------------------------------
    # 从 actor 取一个动作
    # ------------------------------------------------------------------
    def get_action(self, obs):
        """
        从当前策略中采样一个动作（用于 rollout）。

        Parameters
        ----------
        obs : np.ndarray 或 list
            当前观测。

        Returns
        -------
        action : np.ndarray, shape (act_dim,)
        log_prob : torch.Tensor, scalar
        """
        if not isinstance(obs, torch.Tensor):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
        else:
            obs_tensor = obs

        mean = self.actor(obs_tensor)

        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    # ------------------------------------------------------------------
    # 在 batch 上评估 V(s) 和 log pi(a|s)
    # ------------------------------------------------------------------
    def evaluate(self, batch_obs, batch_acts):
        """
        在一个 batch 上计算：

        - V(s) = critic(s)
        - log_probs = log pi(a|s)

        Parameters
        ----------
        batch_obs : torch.Tensor, shape (T, obs_dim)
        batch_acts : torch.Tensor, shape (T, act_dim)

        Returns
        -------
        V : torch.Tensor, shape (T,)
        log_probs : torch.Tensor, shape (T,)
        """
        V = self.critic(batch_obs).squeeze(-1)

        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs

    # ------------------------------------------------------------------
    # 超参数初始化
    # ------------------------------------------------------------------
    def _init_hyperparameters(self, hyperparameters):
        """
        初始化 PPO 超参数，传入 hyperparameters 可以覆盖默认值。
        """
        # 算法超参数（默认值可以按需改）
        self.timesteps_per_batch = 4096    # 每个 batch 的步数
        self.max_timesteps_per_episode = 800
        self.n_updates_per_iteration = 5
        self.lr = 3e-4
        self.gamma = 0.99
        self.clip = 0.2

        # 其它参数
        self.render = False                # DoublePendulumEnv 没有 render，默认关掉
        self.render_every_i = 10
        self.save_freq = 10
        self.seed = None

        # 覆盖默认值
        for param, val in hyperparameters.items():
            setattr(self, param, val)

        # 设置随机种子
        if self.seed is not None:
            assert isinstance(self.seed, int)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

    # ------------------------------------------------------------------
    # 打印日志
    # ------------------------------------------------------------------
    def _log_summary(self):
        delta_t_prev = self.logger["delta_t"]
        self.logger["delta_t"] = time.time_ns()
        delta_t = (self.logger["delta_t"] - delta_t_prev) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger["t_so_far"]
        i_so_far = self.logger["i_so_far"]

        avg_ep_lens = np.mean(self.logger["batch_lens"])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger["batch_rews"]])
        avg_actor_loss = np.mean(
            [losses.float().mean().item() for losses in self.logger["actor_losses"]]
        )

        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Actor Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # 清空 batch 级日志
        self.logger["batch_lens"] = []
        self.logger["batch_rews"] = []
        self.logger["actor_losses"] = []