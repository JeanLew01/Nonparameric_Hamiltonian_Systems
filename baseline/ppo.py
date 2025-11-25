import time
import numpy as np

import gymnasium as gym
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

class PPO:
    def __init__(self, policy_class, env,
                 expert_states=None, expert_actions=None,
                 **hyperparameters):
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)

        self._init_hyperparameters(hyperparameters)
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.actor = policy_class(self.obs_dim, self.act_dim)
        self.critic = policy_class(self.obs_dim, 1)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Initialize the covariance matrix used to query the actor for actions
        # Convariance's value, might be critical
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # logger
        self.logger = {
            "delta_t": time.time_ns(),
            "t_so_far": 0,  # timesteps so far
            "i_so_far": 0,  # iterations so far
            "batch_lens": [], # episodic lengths in batch
            "batch_rews": [], # episodic returns in batch
            "actor_losses": [], # losses of actor network in current iteration
        }

        # expert data for BC regularization
        if expert_states is not None and expert_actions is not None:
            self.expert_states = torch.tensor(expert_states, dtype=torch.float32)
            self.expert_actions = torch.tensor(expert_actions, dtype=torch.float32)
        else:
            self.expert_states = None
            self.expert_actions = None

    def learn(self, total_timesteps: int):
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end="")
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0
        i_so_far = 0
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_lens = self.rollout()

            t_so_far += np.sum(batch_lens)
            i_so_far += 1

            self.logger["t_so_far"] = t_so_far
            self.logger["i_so_far"] = i_so_far

            # 2. 计算 V(s) 和 GAE
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k, batch_rtgs = self.compute_gae(self.logger["batch_rews"], V, batch_lens)
            
            # 标准化优势
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # --- 修改点 2: 判断是否处于 Critic Warmup 阶段 ---
            # 如果处于 Warmup，我们不信任 A_k，因此 Actor 只做 BC，不根据 PPO 更新
            is_warmup = i_so_far <= self.critic_warmup_iters
            if is_warmup:
                print(f"[Iter {i_so_far}] Critic Warmup: Actor follows BC only, Critic learns V(s).")

            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # PPO Loss
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                ppo_actor_loss = (-torch.min(surr1, surr2)).mean()
                
                # Critic Loss
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Entropy Bonus
                with torch.no_grad():
                    mean_for_entropy = self.actor(batch_obs)
                dist_for_entropy = MultivariateNormal(mean_for_entropy, self.cov_mat)
                entropy = dist_for_entropy.entropy().mean()

                # BC Regularization Logic
                bc_loss_val = torch.tensor(0.0)
                
                if (self.expert_states is not None) and (self.expert_actions is not None):
                    idx = torch.randint(low=0, high=self.expert_states.shape[0], size=(256,))
                    expert_s = self.expert_states[idx]
                    expert_a = self.expert_actions[idx]
                    pred_a = self.actor(expert_s)
                    bc_loss = nn.MSELoss()(pred_a, expert_a)
                    bc_loss_val = bc_loss # for logging
                else:
                    bc_loss = torch.tensor(0.0)

                if is_warmup:
                    if self.expert_states is not None:
                        actor_loss = 1.0 * bc_loss
                    else:
                        actor_loss = torch.tensor(0.0, requires_grad=True)
                else:
                    if self.expert_states is not None:
                        actor_loss = ppo_actor_loss + self.bc_reg_coef * bc_loss - self.entropy_coef * entropy
                    else:
                        actor_loss = ppo_actor_loss - self.entropy_coef * entropy

                # Backprop Actor
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Backprop Critic
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                self.logger["actor_losses"].append(actor_loss.detach())

            self._log_summary()

            # Decreasing BC's co-efficients
            if not is_warmup:
                self.bc_reg_coef *= self.bc_reg_decay

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []

        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = []
            obs, _ = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                if self.render and (self.logger["i_so_far"] % self.render_every_i == 0) and len(batch_lens) == 0:
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

        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float32)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float32)

        self.logger["batch_rews"] = batch_rews
        self.logger["batch_lens"] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_lens

    def compute_gae(self, batch_rews, values, batch_lens):
        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)
        idx = 0
        for ep_rews, ep_len in zip(batch_rews, batch_lens):
            ep_slice = slice(idx, idx + ep_len)
            v = values[ep_slice]
            gae = 0.0
            for t in reversed(range(ep_len)):
                r_t = ep_rews[t]
                v_t = v[t].item()
                v_next = 0.0 if t == ep_len - 1 else v[t + 1].item()
                delta = r_t + self.gamma * v_next - v_t
                gae = delta + self.gamma * self.gae_lambda * gae
                advantages[idx + t] = gae
                returns[idx + t] = gae + v_t
            idx += ep_len
        return advantages, returns

    def get_action(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
        else:
            obs_tensor = obs
        mean = self.actor(obs_tensor)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze(-1)
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs

    def _init_hyperparameters(self, hyperparameters):
        self.timesteps_per_batch = 4096
        self.max_timesteps_per_episode = 1600
        self.n_updates_per_iteration = 5
        self.lr = 1e-4
        self.gamma = 0.99
        self.clip = 0.2
        self.render = False
        self.render_every_i = 10
        self.save_freq = 10
        self.seed = None
        self.gae_lambda = 0.95
        self.entropy_coef = 0.01
        self.bc_reg_coef = 0.3
        self.bc_reg_decay = 0.995
        
        # --- 新增参数: Critic 预热轮数 ---
        self.critic_warmup_iters = 10

        for param, val in hyperparameters.items():
            setattr(self, param, val)

        if self.seed is not None:
            assert isinstance(self.seed, int)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

    def _log_summary(self):
        delta_t_prev = self.logger["delta_t"]
        self.logger["delta_t"] = time.time_ns()
        delta_t = (self.logger["delta_t"] - delta_t_prev) / 1e9
        
        t_so_far = self.logger["t_so_far"]
        i_so_far = self.logger["i_so_far"]
        
        if len(self.logger["batch_lens"]) > 0:
            avg_ep_lens = np.mean(self.logger["batch_lens"])
        else:
            avg_ep_lens = 0

        if len(self.logger["batch_rews"]) > 0:
            avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger["batch_rews"]])
        else:
            avg_ep_rews = 0

        if len(self.logger["actor_losses"]) > 0:
            avg_actor_loss = np.mean([losses.float().mean().item() for losses in self.logger["actor_losses"]])
        else:
            avg_actor_loss = 0.0

        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------")
        print(f"Average Episodic Length: {avg_ep_lens:.2f}")
        print(f"Average Episodic Return: {avg_ep_rews:.2f}")
        print(f"Average Actor Loss:      {avg_actor_loss:.5f}")
        print(f"BC Reg Coef (current):   {self.bc_reg_coef:.4f}")
        print(f"Timesteps So Far:        {t_so_far}")
        print(f"Iteration took:          {delta_t:.2f} secs")
        print(f"------------------------------------------------------")
        print(flush=True)

        self.logger["batch_lens"] = []
        self.logger["batch_rews"] = []
        self.logger["actor_losses"] = []