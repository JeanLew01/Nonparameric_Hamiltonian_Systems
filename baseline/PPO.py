import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import gymnasium as gym

from collections import namedtuple

from simple_ppo.policy import BasePolicy
from simple_ppo.utils import evaluate


# convert to float tensor
toTensor = lambda x : torch.tensor(x, dtype=torch.float32)


# define buffer data dype
BuffData = namedtuple('BuffData', ['state', 'action', 'logPi', 'Gt', 'At'])

# define replay buffer
class ReplayBuffer(Dataset) :
    def __init__(self) :
        self.buffer = []
    
    def __len__(self) :
        return len(self.buffer)
    
    def __getitem__(self, idx:int) :
        item = self.buffer[idx]
        return item.state, item.action, item.logPi, item.Gt, item.At
    
    def push(self, x:BuffData) :
        self.buffer.append(x)
        return
    
    def clear(self) :
        self.buffer.clear()
        return


# define transition class
class Transition() :
    def __init__(self, obs:torch.Tensor, act_pdf:torch.Tensor, act:torch.Tensor, logPi:torch.Tensor, reward:torch.Tensor, value:torch.Tensor, n_value:torch.Tensor) :
        self.obs = obs
        self.act_pdf = act_pdf
        self.act = act
        self.logPi = logPi
        self.reward = reward
        self.value = value
        self.n_value = n_value
    
    def __call__(self) :
        return self.obs, self.act_pdf, self.act, self.logPi, self.reward, self.value, self.n_value


# define PPO class
class PPO() :

    def __init__(
            self, policy:BasePolicy, optim:optim.Optimizer, env:gym.Env, eval_env:gym.Env,
            gamma:float=0.99, gae_lambda:float=0.95, n_step:int=2048, batch_size:int=64, n_epochs:int=16,
            clip_eps:float=0.2, vf_coef:float=1.0, ent_coef:float=0.001, max_grad_norm:float=1.0,
            eval_num:int=4
            ) :
        
        self.policy = policy
        self.optim = optim
        self.env = env
        self.eval_env = eval_env
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_step = n_step
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.eval_num = eval_num

    
    def _update(self, replay_buffer:ReplayBuffer) :
        
        loader = DataLoader(replay_buffer, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_bonus_sum = 0.0
        cnt = 0
        
        for epoch in range(1, self.n_epochs + 1) :
            
            for state, action, logPi_old, Gt, At in loader :
                
                act_pdf, value = self.policy(state)
                logPi_now = self.policy.log_prob(act_pdf, action)
                
                # policy loss
                ratio = (logPi_now - logPi_old).exp()
                policy_loss = - torch.min(
                    ratio * At,
                    torch.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * At
                ).mean()
                
                # value loss
                value_loss = F.smooth_l1_loss(value, Gt)

                # entropy bonus
                ent_bonus = self.policy.entropy(act_pdf).mean()
                
                # total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * ent_bonus

                # backpropagation
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optim.step()
                
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_bonus_sum += ent_bonus.item()
                cnt += 1
        
        return policy_loss_sum / cnt, value_loss_sum / cnt, entropy_bonus_sum / cnt
    

    def train(self, total_timesteps:int) :
        
        log = []


        current_episode:list[Transition] = []
        episode_list:list[list[Transition]] = []

        
        # define current variables
        obs, info, act_pdf, value, act, logPi = 0, 0, 0, 0, 0, 0
        
        def sample_action(obs:torch.Tensor) :
            with torch.no_grad() :
                act_pdf, value = self.policy(obs)
                act = self.policy.sample(act_pdf)
                logPi = self.policy.log_prob(act_pdf, act)
            
            return act_pdf, value, act, logPi
        
        def reset_episode() :
            nonlocal obs, info, act_pdf, value, act, logPi

            # initial observation
            obs, info = self.env.reset()
            obs = toTensor(obs).unsqueeze(0)
            # initial action
            act_pdf, value, act, logPi = sample_action(obs)


        # init current variables
        reset_episode()

        for timestep in range(1, total_timesteps + 1) :
            
            # transition
            n_obs, reward, termin, trunc, n_info = self.env.step(act.squeeze(0).numpy())
            n_obs = toTensor(n_obs).unsqueeze(0)
            reward = toTensor(reward).unsqueeze(0)

            # sample next action
            n_act_pdf, n_value, n_act, n_logPi = sample_action(n_obs)
            # next value
            n_value = n_value if not termin else toTensor(0.0).unsqueeze(0)

            # collect transition
            transition = Transition(obs, act_pdf, act, logPi, reward, value, n_value)
            current_episode.append(transition)
            
            # set current variables
            if termin or trunc :
                reset_episode()
                
                episode_list.append(current_episode)
                current_episode = []
            else :
                obs, info, act_pdf, value, act, logPi = n_obs, n_info, n_act_pdf, n_value, n_act, n_logPi
            
            
            # update
            if timestep % self.n_step == 0 :
                
                episode_list.append(current_episode)
                current_episode = []
                
                replay_buffer = ReplayBuffer()

                # data preprocess
                for episode in episode_list :
                    
                    At_old = 0.0

                    for k in range(len(episode) - 1, -1, -1) :
                        
                        tran = episode[k]
                        At = (tran.reward + self.gamma * tran.n_value - tran.value) + (self.gamma * self.gae_lambda) * At_old
                        Gt = At + tran.value
                        replay_buffer.push(BuffData(tran.obs, tran.act, tran.logPi, Gt, At))
                        
                        At_old = At
                
                # update
                policy_loss, value_loss, entropy_bonus = self._update(replay_buffer)
                reward_mean, reward_std = evaluate(self.policy, self.eval_env, self.eval_num)
                
                print('| timestep %6d | policy %+8.3f | value %+8.3f | entropy %+8.3f | reward %+7.1f |'%(timestep, policy_loss, value_loss, entropy_bonus, reward_mean))
                log.append({'timestep':timestep, 'policy_loss':policy_loss, 'value_loss':value_loss, 'entropy':entropy_bonus, 'reward':reward_mean})
                
                # clear buffer
                episode_list.clear()
                
                # resample action
                act_pdf, value, act, logPi = sample_action(obs)
        
        return log