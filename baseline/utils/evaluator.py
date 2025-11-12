import torch

import gymnasium as gym

from ..policy import BasePolicy


toTensor = lambda x: torch.tensor(x, dtype=torch.float32)


def evaluate(policy:BasePolicy, env:gym.Env, eval_num:int=4, deterministic:bool=True) :

    reward_list = []
    
    for _ in range(eval_num) :
        
        # reset environment
        obs, info = env.reset()
        obs = toTensor(obs).unsqueeze(0)
        
        total_reward = 0.0

        for step in range(1000) :
            # sample action
            with torch.no_grad() :
                act_pdf, value = policy(obs)
                act = policy.sample(act_pdf, deterministic)
            
            # transition
            obs, reward, termin, trunc, info = env.step(act.squeeze(0).numpy())
            obs = toTensor(obs).unsqueeze(0)

            total_reward += reward.item()
            
            # episode end
            if termin or trunc :
                break
        
        reward_list.append(total_reward)
    
    # calculate mean and standard deviation of total reward
    vec = toTensor(reward_list)
    mean = vec.mean()
    std = (vec.pow(2).mean() - mean.pow(2)).clamp_min(0.0).sqrt()
    
    return mean.item(), std.item()