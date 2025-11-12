import matplotlib.pyplot as plt

def plot(log:dict, linewidth:int=1, figsize:tuple[int,int]=(8, 6)) :

    extract = lambda name: [l[name] for l in log]
    
    plt.style.use('seaborn-v0_8-darkgrid')

    fig, axs = plt.subplots(2, 2, figsize=figsize)

    x = extract('timestep')

    axs[0,0].plot(x, extract('reward'), color='tab:red', linewidth=linewidth)
    axs[0,0].set_xlabel('timestep')
    axs[0,0].set_ylabel('reward')
    axs[0,0].set_title('Sum of Reward')

    axs[0,1].plot(x, extract('entropy'), color='tab:red', linewidth=linewidth)
    axs[0,1].set_xlabel('timestep')
    axs[0,1].set_ylabel('entropy')
    axs[0,1].set_title('Entropy')

    axs[1,0].plot(x, extract('policy_loss'), color='tab:red', linewidth=linewidth)
    axs[1,0].set_xlabel('timestep')
    axs[1,0].set_ylabel('policy_loss')
    axs[1,0].set_title('Policy loss')

    axs[1,1].plot(x, extract('value_loss'), color='tab:red', linewidth=linewidth)
    axs[1,1].set_xlabel('timestep')
    axs[1,1].set_ylabel('value_loss')
    axs[1,1].set_title('Value loss')

    plt.tight_layout()
    plt.show()

    return