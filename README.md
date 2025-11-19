# Nonparametric Hamiltonian Systems
This package is the simulation for the nonparametric control of the Hamiltonian system.

## Description

The code structure is given as follows.

```bash
baseline/
├─ __pycache__
├─ behavior_cloning.py # module for the behavior cloning
├─ ppo.py # module for the PPO

utils/
├─ __pycache__
├─ __init__.py
├─ mpc.py # module for the NMPC's parameter
├─ mppi.py # module for the MPPI's parameter
├─ rollout_bc_policy.py # behavior policy's rollout
├─ rollout_PPO_policy.py # PPO policy's rollout

data/ # stored expert demonstrations

__init__.py

demo.ipynb # All the results are showed here

Dynamics_cart.py #dynamics for the cartpole

Dynamics_double_pendulum.py #dynamics for the double pendulum

requirements.txt
```

## Enviorment Setup
This code is running in the virtual enviornment. Python 3.12.3 is required.

It is advised to run in a virtual enviornment.

```bash
python -m venv ./venv
source venv/bin/activate
```

Then, install the package as

```bash
pip install -r requirements.txt
```

Experiments can be reproduced using demo.ipynb