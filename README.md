# Nonparametric Hamiltonian Systems

This package contains simulations for nonparametric control of Hamiltonian
systems. The experiments compare the proposed Chain Policy with Vanilla
Behavior Cloning on spring-mass, single-pendulum, and double-pendulum systems.

## Description

The code structure is given as follows.

```text
baseline/
├─ behavior_cloning.py              # Vanilla behavior cloning baseline

dynamics/
├─ Dynamics_spring_mass.py          # Spring-mass dynamics
├─ Dynamics_single_pendulum.py      # Single-pendulum dynamics
├─ Dynamics_double_pendulum.py      # Double-pendulum dynamics
├─ __init__.py

utils/
├─ control_chain.py                 # Chain policy and control alphabet
├─ behavior_cloning_eval.py         # Vanilla BC evaluation
├─ final_uniform_eval.py            # Final uniform-state experiments
├─ run_bc_hitting_times.py          # BC reach-time experiments
├─ run_paper_experiments.py         # Main experiment runner
├─ run_double_hitting_times_150s.py # Double-pendulum 150s reach-time runner
├─ regenerate_expert_rollouts.py    # Expert rollout generation
├─ verify_final_results.py          # Check saved numerical results
├─ __init__.py

data/                               # Saved expert demonstrations and numerical results
results/                            # Generated visualization figures and copied final results

visualize_results.py                # Visualization script for saved results
requirements.txt
__init__.py
```

## Environment Setup

This code is running in a virtual environment. Python 3.12.3 is recommended.

```bash
python -m venv ./venv
source venv/bin/activate
```

Then install the dependencies as

```bash
pip install -r requirements.txt
```

## Reproduce Experiments

Run the main numerical experiments by

```bash
env MPLCONFIGDIR=/tmp/mpl-paper python \
  -m utils.run_paper_experiments \
  --save-dir data \
  --num-inits 500 \
  --systems spring_mass single_pendulum
```

Check that the saved final numerical results are unchanged by

```bash
env MPLCONFIGDIR=/tmp/mpl-paper python \
  -m utils.verify_final_results \
  --save-dir data
```

## Visualization

Generate figures from the saved numerical results by

```bash
env MPLCONFIGDIR=/tmp/mpl-paper python \
  visualize_results.py \
  --data-dir data \
  --output-dir results
```
