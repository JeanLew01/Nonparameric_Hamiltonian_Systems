import numpy as np
import matplotlib.pyplot as plt
from casadi import *
import do_mpc
import matplotlib.animation as animation
import json, os

# Decide whether to compute tuples or load policy
# --- Configuration flags ---
LOAD_TRAJECTORIES = True  # New flag: when True we compute trajectories; existing code used "if PRODUCE_PREDICTION".
                                # We use "not LOAD_TRAJECTORIES" where the old PRODUCE_PREDICTION was used.
LOAD_POLICY = True  # Whether to load precomputed controller tuples (True) or recompute them (False)
PRODUCE_FIGURES = True # Whether to produce figures
PRODUCE_FIGURES_CLOSE_LOOP = True  # Whether to produce closed-loop figures

RADIUS_AMPLIFY = 3  # amplify the computed r_i by this factor for more robustness

DIST_THRESHOLD_LQR = .25  # Define a distance threshold for switching to LQR

FINAL_THETA = np.pi  # target upright angle


RANDOM_CLOSED_LOOP_SEED = 12345
rng_closed_loop = np.random.default_rng(RANDOM_CLOSED_LOOP_SEED)
N_CLOSED_LOOP_RANDOM = 1 # number of random closed-loop simulations

# delta = 1/10  # small offset to avoid numerical issues
# Allow computing predicted trajectories for a list of initial conditions
initial_conditions = [
    [0.0, 0.0],
    [-5 , 5],
    [5+np.pi, -5],
]
# Add 50 equidistant initial conditions on a circle of radius `delta` centered at (pi, 0)
# n_extra = 50
# angles = np.linspace(0, 2 * np.pi, n_extra, endpoint=False)
# extra_ics = [[float(np.pi + delta * np.cos(a)), float(delta * np.sin(a))] for a in angles]
# initial_conditions.extend(extra_ics)
# Define dt once and reuse in later analyses
# n_extra = 7 
# delta = 1/20  # small offset to avoid numerical issues
# angles = np.linspace(0, 2 * np.pi, n_extra, endpoint=False)
# extra_ics = [[float(np.pi + delta * np.cos(a)), float(delta * np.sin(a))] for a in angles]
# initial_conditions.extend(extra_ics)


T = 6.0  # total time horizon (seconds)
dt = 0.01
N = int(T // dt)
N_mpc = int(1 * N)  # prediction horizon

# Closed-loop simulation time (seconds)
T_closed_loop = 20  # <-- change this value as desired
N_closed_loop = int(T_closed_loop // dt)

# Define the model
def create_model():
    model_type = 'continuous'
    model = do_mpc.model.Model(model_type)

    # States (single pendulum)
    theta = model.set_variable(var_type='_x', var_name='theta')
    theta_dot = model.set_variable(var_type='_x', var_name='theta_dot')

    # Control input (torque at base)
    u = model.set_variable(var_type='_u', var_name='u')
    # Time-scaling variable (for minimum-time)
    s = model.set_variable(var_type='_u', var_name='s')

    # Physical parameters
    m = 1.0   # Pendulum mass
    l = 2.0   # Pendulum length
    g = 9.81  # Gravity
    d = 0.00   # Damping coefficient

    # Minimum-time dynamics: multiply by s
    I = m * l**2
    theta_ddot = (u - m * g * l * sin(theta) - d * theta_dot) / I
    model.set_rhs('theta', s * theta_dot)
    model.set_rhs('theta_dot', s * theta_ddot)

    model.setup()
    return model

# Compute Hamiltonian energy for single pendulum
def compute_hamiltonian(theta, theta_dot, m=1.0, l=2.0, g=9.81):
    # KE = 0.5 * I * theta_dot^2, I = m*l^2
    KE = 0.5 * m * l**2 * theta_dot**2
    # PE = m*g*l*(1 - cos(theta)) (zero at downward position)
    PE = m * g * l * (1 - np.cos(theta))
    return KE + PE

# Symbolic Hamiltonian for MPC objective (CasADi expressions)
def symbolic_hamiltonian(theta_sym, theta_dot_sym, m=1.0, l=2.0, g=9.81):
    KE = 0.5 * m * l**2 * theta_dot_sym**2
    PE = m * g * l * (1 - cos(theta_sym))
    return KE + PE

# Setup model and MPC
model = create_model()
mpc = do_mpc.controller.MPC(model)
t_step = dt
mpc.set_param(
    n_horizon=N_mpc,
    t_step=t_step,
    state_discretization='collocation',
    collocation_type='radau',
    collocation_deg=3,
    collocation_ni=2,
    store_full_solution=True
)

# Minimum-time objective
s = model.u['s']
lterm = s
mterm = 0 * model.x['theta']
mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(u=1e-1, s=1e-1)
mpc.bounds['lower', '_u', 'u'] = -10
mpc.bounds['upper', '_u', 'u'] = 10 
mpc.bounds['lower', '_u', 's'] = 0.02
mpc.bounds['upper', '_u', 's'] = 10.0
# Terminal constraint: upright position and zero velocity
mpc.terminal_bounds['lower', 'theta'] = FINAL_THETA
mpc.terminal_bounds['upper', 'theta'] = FINAL_THETA
mpc.terminal_bounds['lower', 'theta_dot'] = 0.0
mpc.terminal_bounds['upper', 'theta_dot'] = 0.0
mpc.setup()

# Numeric stage cost (used for plotting/analysis)
def stage_cost_numeric(theta, theta_dot, u, s):
    """
    Numeric version of the minimum-time MPC stage cost for plotting/analysis.
    Matches the symbolic stage cost: lterm = s
    """
    s_val = float(np.asarray(s).squeeze())
    return s_val



# Cache settings for MPC prediction
PREDICTED_JSON_PATH = 'Hamiltonian/Inverted Pendulum/figures/predicted_trajectory.json'


predicted_trajectories = []

# Prepare path for predicted JSON (directory needed later)
pred_dir = os.path.dirname(PREDICTED_JSON_PATH)

# Only compute and store predicted trajectories when requested
if not LOAD_TRAJECTORIES:
    predicted_trajectories = []

    # Compute and store predicted trajectory for each initial condition
    for x0_vals in initial_conditions:
        x0 = np.array(x0_vals, dtype=float)
        
        # ensure the controller uses the fresh initial guess and forgets previous predictions
        mpc.x0 = x0

        # try clearing stored prediction/history if available in your do-mpc version
        try:
            if hasattr(mpc, 'reset_history'):
                mpc.reset_history()     # newer do-mpc may have this
            elif hasattr(mpc, 'data') and hasattr(mpc.data, 'clear'):
                mpc.data.clear()        # type: ignore # clear logged/prediction data
        except Exception:
            pass

        # reinitialize solver initial guess (collocation/warm-start)
        try:
            mpc.set_initial_guess()
        except Exception:
            # some do-mpc versions may not need / accept this call here
            pass

        # If you still observe warm-starting, the foolproof option is to
        # recreate the MPC object (rebuild params/objective/bounds) so no
        # previous solver state remains.

        # trigger one MPC step to populate predictions (don't need returned u)
        try:
            _ = mpc.make_step(x0)
        except Exception:
            # some do_mpc versions accept the internal state, ignore failures
            pass

        # Extract predicted states and controls using do_mpc Data.prediction
        x_pred_dm = mpc.data.prediction(('_x',))
        u_pred_dm = mpc.data.prediction(('_u',))

        # Convert to numpy and make time-major arrays
        X_pred = np.squeeze(np.array(x_pred_dm, dtype=float), axis=-1).T
        U_pred = np.squeeze(np.array(u_pred_dm, dtype=float), axis=-1).T

        # Compute Hamiltonian for predicted trajectory
        H_pred = np.array([compute_hamiltonian(x[0], x[1]) for x in X_pred], dtype=float)

        # Symmetric trajectory
        X_pred_sym = X_pred.copy()
        X_pred_sym[:, 0] = -X_pred[:, 0]
        X_pred_sym[:, 1] = -X_pred[:, 1]
        U_pred_sym = U_pred.copy()
        U_pred_sym[:, 0] = -U_pred[:, 0]
        U_pred_sym[:, 1] = U_pred[:, 1]
        H_pred_sym = np.array([compute_hamiltonian(x[0], x[1]) for x in X_pred_sym], dtype=float)

        # Store both u and s in predicted trajectories
        # build symmetric control: negate only the torque (u) column, keep time-scaling s unchanged
        # # U_pred_sym = U_pred.copy()
        # if U_pred_sym.ndim == 1:
        #     U_pred_sym = -U_predSym
        # else:
        #     U_pred_sym[:, 0] = -U_predSym[:, 0]

        predicted_trajectories.append({
            'x0': x0_vals,
            'X_pred': X_pred.tolist(),
            'U_pred': U_pred.tolist(),  # U_pred[:,0]=u, U_pred[:,1]=s
            'H_pred': H_pred.tolist(),
            'X_pred_sym': X_pred_sym.tolist(),
            'U_pred_sym': U_pred_sym.tolist(),
            'H_pred_sym': H_pred_sym.tolist()
        })
else:
    # do not compute predictions; keep list empty (will attempt to load from JSON later)
    predicted_trajectories = []

if not LOAD_TRAJECTORIES:
    # Ensure directory exists and save aggregated predictions.
    os.makedirs(pred_dir, exist_ok=True)
    dump_obj = {
        'predicted_trajectories': predicted_trajectories,
        'n_trajectories': len(predicted_trajectories)
    }
    with open(PREDICTED_JSON_PATH, 'w') as f:
        json.dump(dump_obj, f, indent=2)
    print(f"Saved {len(predicted_trajectories)} predicted trajectories to '{PREDICTED_JSON_PATH}'.")
else:
    # Load previously saved predictions (if available)
    if os.path.exists(PREDICTED_JSON_PATH):
        with open(PREDICTED_JSON_PATH, 'r') as f:
            data = json.load(f)
        # Support both dict with key or a raw list
        if isinstance(data, dict) and 'predicted_trajectories' in data:
            predicted_trajectories = data['predicted_trajectories']
        elif isinstance(data, list):
            predicted_trajectories = data
        else:
            predicted_trajectories = []
        print(f"Loaded {len(predicted_trajectories)} predicted trajectories from '{PREDICTED_JSON_PATH}'.")
    else:
        print(f"Predicted JSON not found at '{PREDICTED_JSON_PATH}'. Continuing with empty predicted_trajectories.")
        predicted_trajectories = []

# Expose variables for compatibility with the rest of the script,
# but also collect all predicted trajectories for multi-trajectory plotting.
if len(predicted_trajectories) > 0:
    X_preds = [np.array(p['X_pred'], dtype=float) for p in predicted_trajectories]
    # Normalize U_pred entries to shape (n_steps, 2) -> [u, s]
    def _normalize_U_from_entry(p):
        U = np.array(p.get('U_pred', []), dtype=float)
        if U.size == 0:
            return np.zeros((N_mpc, 2), dtype=float)
        if U.ndim == 1:
            U = U.reshape(-1, 1)
        if U.shape[1] == 1:
            s_col = np.ones((U.shape[0], 1), dtype=float)
            U = np.hstack([U, s_col])
        return U

    U_preds = [_normalize_U_from_entry(p) for p in predicted_trajectories]
    H_preds = [np.array(p['H_pred'], dtype=float) for p in predicted_trajectories]
    X_preds_sym = [np.array(p['X_pred_sym'], dtype=float) for p in predicted_trajectories]
    U_preds_sym = [np.array(p['U_pred_sym'], dtype=float) for p in predicted_trajectories]
    H_preds_sym = [np.array(p['H_pred_sym'], dtype=float) for p in predicted_trajectories]

    # Keep the first trajectory accessible for backward compatibility
    # X_pred = X_preds[0]
    # U_pred = U_preds[0]
    # H_pred = H_preds[0]
    # X_pred_sym = X_preds_sym[0]
    # U_pred_sym = U_preds_sym[0]
    # H_pred_sym = H_preds_sym[0]
else: 
    X_preds = [np.zeros((N_mpc + 1, 2), dtype=float)]
    # fallback U must have two columns (u,s)
    U_preds = [np.zeros((N_mpc, 2), dtype=float)]
    H_preds = [np.zeros((N_mpc + 1,), dtype=float)]
    X_preds_sym = [X_preds[0].copy()]
    U_preds_sym = [-U_preds[0]]
    H_preds_sym = [H_preds[0].copy()]

    # X_pred = X_preds[0]
    # U_pred = U_preds[0]
    # H_pred = H_preds[0]
    # X_pred_sym = X_preds_sym[0]
    # U_pred_sym = U_preds_sym[0]
    # H_pred_sym = H_preds_sym[0]



# Compute Hamiltonian at the terminal (upright, zero velocity) state
H_star = compute_hamiltonian(FINAL_THETA, 0.0)
print(f"Steady-state Hamiltonian (H*): {H_star}")

if PRODUCE_FIGURES:
    cmap = plt.get_cmap('tab10')
    n_traj = len(X_preds)

    # Plot Hamiltonian energy for all predicted trajectories
    plt.figure()
    for i, (H_p, U_p) in enumerate(zip(H_preds, U_preds)):
        s_vals = np.squeeze(U_p[:, 1])
        t_pred = np.cumsum(s_vals * dt)
        plt.plot(np.concatenate([[0], t_pred]), H_p, color=cmap(i % 10), alpha=0.9, label=f"x0={predicted_trajectories[i]['x0']}" if i < len(predicted_trajectories) else f"traj_{i}")
    plt.title('Predicted Hamiltonian Energy Over Time (all trajectories)')
    plt.xlabel('Time (s)')
    plt.ylabel('Hamiltonian Energy')
    plt.grid(True)
    plt.legend(loc='best', fontsize='small')
    plt.savefig('Hamiltonian/Inverted Pendulum/figures/predicted_hamiltonian_energy_all.png')

    # Plot Hamiltonian deviation for all predicted trajectories
    plt.figure()
    for i, (H_p, U_p) in enumerate(zip(H_preds, U_preds)):
        s_vals = np.squeeze(U_p[:, 1])
        t_pred = np.cumsum(s_vals * dt)
        plt.plot(np.concatenate([[0], t_pred]), np.abs(H_p - H_star), color=cmap(i % 10), alpha=0.9, label=f"x0={predicted_trajectories[i]['x0']}" if i < len(predicted_trajectories) else f"traj_{i}")
    plt.title('Predicted Hamiltonian Energy Deviation Over Time (all trajectories)')
    plt.xlabel('Time (s)')
    plt.ylabel('|H(t) - H*|')
    plt.grid(True)
    plt.legend(loc='best', fontsize='small')
    plt.savefig('Hamiltonian/Inverted Pendulum/figures/predicted_hamiltonian_deviation_all.png')

    # Plot predicted states: theta and theta_dot for all trajectories
    plt.figure()
    for i, (X_p, U_p) in enumerate(zip(X_preds, U_preds)):
        s_vals = np.squeeze(U_p[:, 1])
        t_pred = np.cumsum(s_vals * dt)
        plt.plot(np.concatenate([[0], t_pred]), X_p[:, 0], color=cmap((2*i) % 10), alpha=0.8, linestyle='-', label=f"theta, x0={predicted_trajectories[i]['x0']}" if i < len(predicted_trajectories) else f"theta_traj_{i}")
        plt.plot(np.concatenate([[0], t_pred]), X_p[:, 1], color=cmap((2*i+1) % 10), alpha=0.6, linestyle='--', label=f"omega, x0={predicted_trajectories[i]['x0']}" if i < len(predicted_trajectories) else f"omega_traj_{i}")
    plt.title('Predicted State Trajectories (all trajectories)')
    plt.xlabel('Time (s)')
    plt.ylabel('State Value')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.savefig('Hamiltonian/Inverted Pendulum/figures/predicted_state_trajectories_all.png')

    # Plot predicted control actions for all trajectories
    plt.figure()
    for i, U_p in enumerate(U_preds):
        if U_p.size == 0:
            continue
        u_vals = np.squeeze(U_p[:, 0])  # u
        s_vals = np.squeeze(U_p[:, 1])  # s
        t_pred = np.cumsum(s_vals * dt)
        plt.step(t_pred, u_vals, where='post', color=cmap(i % 10), alpha=0.9, label=f"u, x0={predicted_trajectories[i]['x0']}" if i < len(predicted_trajectories) else f"u_traj_{i}")
        plt.step(t_pred, s_vals, where='post', color=cmap((i+1) % 10), alpha=0.5, linestyle='--', label=f"s, x0={predicted_trajectories[i]['x0']}" if i < len(predicted_trajectories) else f"s_traj_{i}")
    plt.title('Predicted Control Actions Over Time (all trajectories)')
    plt.xlabel('Time (s)')
    plt.ylabel('u (solid), s (dashed)')
    plt.grid(True)
    plt.legend(loc='best', fontsize='small')
    plt.savefig('Hamiltonian/Inverted Pendulum/figures/predicted_control_actions_all.png')

    # Optional: also plot symmetric trajectories on separate figures (if available)
    plt.figure()
    for i, (H_p, U_p) in enumerate(zip(H_preds_sym, U_preds_sym)):
        s_vals = np.squeeze(U_p[:, 1])
        t_pred = np.cumsum(s_vals * dt)
        plt.plot(np.concatenate([[0], t_pred]), H_p, color=cmap(i % 10), alpha=0.9, linestyle=':', label=f"sym x0={predicted_trajectories[i]['x0']}" if i < len(predicted_trajectories) else f"sym_{i}")
    plt.title('Predicted Hamiltonian Energy Over Time (symmetric, all trajectories)')
    plt.xlabel('Time (s)')
    plt.ylabel('Hamiltonian Energy')
    plt.grid(True)
    plt.legend(loc='best', fontsize='small')
    plt.savefig('Hamiltonian/Inverted Pendulum/figures/predicted_hamiltonian_energy_all_sym.png')



# Ensure X is an array for later plots that use X[:, ...]
# X = np.array(X_pred, dtype=float)
# U = np.array(U_pred, dtype=float)
# H = np.array(H_pred, dtype=float)


if PRODUCE_FIGURES:
    # ------------------------------------------------------------------
    # Multi-trajectory plots (use X_preds, U_preds, H_preds produced earlier)
    cmap = plt.get_cmap('tab10')
    # Instantaneous objective for all predicted trajectories
    plt.figure()
    any_plotted = False
    for i, (Xp, Up) in enumerate(zip(X_preds, U_preds)):
        if Xp is None or Up is None:
            continue
        n_steps_i = min(Xp.shape[0] - 1, Up.shape[0]) if Up.size > 0 else 0
        inst_cost_i = []
        s_vals = np.squeeze(Up[:, 1])
        t_pred = np.cumsum(s_vals * dt)
        t_plot = np.concatenate([[0], t_pred])[:n_steps_i+1]
        for k in range(n_steps_i):
            th, thdot = float(Xp[k, 0]), float(Xp[k, 1])
            uval = float(np.squeeze(Up[k, 0]))
            sval = float(np.squeeze(Up[k, 1]))
            inst_cost_i.append(stage_cost_numeric(th, thdot, uval, sval))
        if len(inst_cost_i) == 0:
            continue
        plt.plot(t_plot[:-1], inst_cost_i, color=cmap(i % 10), alpha=0.9, label=f"x0={predicted_trajectories[i]['x0']}")
        any_plotted = True
    if any_plotted:
        plt.title('Instantaneous MPC Objective Over Time (all predicted trajectories)')
        plt.xlabel('Time (s)')
        plt.ylabel('Instantaneous Objective')
        plt.grid(True)
        plt.legend(loc='best', fontsize='small')
        plt.savefig('Hamiltonian/Inverted Pendulum/figures/instantaneous_objective_all.png')
    plt.close()
    
    plt.figure()
    any_plotted = False
    for i, (Xp, Up) in enumerate(zip(X_preds, U_preds)):
        if Xp is None or Up is None:
            continue
        n_steps_i = min(Xp.shape[0] - 1, Up.shape[0]) if Up.size > 0 else 0
        inst_cost_i = []
        s_vals = np.squeeze(Up[:, 1])
        t_pred = np.cumsum(s_vals * dt)
        t_plot = np.concatenate([[0], t_pred])[:n_steps_i+1]
        for k in range(n_steps_i):
            th, thdot = float(Xp[k, 0]), float(Xp[k, 1])
            uval = float(np.squeeze(Up[k, 0]))
            sval = float(np.squeeze(Up[k, 1]))
            inst_cost_i.append(stage_cost_numeric(th, thdot, uval, sval))
        if len(inst_cost_i) == 0:
            continue
        c2g = np.array([np.sum(inst_cost_i[j:]) for j in range(len(inst_cost_i))])
        c2g[c2g <= 0] = 1e-12
        plt.plot(t_plot[:-1], c2g, color=cmap(i % 10), alpha=0.9, label=f"x0={predicted_trajectories[i]['x0']}")
        any_plotted = True
    if any_plotted:
        plt.yscale('log')
        plt.title('Cost-to-Go Over Time (log scale) - all predicted trajectories')
        plt.xlabel('Time (s)')
        plt.ylabel('Cost-to-Go (log scale)')
        plt.grid(True, which='both')
        plt.legend(loc='best', fontsize='small')
        plt.savefig('Hamiltonian/Inverted Pendulum/figures/cost_to_go_all.png')
    plt.close()

    plt.figure()
    any_plotted = False
    for i, Xp in enumerate(X_preds):
        if Xp is None:
            continue
        plt.plot(Xp[:, 0], Xp[:, 1], '-', color=cmap(i % 10), alpha=0.9, label=f"x0={predicted_trajectories[i]['x0']}")
        any_plotted = True
    if any_plotted:
        plt.title('Phase Plot: theta vs theta_dot (all predicted trajectories)')
        plt.xlabel('theta (rad)')
        plt.ylabel('theta_dot (rad/s)')
        plt.grid(True)
        plt.legend(loc='best', fontsize='small')
        plt.savefig('Hamiltonian/Inverted Pendulum/figures/phase_plot_all.png')
    plt.close()

L = 4.905
tau = 2.0
tau_steps = int(np.ceil(tau / dt))
alpha = 1e-5

out_dir = 'Hamiltonian/Inverted Pendulum/figures'
os.makedirs(out_dir, exist_ok=True)

def state_distance(x, x_ref):
    theta_err = np.arctan2(np.sin(x[0] - x_ref[0]), np.cos(x[0] - x_ref[0]))
    theta_dot_err = x[1] - x_ref[1]
    return np.sqrt(theta_err**2 + theta_dot_err**2)

def get_state_array(x_struct):
    try:
        return np.array([float(x_struct['theta']), float(x_struct['theta_dot'])], dtype=float)
    except Exception:
        return np.array(x_struct).astype(float).flatten()

# target state
x_star = np.array([FINAL_THETA, 0.0])


# If True, attempt to recover results_alpha from JSON instead of using in-memory results
# RECOVER_RESULTS_ALPHA = False

json_path = os.path.join(out_dir, 'controller_robust_tuples.json')

if not LOAD_POLICY:
    # Aggregate results for all predicted trajectories
    all_results_alpha = []   # list of tuples (x, r, t, u_seq, source_idx, x0)
    robust_tuples = []       # list of dicts to save to JSON
    print("Computing controller robustness tuples across all predicted trajectories...")

    # Ensure we have X_preds and U_preds (created earlier). Iterate over each predicted trajectory.
    for traj_idx, (X_pred, U_pred) in enumerate(zip(X_preds, U_preds)):
        print(f" Processing trajectory {traj_idx+1}/{len(X_preds)}...") 
        Np = int(X_pred.shape[0])
        # Use x0 from predicted_trajectories if available for metadata
        try:
            x0_meta = predicted_trajectories[traj_idx]['x0']
        except Exception:
            x0_meta = None

        # If the trajectory is empty, skip processing
        if Np <= 0:
            continue

        # Iterate over all valid starting indices; inner loop will break when j >= Np,
        # so we don't need to artificially limit the outer loop by tau_steps.
        for i in range(Np):
            x_i = X_pred[i].astype(float)
            dist_i = state_distance(x_i, x_star)

            best_r = None
            best_tp = None

            # allow tp to grow up to the available horizon (max possible = Np - i - 1)
            for tp in range(1, Np - i):
                j = i + tp
                if j >= Np:
                    break
                dist_j = state_distance(X_pred[j], x_star)
                exp_alpha_t = np.exp(alpha * tp * dt)
                denom = np.exp((alpha + L) * tp * dt) + 1.0
                r_candidate = (dist_i - exp_alpha_t * dist_j) / denom
                if r_candidate >= 0.0:
                    if (best_r is None) or (r_candidate > best_r):
                        best_r = r_candidate
                        best_tp = tp

            if best_r is None:
                all_results_alpha.append((x_i.copy(), None, None, None, traj_idx, x0_meta))
                robust_tuples.append({
                    'x': [float(x_i[0]), float(x_i[1])],
                    'r': None,
                    't': None,
                    'u_seq': None,
                    'source_idx': int(traj_idx),
                    'x0': x0_meta
                })
            else:
                u_seq = []
                for k in range(i, i + best_tp):
                    if k < len(U_pred):
                        u_seq.append([float(np.squeeze(U_pred[k, 0])), float(np.squeeze(U_pred[k, 1]))])
                    else:
                        u_seq.append([0.0, 0.0])
                
                robust_tuples.append({
                    'x': [float(x_i[0]), float(x_i[1])],
                    'r': float(best_r),
                    't': float(best_tp * dt),
                    'u_seq': list(u_seq),
                    'source_idx': int(traj_idx),
                    'x0': x0_meta
                })
                all_results_alpha.append((x_i.copy(), float(best_r), best_tp * dt, u_seq, traj_idx, x0_meta))

                # REMOVED: symmetric entry generation
                # The symmetry (θ, ω) → (-θ, -ω) with control (u, s) → (-u, s)
                # is valid for targets at θ=0, but NOT for θ=π.
    
    # Save aggregated robust tuples (no symmetric entries)
    with open(json_path, 'w') as f:
        json.dump(robust_tuples, f, indent=2)
    print(f"Saved controller robustness tuples for {len(robust_tuples)} entries to '{json_path}'.")

    # Populate results_alpha (no symmetric tuples added)
    results_alpha = []
    for (x, r, t, u_seq, src_idx, x0_meta) in all_results_alpha:
        results_alpha.append((np.array(x, dtype=float), r, t, list(u_seq) if u_seq is not None else None))

else:
    # LOAD_POLICY is True -> load existing JSON produced earlier and build results_alpha
    robust_tuples = []
    results_alpha = []
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            try:
                robust_tuples = json.load(f)
            except Exception:
                robust_tuples = []

        for item in robust_tuples:
            # safe parsing with fallbacks
            x = np.array(item.get('x', [np.nan, np.nan]), dtype=float)
            r = item.get('r', None)
            try:
                r = None if r is None else float(r)
            except Exception:
                r = None
            t = item.get('t', None)
            try:
                t = None if t is None else float(t)
            except Exception:
                t = None
            u_seq = item.get('u_seq', None)
            if u_seq is not None:
                try:
                    # ensure each u entry is flattened; results_alpha expects list of [u,s] or similar
                    u_seq = [list(np.asarray(u).flatten()) for u in u_seq]
                except Exception:
                    u_seq = [list(u) for u in u_seq]
            results_alpha.append((x, r, t, u_seq))
    else:
        # ensure variable exists even if file missing
        results_alpha = []

# --- End of reachability / robustness analysis ---

# --- Histograms: alpha r/duration, and successive state distance ---
# Prepare alpha-only data
r_vals_alpha = np.array([ri for (_, ri, _, _) in results_alpha if ri is not None], dtype=float)
t_vals_alpha = np.array([ti for (_, _, ti, _) in results_alpha if ti is not None], dtype=float)

if PRODUCE_FIGURES:
    # helper: histogram with log-scaled x-axis (skip non-positive)
    def hist_logx(data, title, xlabel, filename, bins_count=50):
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data) & (data > 0.0)]
        if data.size == 0:
            return
        xmin, xmax = data.min(), data.max()
        if xmin <= 0 or xmax <= 0 or not np.isfinite(xmin) or not np.isfinite(xmax):
            return
        bins = np.logspace(np.log10(xmin), np.log10(xmax), bins_count)
        plt.figure()
        plt.hist(data, bins=bins)
        plt.xscale('log')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('count')
        plt.grid(True, which='both', axis='both')
        plt.savefig(os.path.join(out_dir, filename))

    # Alpha histograms (log x-axis)
    hist_logx(r_vals_alpha, 'Histogram of r (alpha)', 'r', 'hist_r_alpha.png')
    hist_logx(t_vals_alpha, 'Histogram of controller duration t (alpha)', 't (s)', 'hist_duration_alpha.png')

    closed_loop_initial_states = [
        np.array([-np.pi/8, 0.0], dtype=float)
    ] + [
        np.array([
            rng_closed_loop.uniform(-np.pi, np.pi),
            rng_closed_loop.uniform(-4.0, 4.0)
        ], dtype=float)
        for _ in range(N_CLOSED_LOOP_RANDOM-1)
    ]

    # ensure max_steps is defined for the helper below
    max_steps = N_closed_loop

    # --- LQR stabilization gain (needed by run_closed_loop) ---
    # Linearize dynamics around upright: theta = pi, theta_dot = 0
    m = 1.0
    l = 2.0
    g = 9.81
    I = m * l**2
    A = np.array([[0.0, 1.0],
                  [m * g * l / I, 0.0]])
    B = np.array([[0.0],
                  [1.0 / I]])
    Q_lqr = np.diag([100.0, 10.0])
    R_lqr = np.array([[0.1]])
    # why the system here is linearized around theta=0 instead of theta=pi is unclear,
    try:
        from scipy.linalg import solve_continuous_are
        P = solve_continuous_are(A, B, Q_lqr, R_lqr)
        K_lqr = np.linalg.inv(R_lqr) @ B.T @ P
    except Exception:
        K_lqr = np.array([[50.0, 20.0]])
    print(f'LQR gain K = {K_lqr}')

    # Helper to run closed-loop for one initial condition and return X,U,H,L
    def run_closed_loop(x0_init, run_idx):
        # ensure MPC initial guess set for this x0
        mpc.x0 = x0_init
        try:
            mpc.set_initial_guess()
        except Exception:
            pass

        # create fresh simulator to avoid state carry-over
        sim = do_mpc.simulator.Simulator(model)
        sim.set_param(t_step=dt)
        sim.x0 = x0_init
        sim.setup()

        X_hist = [get_state_array(sim.x0)]
        U_hist = []
        H_hist = [compute_hamiltonian(X_hist[0][0], X_hist[0][1])]
        # mode history per step: 0 = no special controller, 1 = controller-tuple active, 2 = LQR active
        mode_hist = [0]

        step = 0
        while len(X_hist) < max_steps + 1:
            x_cur = get_state_array(sim.x0)
            dist_to_target = state_distance(x_cur, x_star)

            # LQR region
            if dist_to_target < DIST_THRESHOLD_LQR:
                x_err = x_cur - x_star
                u_lqr = -K_lqr @ x_err
                u_val = float(np.squeeze(u_lqr))
                s_val = 1.0
                u_arr = np.array([[u_val], [s_val]], dtype=float)
                sim.make_step(u_arr)
                x_next = get_state_array(sim.x0)
                X_hist.append(x_next.copy())
                U_hist.append([float(u_val), float(s_val)])
                H_hist.append(compute_hamiltonian(x_next[0], x_next[1]))
                mode_hist.append(2)
                step += 1
                continue

            # policy selection
            candidates = [(idx, xi, RADIUS_AMPLIFY * ri, ti, ui) for idx, (xi, ri, ti, ui) in enumerate(results_alpha)
                          if (ri is not None) and (ri > 0) and (ui is not None) and (ti is not None)]
            best_idx = None
            best_metric = np.inf
            best_entry = None
            for (idx, xi, ri, ti, ui) in candidates:
                d = state_distance(x_cur, np.asarray(xi, dtype=float))
                metric = d / ri
                if metric < best_metric:
                    best_metric = metric
                    best_idx = idx
                    best_entry = (xi, ri, ti, ui)

            if (best_entry is None) or (best_metric >= 1.0):
                u_val = 0.0
                s_val = 1.0
                u_arr = np.array([[u_val], [s_val]], dtype=float)
                sim.make_step(u_arr)
                x_next = get_state_array(sim.x0)
                X_hist.append(x_next.copy())
                U_hist.append([float(u_val), float(s_val)])
                H_hist.append(compute_hamiltonian(x_next[0], x_next[1]))
                mode_hist.append(0)
                step += 1
                continue

            xi, ri, ti, u_seq = best_entry
            steps_to_apply = min(int(np.round(ti / dt)), len(u_seq)) if ti > 0 else 0
            steps_remaining = max_steps + 1 - len(X_hist)
            steps_to_apply = min(steps_to_apply, steps_remaining)

            if steps_to_apply <= 0:
                u_val = 0.0
                s_val = 1.0
                u_arr = np.array([[u_val], [s_val]], dtype=float)
                sim.make_step(u_arr)
                x_next = get_state_array(sim.x0)
                X_hist.append(x_next.copy())
                U_hist.append([float(u_val), float(s_val)])
                H_hist.append(compute_hamiltonian(x_next[0], x_next[1]))
                mode_hist.append(0)
                step += 1
                continue

            for k in range(steps_to_apply):
                u_val = float(np.squeeze(u_seq[k][0]))
                s_val = float(np.squeeze(u_seq[k][1]))
                u_arr = np.array([[u_val], [s_val]], dtype=float)
                sim.make_step(u_arr)
                x_next = get_state_array(sim.x0)
                X_hist.append(x_next.copy())
                U_hist.append([float(u_val), float(s_val)])
                H_hist.append(compute_hamiltonian(x_next[0], x_next[1]))
                mode_hist.append(1)  # tuple applied
                step += 1
                if len(X_hist) >= max_steps + 1:
                    break

        X_arr = np.asarray(X_hist, dtype=float)
        U_arr = np.asarray(U_hist, dtype=float)
        H_arr = np.asarray(H_hist, dtype=float)
        L_arr = np.asarray(mode_hist, dtype=int)  # integer mode array

        # Do NOT save per-run figures here. Return data (including LQR mask) for combined plotting.
        return X_arr, U_arr, H_arr, L_arr

    # Run closed-loop for each initial state and collect trajectories
    all_runs = []
    for idx, x0_candidate in enumerate(closed_loop_initial_states):
        print(f'Running closed-loop from initial {idx}: {x0_candidate}')
        Xr, Ur, Hr, Lr = run_closed_loop(x0_candidate, idx)
        all_runs.append((Xr, Ur, Hr, Lr))

    # Combined plots: overlay all runs in same figures (state-time and phase)
    if PRODUCE_FIGURES_CLOSE_LOOP:
        # Combined state (theta and theta_dot) over time
        plt.figure()
        for idx, (Xr, Ur, Hr, Lr) in enumerate(all_runs):
            t_state = np.arange(Xr.shape[0]) * dt
            plt.plot(t_state, Xr[:, 0], label=f'run_{idx} theta', alpha=0.8)
            plt.plot(t_state, Xr[:, 1], label=f'run_{idx} omega', alpha=0.4, linestyle='--')
        plt.title('Closed-loop State Trajectories (all runs)')
        plt.xlabel('Time (s)')
        plt.ylabel('State value')
        plt.grid(True)
        plt.legend(loc='best', fontsize='small', ncol=2)
        plt.savefig(os.path.join(out_dir, 'closed_loop_state_all_runs.png'))
        plt.close()

        # Combined control overlay (solid lines per run)
        plt.figure()
        for idx, (Xr, Ur, Hr, Lr) in enumerate(all_runs):
            if Ur.size > 0:
                t_ctrl = np.arange(Ur.shape[0]) * dt
                plt.step(t_ctrl, Ur.squeeze(), where='post', alpha=0.8, label=f'run_{idx}')
        plt.title('Closed-loop Control Input (all runs)')
        plt.xlabel('Time (s)')
        plt.ylabel('u')
        plt.grid(True)
        plt.legend(loc='best', fontsize='small')
        plt.savefig(os.path.join(out_dir, 'closed_loop_control_all_runs.png'))
        plt.close()

        # Combined phase plot: overlay trajectories and draw LQR activation region (circle)
        import matplotlib.patches as mpatches
        fig, ax = plt.subplots()
        # draw LQR activation circle once
        lqr_circle = mpatches.Circle((x_star[0], x_star[1]), DIST_THRESHOLD_LQR,
                                    facecolor='gray', alpha=0.25, edgecolor='k', lw=0.5, zorder=0, label='LQR region')
        ax.add_patch(lqr_circle)
        for idx, (Xr, Ur, Hr, Lr) in enumerate(all_runs):
            ax.plot(Xr[:, 0], Xr[:, 1], '-', lw=1, label=f'run_{idx}')
        ax.set_title('Closed-loop Phase Plot (all runs)')
        ax.set_xlabel('theta (rad)')
        ax.set_ylabel('theta_dot (rad/s)')
        ax.grid(True)
        try:
            ax.set_aspect('equal', adjustable='datalim')
        except Exception:
            pass
        ax.legend(loc='best', fontsize='small')
        plt.savefig(os.path.join(out_dir, 'closed_loop_phase_all_runs.png'))
        plt.close()

# --- Animation helper: render a pendulum video from one initial condition ---
def save_pendulum_animation_from_x0(x0, filename='inverted_pendulum.mp4', fps=30, dpi=150):
    """
    Run closed-loop from x0 and save an animation showing pendulum rod and tip.
    Falls back to GIF if FFMpegWriter not available.
    """
    # run closed-loop (uses run_closed_loop defined above)
    Xr, Ur, Hr, Lr = run_closed_loop(np.array(x0, dtype=float), 0)

    Ur_arr = np.asarray(Ur, dtype=float)
    if Ur_arr.ndim == 1:
        Ur_arr = Ur_arr.reshape(-1, 1)

    if Ur_arr.shape[0] > 0:
        s_hist = Ur_arr[:, 1] if Ur_arr.shape[1] > 1 else np.ones(Ur_arr.shape[0], dtype=float)
        real_time = np.concatenate([[0.0], np.cumsum(s_hist * dt)])
    else:
        real_time = np.linspace(0.0, max((Xr.shape[0] - 1) * dt, 0.0), Xr.shape[0])

    total_time = real_time[-1] if real_time.size > 0 else 0.0
    uniform_time = np.arange(0.0, total_time + dt * 0.5, dt) if total_time > 0.0 else np.array([0.0])

    print(f"Creating animation from x0={x0}, total_time={total_time:0.2f}s, frames={uniform_time.size}")    

    def _interp_nd(time_src, data, time_target):
        if time_target.size == 0:
            if data.ndim == 1:
                return np.zeros((0,), dtype=float)
            return np.zeros((0, data.shape[1]), dtype=float)
        if data.ndim == 1 or data.shape[1] == 1:
            return np.interp(time_target, time_src, data if data.ndim == 1 else data.flatten())
        return np.column_stack([np.interp(time_target, time_src, data[:, dim]) for dim in range(data.shape[1])])

    ctrl_time = real_time[:-1] if real_time.size > 1 else np.array([], dtype=float)
    target_ctrl_time = uniform_time[:-1] if uniform_time.size > 1 else np.array([], dtype=float)

    Xr = _interp_nd(real_time, Xr, uniform_time)
    Hr = _interp_nd(real_time, Hr, uniform_time)
    Ur = _interp_nd(ctrl_time, Ur_arr, target_ctrl_time)

    # --- interpolate integer mode history (Lr) to the uniform timeline ---
    try:
        Lr_arr = np.asarray(Lr, dtype=float)
        Lr_interp = _interp_nd(real_time, Lr_arr, uniform_time)
        # boolean masks per frame
        LQR_active = (np.asarray(Lr_interp, dtype=float) >= 1.5)  # mode==2
        TUPLE_active = np.round(np.asarray(Lr_interp, dtype=float)) == 1  # mode==1
    except Exception:
        LQR_active = np.zeros_like(uniform_time, dtype=bool)
        TUPLE_active = np.zeros_like(uniform_time, dtype=bool)

    frame_times = uniform_time

    # define pendulum length BEFORE computing tips
    l = 2.0  # pendulum length (must match model)

    # compute tip coordinates after l is defined
    x_tip = l * np.sin(Xr[:, 0])
    y_tip = -l * np.cos(Xr[:, 0])
    frames = Xr.shape[0]

    # figure setup
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-l * 1.2, l * 1.2)
    ax.set_ylim(-l * 1.2, l * 1.2)
    ax.set_aspect('equal')
    # disable grid and ticks for a clean slide-friendly render
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # also hide tick labels (redundant with set_*ticks but explicit)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Rod: black line (rotates about pivot at origin)
    rod_line, = ax.plot([], [], '-', lw=2, color='k', zorder=2)

    # Bob: blue filled circle, larger radius for visibility
    bob_radius = 0.055 * l  # tune for size
    # explicit face and edge colors so we can change both during animation
    bob_patch = plt.Circle((0.0, -l),
                           bob_radius,
                           facecolor='C0',
                           edgecolor='k',
                           linewidth=1,
                           zorder=3)

    # Pivot: small black dot at origin
    pivot_dot, = ax.plot(0.0, 0.0, 'ko', markersize=4, zorder=4)

    ax.add_patch(bob_patch)

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        rod_line.set_data([], [])
        bob_patch.set_center((0.0, -l))
        time_text.set_text('')
        return rod_line, bob_patch, time_text

    def update(i):
        xt = x_tip[i]
        yt = y_tip[i]
        # rod from pivot (0,0) to bob (xt,yt)
        rod_line.set_data([0.0, xt], [0.0, yt])
        # move bob patch
        bob_patch.set_center((xt, yt))
        # change bob color to red when LQR is active at this frame
        try:
            if LQR_active[i]:
                face_col = 'red'
                edge_col = '#8B0000'  # dark red border
            elif TUPLE_active[i]:
                face_col = '#90EE90'  # lightgreen fill
                edge_col = '#3CB371'  # mediumseagreen border
            else:
                face_col = 'C0'       # original blue fill
                edge_col = 'k'        # black border
            bob_patch.set_facecolor(face_col)
            bob_patch.set_edgecolor(edge_col)
            bob_patch.set_linewidth(1.5)
        except Exception:
            bob_patch.set_facecolor('C0')
            bob_patch.set_edgecolor('k')
            bob_patch.set_linewidth(1.5)
        time_text.set_text(f't={frame_times[i]:0.2f}s')
        return rod_line, bob_patch, time_text

    # Compute animation timing so saved video duration matches total_time
    if total_time > 0 and frames > 1:
        fps_anim = float(frames) / float(total_time)
        interval_ms = int(np.round(1000.0 * total_time / float(frames)))
    else:
        fps_anim = float(fps)
        interval_ms = int(np.round(dt * 1000))

    anim = animation.FuncAnimation(fig, update, frames=frames, init_func=init,
                                   blit=True, interval=interval_ms)

    out_path = os.path.join(out_dir, filename)
    # If the caller passed a path (absolute or with directories), use it directly;
    # otherwise join with our default out_dir.
    if os.path.dirname(filename):
        out_path = filename
    else:
        out_path = os.path.join(out_dir, filename)

    # Ensure destination directory exists
    dest_dir = os.path.dirname(out_path) if os.path.dirname(out_path) else out_dir
    os.makedirs(dest_dir, exist_ok=True)

    # Try saving with ffmpeg, else save as gif via Pillow
    try:
        print(f"Saving animation to '{out_path}' with ffmpeg (fps={fps_anim:.2f})...")
        writer = animation.FFMpegWriter(fps=fps_anim)
        anim.save(out_path, writer=writer, dpi=dpi)
        print(f"Saved animation: {out_path}")
    except Exception as e_ffmpeg:
        # fallback to gif
        try:
            gif_path = os.path.splitext(out_path)[0] + '.gif'
            os.makedirs(os.path.dirname(gif_path) or out_dir, exist_ok=True)
            print(f"FFmpeg save failed ({e_ffmpeg}); falling back to GIF at '{gif_path}'...")
            writer = animation.PillowWriter(fps=fps_anim)
            anim.save(gif_path, writer=writer, dpi=dpi)
            print(f"Saved animation (GIF): {gif_path}")
        except Exception as e_gif:
            print(f"Failed to save animation: ffmpeg error: {e_ffmpeg}; gif error: {e_gif}")
    plt.close(fig)


# --- New helper: create animation from one stored predicted trajectory entry ---
def save_pendulum_animation_from_prediction(pred_entry, filename='predicted_traj.mp4', fps=30, dpi=150,
                                            bob_face='C0', bob_edge='k'):
    """
    pred_entry: dict with keys 'X_pred' (Nx2) and 'U_pred' (N-1 x 2) as in predicted_trajectories
    Produces a video whose duration matches the real_time computed from s(t) in U_pred.
    """
    # Accept either raw arrays or dict entry
    if isinstance(pred_entry, dict):
        X_pred = np.asarray(pred_entry.get('X_pred', []), dtype=float)
        U_pred = np.asarray(pred_entry.get('U_pred', []), dtype=float)
    else:
        # assume it's already an array-like tuple (X_pred, U_pred)
        X_pred, U_pred = map(np.asarray, pred_entry)

    if X_pred.size == 0:
        print("Empty prediction, skipping video.")
        return

    # Ensure shapes: X_pred (M,2), U_pred (M-1,2) or (M-1,1)
    if U_pred.ndim == 1:
        U_pred = U_pred.reshape(-1, 1)
    if U_pred.shape[1] == 1:
        s_col = np.ones((U_pred.shape[0], 1), dtype=float)
        U_pred = np.hstack([U_pred, s_col])

    # compute real_time from s
    s_hist = np.squeeze(U_pred[:, 1])
    real_time = np.concatenate([[0.0], np.cumsum(s_hist * dt)]) if s_hist.size > 0 else np.arange(X_pred.shape[0]) * dt
    total_time = float(real_time[-1]) if real_time.size > 0 else 0.0
    uniform_time = np.arange(0.0, total_time + dt * 0.5, dt) if total_time > 0.0 else np.array([0.0])

    # interpolation helper (works for 1D or 2D)
    def _interp_nd(time_src, data, time_target):
        if time_target.size == 0:
            return np.zeros((0, data.shape[1])) if data.ndim > 1 else np.zeros((0,))
        if data.ndim == 1 or (data.ndim > 1 and data.shape[1] == 1):
            return np.interp(time_target, time_src, data if data.ndim == 1 else data.flatten())
        return np.column_stack([np.interp(time_target, time_src, data[:, dim]) for dim in range(data.shape[1])])

    Xu = _interp_nd(real_time, X_pred, uniform_time)
    # compute tips
    l_local = 2.0
    x_tip = l_local * np.sin(Xu[:, 0])
    y_tip = -l_local * np.cos(Xu[:, 0])
    frames = Xu.shape[0]

    # Plot setup (match closed-loop style)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-l_local * 1.2, l_local * 1.2)
    ax.set_ylim(-l_local * 1.2, l_local * 1.2)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    rod_line, = ax.plot([], [], '-', lw=2, color='k', zorder=2)
    bob_radius = 0.055 * l_local
    bob_patch = plt.Circle((0.0, -l_local), bob_radius, facecolor=bob_face, edgecolor=bob_edge, linewidth=1.0, zorder=3)
    ax.add_patch(bob_patch)
    pivot_dot, = ax.plot(0.0, 0.0, 'ko', markersize=4, zorder=4)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        rod_line.set_data([], [])
        bob_patch.set_center((0.0, -l_local))
        time_text.set_text('')
        return rod_line, bob_patch, time_text

    def update(i):
        xt = x_tip[i]
        yt = y_tip[i]
        rod_line.set_data([0.0, xt], [0.0, yt])
        bob_patch.set_center((xt, yt))
        time_text.set_text(f't={uniform_time[i]:0.2f}s')
        return rod_line, bob_patch, time_text

    # Compute fps so saved video lasts total_time
    if total_time > 0 and frames > 1:
        fps_anim = float(frames) / float(total_time)
        interval_ms = int(np.round(1000.0 * total_time / float(frames)))
    else:
        fps_anim = float(fps)
        interval_ms = int(np.round(dt * 1000))

    anim = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=interval_ms)

    out_path = os.path.join(out_dir, filename)
    # If the caller passed a path (absolute or with directories), use it directly;
    # otherwise join with our default out_dir.
    if os.path.dirname(filename):
        out_path = filename
    else:
        out_path = os.path.join(out_dir, filename)

    # Ensure destination directory exists
    dest_dir = os.path.dirname(out_path) if os.path.dirname(out_path) else out_dir
    os.makedirs(dest_dir, exist_ok=True)

    try:
         writer = animation.FFMpegWriter(fps=fps_anim)
         anim.save(out_path, writer=writer, dpi=dpi)
         print(f"Saved animation: {out_path}")
    except Exception as e_ffmpeg:
        # fallback to gif
        try:
            gif_path = os.path.splitext(out_path)[0] + '.gif'
            os.makedirs(os.path.dirname(gif_path) or out_dir, exist_ok=True)
            writer = animation.PillowWriter(fps=fps_anim)
            anim.save(gif_path, writer=writer, dpi=dpi)
            print(f"Saved animation (GIF): {gif_path}")
        except Exception as e_gif:
            print(f"Failed to save animation: ffmpeg error: {e_ffmpeg}; gif error: {e_gif}")
    plt.close(fig)


# --- Batch-save predicted trajectories as animations (one video per predicted trajectory) ---
if len(predicted_trajectories) > 0:
    os.makedirs(out_dir, exist_ok=True)
    for idx, pred in enumerate(predicted_trajectories):
        fname = os.path.join(out_dir, f'predicted_traj_{idx}.mp4')
        try:
            save_pendulum_animation_from_prediction(pred, filename=fname, fps=30, dpi=150)
        except Exception as e:
            print(f"Failed to create predicted animation {idx}: {e}")
