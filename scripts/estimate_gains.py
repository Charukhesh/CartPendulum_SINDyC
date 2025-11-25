import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import src.config as config
from src.dynamics import inverted_pendulum_dynamics
from src.controller import design_lqr_controller
from src.filter import RecursiveLeastSquares

def compute_numerical_accelerations(x_data, t):
    dt = np.mean(np.diff(t))
    
    # Direct differentiation (No smoothing)
    # We use second order central difference for accuracy
    x = x_data[:, 0]
    v = x_data[:, 1]
    theta = x_data[:, 2]
    omega = x_data[:, 3]
    
    # Acceleration is derivative of velocity
    x_ddot = np.gradient(v, dt, edge_order=2)
    theta_ddot = np.gradient(omega, dt, edge_order=2)
    
    return x_ddot, theta_ddot

def get_open_loop_data():
    """STAGE 1: Generate 'Rich' Data (Open Loop) for Structure Discovery"""
    def u_func(t, x): return 5.0 * np.sin(2 * np.pi * 0.5 * t)
    
    sol = solve_ivp(
        lambda t, x: inverted_pendulum_dynamics(t, x, u_func, config),
        [0, 5], config.X0, t_eval=np.arange(0, 5, config.DT), dense_output=True
    )
    return sol.t, sol.y.T, u_func

def get_closed_loop_data():
    """STAGE 2: Generate 'Poor' Data (Closed Loop) for Parameter Refinement"""
    K_true = design_lqr_controller(config)
    target = np.array([0, 0, np.pi, 0])
    
    def u_func(t, x): return (-K_true @ (x - target)).item()
    
    sol = solve_ivp(
        lambda t, x: inverted_pendulum_dynamics(t, x, u_func, config),
        [0, 10], [0, 0, np.pi+0.4, 0], t_eval=np.arange(0, 10, config.DT),
        rtol=1e-9, atol=1e-9
    )
    # Add small noise
    #meas_noise = 0.0001 * np.random.randn(*sol.y.T.shape)
    return sol.t, sol.y.T, K_true

if __name__ == "__main__":
    
    print("Stage 1: SINDy identifying the inertial physics structure.")
    physics_model = {
        'M_total': config.M + config.m,
        'mL': config.m * config.L,
        'gravity_term': config.m * config.g * config.L,
        'b_friction': config.b 
    }
    
    t_cl, x_data, K_true = get_closed_loop_data()
    x_ddot, theta_ddot = compute_numerical_accelerations(x_data, t_cl)
    
    rls = RecursiveLeastSquares(n_features=4)
    history = []
    target = np.array([0, 0, np.pi, 0])

    print("Running RLS to decouple Friction and Controller...")
    for i in range(len(t_cl)):
        x, v, th, w = x_data[i]
        
        # LHS: Total Forces (Inertial + Physical Friction)
        # We use our Stage 1 knowledge to subtract the physical friction 'b*v'
        # leaving only the Control Force 'u' as the remainder.
        inertial_term = (
            physics_model['M_total'] * x_ddot[i] + 
            physics_model['mL'] * np.cos(th) * theta_ddot[i] - 
            physics_model['mL'] * np.sin(th) * w**2
        )
        
        # Equation: Inertial_Term = u - b*v
        # Rearranged: Inertial_Term + b*v = u
        # And we know u = -K * error
        
        # So: LHS = Inertial_Term + b*v
        lhs_value = inertial_term + physics_model['b_friction'] * v
        
        # RHS: -K * error
        error = x_data[i] - target
        phi = -error # Regressor is just the negative error state
        
        params = rls.step(phi, lhs_value)
        history.append(params.copy())

    K_est = history[-1]
    
    print("\n--- IDENTIFICATION RESULTS ---")
    print(f"True LQR Gains:  {K_true.flatten()}")
    print(f"Estimated Gains: {K_est}")
    
    err_k = np.linalg.norm(K_true.flatten() - K_est)
    print(f"\nController Identification Error: {err_k:.5f}")

    K_history = np.array(history)
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'lines.linewidth': 1.5,
        'legend.fontsize': 8
    })

    fig, axs = plt.subplots(2, 1, figsize=(4.0, 6.0), constrained_layout=True)
    t_plot = t_cl 

    labels = [r'$K_x$', r'$K_{\dot{x}}$', r'$K_{\theta}$', r'$K_{\dot{\theta}}$']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i in range(4):
        axs[0].plot(t_plot, K_history[:, i], color=colors[i], label=f'Est. {labels[i]}')
        axs[0].axhline(K_true.flatten()[i], color=colors[i], linestyle='--', linewidth=1.0, alpha=0.6)

    axs[0].set_ylabel('Gain Magnitude')
    axs[0].set_title('Convergence of Controller Gains')
    axs[0].grid(True, linestyle=':', alpha=0.7)
    axs[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), 
                  ncol=2, frameon=False, borderaxespad=0, fontsize=11)

    error_norm = np.linalg.norm(K_history - K_true.flatten(), axis=1)
    
    axs[1].plot(t_plot, error_norm, 'k-', linewidth=1.5)
    axs[1].set_ylabel(r'$||K_{est} - K_{true}||_2$')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_title('Parameter Estimation Error')
    axs[1].set_yscale('log')
    axs[1].grid(True, which="both", linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    plt.show()