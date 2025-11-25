import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import scripts.generate_data as generate_data
from src.discover_model import create_sindy_dynamics_implicit_solver, extract_coefficients, discover_model_stslq, plot_comparison
import scripts.test_model as test_model
from src.controller import design_lqr_controller
from scipy.integrate import solve_ivp
import src.visualise as visualise
import src.config as config
import matplotlib.pyplot as plt

def main():
    """
    Main function to run the complete SINDy pipeline for the inverted pendulum.
    """

    # Generate training data from a controlled simulation
    generate_data.generate_training_data()
    data = np.load(config.TRAIN_DATA_PATH)
    t_data, X_data, u_data, dX_data = data['t'], data['x'], data['u'].flatten(), data['dx_dt']

    # print("\nDisplaying animation of the controlled (training) phase...")
    #path = "./data/baseline/"
    # visualise.animate_pendulum(t_data, X_data, config.TRAIN_GIF_PATH)
    
    print("Displaying state variable plots...")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    state_labels = [
        r'Cart Position $x$ (m)', 
        r'Cart Velocity $\dot{x}$ (m/s)', 
        r'Pendulum Angle $\theta$ (rad)', 
        r'Pendulum Angular Velocity $\dot{\theta}$ (rad/s)'
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Blue, Orange, Green, Red
    axs_flat = axs.flatten()
    
    for i in range(4):
        ax = axs_flat[i] 
        ax.plot(t_data, X_data[:, i], color=colors[i], linewidth=2)
        ax.set_title(state_labels[i], fontsize=12)
        ax.grid(True)
        
        ax.legend()
        if i >= 2:
            ax.set_xlabel("Time (s)")

    fig.suptitle("State Trajectories During Controlled (Training) Phase", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Phase plot of the pendulum's angle vs. angular velocity
    plt.figure(figsize=(8, 6))
    plt.plot(X_data[:, 2], X_data[:, 3])
    # Red dot at the starting point and a green dot at the equilibrium point
    plt.plot(X_data[0, 2], X_data[0, 3], 'ro', markersize=10, label='Start')
    plt.plot(np.pi, 0, 'go', markersize=10, label='Equilibrium (Setpoint)')
    plt.title("Pendulum Phase Plot During Controlled (Training) Phase")
    plt.xlabel("Pendulum Angle $\\theta$ (rad)")
    plt.ylabel(r"Pendulum Angular Velocity $\dot{\theta}$ (rad/s)")
    plt.grid(True)
    plt.legend()
    plt.axis('equal') # Helps to visualize the dynamics more intuitively
    plt.show()
    
    # Use SINDy to discover the governing equations from the data
    state_names = ['x', 'v', 'theta', 'omega']
    implicit_equations = discover_model_stslq(X_data, u_data, dX_data, state_names)

    if implicit_equations:
        coeffs_list = extract_coefficients(implicit_equations, state_names)
        K = design_lqr_controller(config)
        equilibrium_state = np.array([0, 0, np.pi, 0])

        def controller(t, x_state):
            return (-K @ (x_state - equilibrium_state)).item() + 1.5

        def get_desired_state(t):
            x_desired = 0.5 * np.sin(1.0 * t)
            v_desired = 0.5 * 1.0 * np.cos(1.0 * t)
            return np.array([x_desired, v_desired, equilibrium_state[2], equilibrium_state[3]])

        def tracking_controller(t, x_state):
            desired_state = get_desired_state(t)
            error = x_state - desired_state
            return (-K @ error).item()
        
        def indep_controller(t, x_state):
            return -0.1#2.5 * np.sin(0.5 * t) + 1.5 * np.cos(1.2 * t + 0.5) - 2.0 * np.sin(2.5 * t - 1.0)

        sindy_dynamics_func = create_sindy_dynamics_implicit_solver(coeffs_list, state_names, controller_func=controller)
        
        if sindy_dynamics_func:
            from src.dynamics import inverted_pendulum_dynamics
            def controlled_system_clean(t, x_state):
                 return inverted_pendulum_dynamics(t, x_state, u_func=controller, params=config)
            
            y0 = X_data[0, :]
            t_eval = t_data
            sol_original = solve_ivp(controlled_system_clean, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval)

            sindy_sim_results = solve_ivp(sindy_dynamics_func, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval, method='RK45')

            plot_comparison(t_eval, sol_original.y.T, sindy_sim_results, state_names)
    else:
        print("\nModel discovery failed. Cannot proceed to simulation.")

    # Validate the discovered model on an uncontrolled, unseen scenario
    test_model.test_discovered_model(sindy_dynamics_func)
    
if __name__ == "__main__":
    main()