import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import src.config as config
from src.dynamics import inverted_pendulum_dynamics
from src.discover_model import simulate_sindy_model
from src.controller import design_lqr_controller
import src.visualise as visualise

def test_discovered_model(sindy_model):
    """Validates the SINDy model by comparing its prediction on an unseen trajectory against the true system dynamics"""
    print("\n--- Validating Discovered Model ---")
        
    # Define an unseen, challenging initial condition (uncontrolled)
    x0_test = [1.5, 2, np.pi+1, 2.5]
    t_end_test = 10
    t_test = np.arange(0, t_end_test, config.DT)

    print(f"Validation scenario: initial state = {x0_test}")

    K = design_lqr_controller(config)
    equilibrium_state = np.array([0, 0, np.pi, 0])
    def controller(t, x_state):
        error_state = x_state - equilibrium_state
        return (-K @ error_state).item() + 1*np.sin(2*np.pi*t)
    
    def tracking_controller(t, x_state):
        # Define the desired trajectory for the cart's position
        x_desired = 0.5 * np.sin(1.0 * t)
        v_desired = 0.5 * 1.0 * np.cos(1.0 * t) # Derivative of x_desired
        
        # The desired state is to follow the path while balanced
        desired_state = np.array([x_desired, v_desired, np.pi, 0])
        
        # LQR control law now tracks the error from the moving target
        error = x_state - desired_state
        return (-K @ error).item() # + 1*np.sin(2*np.pi*t)
    
    # Simulate the TRUE system dynamics for this test case
    def true_system_uncontrolled(t, x_state):
        return inverted_pendulum_dynamics(t, x_state, u_func=None, params=config)
    
    sol_true = solve_ivp(
        true_system_uncontrolled, 
        [0, t_end_test], 
        x0_test, 
        t_eval=t_test
    )
    x_true = sol_true.y.T

    # Simulate the DISCOVERED SINDy model for the same test case
    # Use the new function to run the simulation
    simulation_result = simulate_sindy_model(
        sindy_model,
        x0_test,
        [0, t_end_test],
        config.DT
    )
    x_sindy = simulation_result.y.T

    plt.rcParams.update({
        'font.family': 'serif',      
        'font.size': 9,              
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'lines.linewidth': 1.2
    })

    fig, axs = plt.subplots(4, 1, figsize=(3.5, 8.5), sharex=True)
    y_labels = [r'$x$ (m)', r'$\dot{x}$ (m/s)', r'$\theta$ (rad)', r'$\dot{\theta}$ (rad/s)']
    state_names = [r'Position', r'Velocity', r'Angle', r'Ang. Vel.']

    marker_freq = int(len(t_test) / 25) 
    for i in range(4):
        true_signal = x_true[:, i]
        sindy_signal = x_sindy[:, i]
        
        # Calculate Relative Error
        true_norm = np.linalg.norm(true_signal)
        error_norm = np.linalg.norm(true_signal - sindy_signal)
        
        if true_norm < 1e-12:
            rel_error = 0.0
        else:
            rel_error = (error_norm / true_norm) * 100

        axs[i].plot(t_test, true_signal, 'k-', alpha=0.8, label='True Dynamics')
        
        axs[i].plot(t_test[::marker_freq], sindy_signal[::marker_freq], 'r--o', 
                   markersize=3, markerfacecolor='white', label='SINDy Model')
        
        axs[i].grid(True, linestyle=':', alpha=0.7)
        axs[i].set_ylabel(y_labels[i], fontsize=10)
        
        # Error info in the subplot title saves legend space
        axs[i].set_title(f"{state_names[i]} (Error: {rel_error:.2f}%)", pad=2, fontsize=11)

    axs[-1].set_xlabel('Time (s)')
    handles, labels = axs[0].get_legend_handles_labels()
    
    fig.legend(handles, labels, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 0.99), 
               ncol=2,
               fontsize=10, 
               frameon=False)    
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.show()
    # path = "./data/validate"
    # visualise.animate_pendulum(t_test, x_sindy, path)

    

    
