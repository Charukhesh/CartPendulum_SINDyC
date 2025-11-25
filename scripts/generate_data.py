import numpy as np
from scipy.integrate import solve_ivp
import src.config as config
from src.controller import design_lqr_controller
from src.dynamics import inverted_pendulum_dynamics

def generate_training_data():
    """
    Simulates the inverted pendulum with an LQR controller and saves the data
    """
    print("--- Generating Training Data ---")
    # Design the controller
    K = design_lqr_controller(config)

    # Stabilize around the upright position
    equilibrium_state = np.array([0, 0, np.pi, 0])

    # Define the control law u = -Kx
    def controller(t, x_state):
        error_state = x_state - equilibrium_state
        return (-K @ error_state).item()# + 1.5
    
    def tracking_controller(t, x_state):
        # Define the desired trajectory for the cart's position
        x_desired = 0.5 * np.sin(1.0 * t)
        v_desired = 0.5 * 1.0 * np.cos(1.0 * t) # Derivative of x_desired
        
        # The desired state is to follow the path while balanced
        desired_state = np.array([x_desired, v_desired, np.pi, 0])
        
        # LQR control law now tracks the error from the moving target
        error = x_state - desired_state
        return (-K @ error).item()

    def indep_controller(t, x_state):
        return -0.1#0.1*np.sin(2*np.pi*t) + 0.1*np.cos(2*np.pi*t)

    # The dynamics function for solve_ivp needs to use the controller
    def controlled_system(t, x_state):
        return inverted_pendulum_dynamics(t, x_state, u_func=controller, params=config)
    
    # Time vector for simulation
    t_span = [0, config.T_END]
    t_eval = np.arange(0, config.T_END, config.DT)

    # Run the simulation
    print(f"Simulating from initial state: {config.X0}")
    sol = solve_ivp(
        controlled_system, 
        t_span, 
        config.X0, 
        t_eval=t_eval, 
        dense_output=True
    )
    
    # Extract states and calculate control inputs used
    x_train = sol.y.T  # Transpose to get (n_samples, n_features)
    t_train = sol.t
    
    # Recompute the control inputs used at each time step
    u_train = np.array([controller(t, state) for t, state in zip(t_train, x_train)]).reshape(-1, 1)

    # State Derivatives
    dx_dt_train = np.array([controlled_system(t, state) for t, state in zip(t_train, x_train)])
    # dx_dt_train = np.gradient(x_train, axis=0) / config.DT

    '''
    v, theta, omega, u = x_train[:, 1], x_train[:, 2], x_train[:, 3], u_train
    m, M, L, g = config.m, config.M, config.L, config.g
    # Derivatives
    dxdt = v
    dvdt = (u + m*L*omega**2*np.sin(theta) - m*g*np.sin(theta)*np.cos(theta)) / (M + m*np.sin(theta)**2)
    dthetadt = omega
    domegadt = (-u*np.cos(theta) - m*L*omega**2*np.sin(theta)*np.cos(theta) + (M+m)*g*np.sin(theta)) / (L * (M + m*np.sin(theta)**2))

    dx_dt_train = [dxdt, dvdt, dthetadt, domegadt]
    '''

    # Save the data
    np.savez(
        config.TRAIN_DATA_PATH,
        x=x_train,
        u=u_train,
        t=t_train,
        dx_dt=dx_dt_train
    )
    print(f"Training data successfully generated and saved to {config.TRAIN_DATA_PATH}")

    return x_train, u_train, t_train, dx_dt_train
