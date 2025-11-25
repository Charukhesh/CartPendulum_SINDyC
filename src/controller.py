import numpy as np
import control as ct
import src.config as config

def get_linearized_model(params=config):
    """
    Linearizes the inverted pendulum dynamics around the upright equilibrium point
    (theta=pi, x_dot=0, theta_dot=0)
    The state-space form is x_dot = A*x + B*u.
    
    Returns:
        A (np.array): State matrix.
        B (np.array): Input matrix.
    """
    M, m, L, g, b, I = params.M, params.m, params.L, params.g, params.b, params.I
    
    # Using Taylor expansion around theta = pi (upright position)
    # Denominator of the solved system for accelerations:
    den = (I + m * L**2) * (M + m) - (m * L)**2
    
    # From solving the linearized system for x_ddot and theta_ddot in terms of states and F:
    A = np.array([
        [0, 1, 0, 0],
        [0, -b * (I + m * L**2) / den,  (m**2 * g * L**2) / den,  0],
        [0, 0, 0, 1],
        [0, -b * m * L / den,           m * g * L * (M + m) / den, 0]
    ])

    B = np.array([
        [0],
        [(I + m * L**2) / den],
        [0],
        [m * L / den]
    ])
    
    return A, B

def design_lqr_controller(params=config):
    """
    Designs an LQR controller for the inverted pendulum
    
    Returns:
        K (np.array): The optimal gain matrix for the control law u = -K*x
    """
    A, B = get_linearized_model(params)
    Q = np.array(params.Q)
    R = np.array([[params.R]]) # R must be a 2D array
    
    # Use the control library to solve the continuous-time algebraic Riccati equation
    K, _, _ = ct.lqr(A, B, Q, R) 
    return K