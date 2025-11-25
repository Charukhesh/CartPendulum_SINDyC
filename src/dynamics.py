import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import src.config as config

def inverted_pendulum_dynamics(t, x_state, u_func=None, params=config):
    """
    Computes the state derivatives for the inverted pendulum on a cart.
    This function solves the system of coupled ODEs for [x_ddot, theta_ddot].
    
    The governing equations are:
    (M+m)x_ddot + b*x_dot + m*L*cos(theta)*theta_ddot - m*L*sin(theta)*theta_dot^2 = F
    (I + m*L^2)*theta_ddot + m*L*cos(theta)*x_ddot + m*g*L*sin(theta) = 0
    
    Args:
        t (float): Current time (required by solve_ivp)
        x_state (np.array): Current state vector [x, x_dot, theta, theta_dot]
        u_func (callable): Function that returns control force F for a given state. If None, F=0
        params: Configuration object with physical parameters

    Returns:
        np.array: The state derivative vector [x_dot, x_ddot, theta_dot, theta_ddot].
    """
    M, m, L, g, b, I = params.M, params.m, params.L, params.g, params.b, params.I
    
    x, x_dot, theta, theta_dot = x_state
    
    # Calculate the control force F
    F = u_func(t, x_state) if u_func is not None else 0.0

    # x_ddot = (F + m*L*np.sin(theta)*theta_dot**2 - (m**2*g*L**2*np.sin(theta)*np.cos(theta)/(I+m*L**2)) - b*x_dot) / ((M+m) - m**2*L**2*np.cos(theta)**2/(I+m*L**2))
    # theta_ddot = (F*m*L*np.cos(theta) + (M+m)*m*g*L*np.sin(theta) + m**2*L**2*np.sin(theta)*np.cos(theta)*theta_dot**2 - b*m*L*x_dot*np.cos(theta)) / ((M+m)*(I+m*L**2) - m**2*L**2*np.cos(theta)**2)

    # We need to solve a 2x2 linear system M * [x_ddot, theta_ddot]^T = B for the accelerations
    # [ M11  M12 ] [ x_ddot     ] = [ B1 ]
    # [ M21  M22 ] [ theta_ddot ] = [ B2 ]
    
    M11 = M + m
    M12 = m * L * np.cos(theta)
    M21 = m * L * np.cos(theta)
    M22 = I + m * L**2
    
    mass_matrix = np.array([[M11, M12], [M21, M22]])
    
    B1 = F - b * x_dot + m * L * np.sin(theta) * theta_dot**2
    B2 = -m * g * L * np.sin(theta)
    
    b_vector = np.array([B1, B2])
    
    # Solve for accelerations
    try:
        accelerations = np.linalg.solve(mass_matrix, b_vector)
    except np.linalg.LinAlgError:
        print("Error: Mass matrix is singular.")
        return np.array([0,0,0,0])

    x_ddot = accelerations[0]
    theta_ddot = accelerations[1]
    
    return np.array([x_dot, x_ddot, theta_dot, theta_ddot])