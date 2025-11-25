import numpy as np

# Physical Parameters of the Inverted Pendulum
M = 1.0       # Mass of the cart (kg)
m = 0.2       # Mass of the pendulum bob (kg)
L = 0.5       # Length of the pendulum rod (m)
g = 9.81      # Acceleration due to gravity (m/s^2)
b = 0.1       # Friction coefficient of the cart (N/(m/s))
I = 0.006     # Moment of inertia of the pendulum (kg*m^2)

# Simulation Parameters
DT = 0.01                # Time step for simulation (s)
T_END = 10.0             # Total simulation time (s)
X0 = [1.0, 1.0, np.pi + 0.2, 1.0] # Initial state [x, x_dot, theta, theta_dot]

# LQR Controller Parameters
# Cost matrices for LQR
Q = [[1.0, 0.0, 0.0, 0.0],    # Penalizes x
     [0.0, 1.0, 0.0, 0.0],    # Penalizes x_dot
     [0.0, 0.0, 10.0, 0.0],   # Penalizes theta (most important)
     [0.0, 0.0, 0.0, 1.0]]    # Penalizes theta_dot
R = 0.001                    # Penalizes control effort (force F)

# SINDy Parameters
SINDY_THRESHOLD = 1e-2      # Sparsity threshold for STLSQ algorithm

# File Paths
TRAIN_DATA_PATH = "./data/pendulum_train_data.npz"
TRAIN_GIF_PATH = "./data/pendulum_training_animation.gif"