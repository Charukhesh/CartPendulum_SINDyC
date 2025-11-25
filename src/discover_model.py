import numpy as np
import sympy
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def stlsq(X, y, threshold, max_iter=20):
    """ Implements the Sequentially Thresholded Least Squares (STLSQ) algorithm """
    try:
        w = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError: return np.zeros(X.shape[1])
    for _ in range(max_iter):
        small_coeffs = np.abs(w) < threshold; w[small_coeffs] = 0
        big_coeffs = ~small_coeffs
        if np.sum(big_coeffs) == 0: break
        try:
            w[big_coeffs] = np.linalg.lstsq(X[:, big_coeffs], y, rcond=None)[0]
        except np.linalg.LinAlgError: break
    return w

def discover_model_stslq(X, u, dX, state_names):
    """
    Discovers the model using a robust, two-pass STLSQ SINDy formulation
    to ensure two independent equations are found.
    """
    print("\n--- Discovering model with Two-Pass STLSQ-SINDy ---")
    
    dvdt_sym, domegadt_sym = sympy.symbols(['dv/dt', 'domega/dt'])
    x_sym, v_sym, theta_sym, omega_sym = sympy.symbols(state_names)
    u_sym = sympy.symbols('u_1')
    
    x, v, theta, omega = X[:,0], X[:,1], X[:,2], X[:,3]
    F = u.flatten()
    dvdt, domegadt = dX[:,1], dX[:,3]

    full_Theta = np.column_stack([
        dvdt, domegadt * np.cos(theta), v, omega**2 * np.sin(theta), F,
        domegadt, dvdt * np.cos(theta), np.sin(theta)
    ])
    
    full_sym_lib = [
        dvdt_sym, domegadt_sym * sympy.cos(theta_sym), v_sym,
        omega**2 * sympy.sin(theta_sym), u_sym,
        domegadt_sym, dvdt_sym * sympy.cos(theta_sym), sympy.sin(theta_sym)
    ]
    
    norms = np.linalg.norm(full_Theta, axis=0); norms[norms==0] = 1
    Theta_normalized = full_Theta / norms

    equations = []
    threshold = 0.1
    
    # Pass 1 (Finding the first equation)
    print("\n--- Pass 1: Searching for the first equation ---")
    candidate_models_pass1 = []
    for i in range(Theta_normalized.shape[1]):
        target_y = Theta_normalized[:, i]
        rhs_X = np.delete(Theta_normalized, i, axis=1)
        w_normalized = stlsq(rhs_X, -target_y, threshold=threshold)
        ksi_normalized = np.insert(w_normalized, i, 1.0)
        ksi = ksi_normalized / norms
        error = np.linalg.norm(full_Theta @ ksi)
        candidate_models_pass1.append({'ksi': ksi, 'error': error, 'lhs_idx': i})

    candidate_models_pass1.sort(key=lambda x: x['error'])
    best_ksi_pass1 = candidate_models_pass1[0]['ksi']
    
    max_coeff = np.max(np.abs(best_ksi_pass1))
    ksi1_readable = best_ksi_pass1 / max_coeff if max_coeff > 1e-9 else best_ksi_pass1
    
    eq1 = sum(float(c) * s for c, s in zip(ksi1_readable, full_sym_lib) if abs(c) > 1e-3)
    equations.append(eq1)
    print("Found First Implicit Equation:")
    sympy.pprint(sympy.Eq(sympy.N(eq1, 4), 0), use_unicode=False)

    # Pass 2 (Finding the second equation with a constrained library)
    print("\n--- Pass 2: Finding the second equation with a constrained library ---")
    
    constrained_sym_lib = [
        dvdt_sym, 
        domegadt_sym * sympy.cos(theta_sym),
        u_sym, 
        v_sym,
        omega_sym**2 * sympy.sin(theta_sym)
    ]
    constrained_Theta = np.column_stack([
        dvdt, domegadt * np.cos(theta), F, v, omega**2 * np.sin(theta)
    ])

    try:
        _, _, Vt = np.linalg.svd(constrained_Theta, full_matrices=False)
        ksi2 = Vt[-1, :]
    except np.linalg.LinAlgError:
        print("SVD failed for the constrained library.")
        return None

    max_coeff = np.max(np.abs(ksi2))
    ksi2_readable = ksi2 / max_coeff if max_coeff > 1e-9 else ksi2
    
    eq2 = sum(float(c) * s for c, s in zip(ksi2_readable, constrained_sym_lib) if abs(c) > 1e-3)
    equations.append(eq2)
    print("Found Second Implicit Equation (Constrained):")
    sympy.pprint(sympy.Eq(sympy.N(eq2, 4), 0), use_unicode=False)
        
    return equations

def extract_coefficients(equations, state_names):
    """
    Parses the discovered implicit equations to separate them into
    mass matrix components and forcing vector components.
    """
    dvdt_sym, domegadt_sym = sympy.symbols(['dv/dt', 'domega/dt'])
    coeffs_list = []
    
    # Iterate through the list of discovered symbolic equations
    for eq in equations:
        coeffs = {
            # Extract coefficients of acceleration terms to build the Mass Matrix M(y)
            # This will include coupled terms like cos(theta)
            'M11': eq.coeff(dvdt_sym),
            'M12': eq.coeff(domegadt_sym),
            # Extract all other terms for the Forcing Vector F(y)
            # We move them to the RHS, so we negate the expression
            'F': -eq.subs([(dvdt_sym, 0), (domegadt_sym, 0)])
        }
        coeffs_list.append(coeffs)
    return coeffs_list

def create_sindy_dynamics_implicit_solver(coeffs_list, state_names, controller_func):
    """
    Creates a callable ODE function that solves the discovered system numerically
    by building and solving M(y)*y_ddot = F(y) at each time step.
    """
    if not coeffs_list or len(coeffs_list) < 2: return None

    x_sym, v_sym, theta_sym, omega_sym = sympy.symbols(state_names)
    u_sym = sympy.symbols('u_1')
    
    # Pre-compile all coefficient expressions into fast numerical functions
    lambdified_funcs = []
    for coeffs in coeffs_list:
        l_funcs = {
            'M11': sympy.lambdify([x_sym, v_sym, theta_sym, omega_sym, u_sym], coeffs['M11'], 'numpy'),
            'M12': sympy.lambdify([x_sym, v_sym, theta_sym, omega_sym, u_sym], coeffs['M12'], 'numpy'),
            'F': sympy.lambdify([x_sym, v_sym, theta_sym, omega_sym, u_sym], coeffs['F'], 'numpy'),
        }
        lambdified_funcs.append(l_funcs)

    def sindy_dynamics(t, y):
        x, v, theta, omega = y
        u = controller_func(t, y)
        mass_matrix, b_vector = np.zeros((2, 2)), np.zeros(2)
        
        # Build the mass matrix and forcing vector using the discovered coefficients
        for i, funcs in enumerate(lambdified_funcs):
            mass_matrix[i, 0] = funcs['M11'](x, v, theta, omega, u)
            mass_matrix[i, 1] = funcs['M12'](x, v, theta, omega, u)
            b_vector[i] = funcs['F'](x, v, theta, omega, u)
        
        try:
            accelerations = np.linalg.solve(mass_matrix, b_vector)
        except np.linalg.LinAlgError:
            return np.array([v, 0, omega, 0])
        return np.array([v, accelerations[0], omega, accelerations[1]])

    return sindy_dynamics

def plot_comparison(t, X_original, sindy_sim_results, state_names, title="Original Dynamics vs. SINDy-Discovered Model"):
    """Generates plot comparing original and simulated trajectories"""
    print("\n--- Generating Comparison Plot ---")
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axs = axs.flatten()
    
    y_labels = [
        r'Cart Position $x$ (m)', r'Cart Velocity $\dot{x}$ (m/s)', 
        r'Pendulum Angle $\theta$ (rad)', r'Pendulum Angular Velocity $\dot{\theta}$ (rad/s)'
    ]

    for i, name in enumerate(state_names):
        true_signal = X_original[:, i]
        sindy_signal = sindy_sim_results.y.T[:, i]
        
        # Calculate the L2 norm of the error
        error_norm = np.linalg.norm(true_signal - sindy_signal)
        true_norm = np.linalg.norm(true_signal)
        
        # Avoid division by zero if the true signal is all zeros
        if true_norm < 1e-12:
            relative_error = 0.0 if error_norm < 1e-12 else np.inf
        else:
            relative_error = error_norm / true_norm
        error_percent_str = f"{relative_error * 100:.2f}%"
        sindy_label = f"SINDy Model (Rel. Error: {error_percent_str})"

        axs[i].plot(t, X_original[:, i], 'k-', linewidth=2.5, label='True System')

        marker_frequency = 10 
        axs[i].plot(
            sindy_sim_results.t[::marker_frequency], 
            sindy_sim_results.y[i, ::marker_frequency], 
            'r--o',
            markersize=2,
            label=sindy_label 
        )
        
        axs[i].set_ylabel(y_labels[i], fontsize=14)
        axs[i].grid(True)
        axs[i].legend(fontsize=12)
        axs[i].tick_params(labelsize=12)
        
    axs[2].set_xlabel('Time (s)', fontsize=14)
    axs[3].set_xlabel('Time (s)', fontsize=14)
    
    fig.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def simulate_sindy_model(sindy_dynamics_func, initial_condition, t_span, dt):
    """
    Simulates the discovered SINDy model from a new initial condition.

    Args:
        sindy_dynamics_func (callable): The dynamics function created by create_sindy_dynamics_implicit_solver.
        initial_condition (np.array): The new starting state [x, v, theta, omega].
        t_span (list): The time interval for the simulation, e.g., [0, 10].
        dt (float): The time step for the output.

    Returns:
        OdeResult: The solution object from solve_ivp.
    """
    if not sindy_dynamics_func:
        print("Error: The SINDy dynamics function is not valid. Cannot simulate.")
        return None

    t_eval = np.arange(t_span[0], t_span[1], dt)
    
    print(f"\n--- Simulating SINDy model with new initial condition: {initial_condition} ---")
    
    solution = solve_ivp(
        sindy_dynamics_func,
        t_span,
        initial_condition,
        t_eval=t_eval,
        method='RK45' 
    )
    
    return solution


    