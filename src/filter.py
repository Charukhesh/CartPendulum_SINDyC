import numpy as np
import src.config as cfg

class RecursiveLeastSquares:
    def __init__(self, n_features):
        self.theta = np.zeros(n_features) # The parameters we want to find
        self.P = np.eye(n_features) * 1000.0 # High initial uncertainty
        self.lam = 0.999 # Forgetting factor

    def step(self, phi, y):
        phi = phi.reshape(-1, 1)
        # Standard RLS update
        P_phi = self.P @ phi
        gain = P_phi / (self.lam + phi.T @ P_phi)
        error = y - phi.T @ self.theta
        self.theta += gain.flatten() * error
        self.P = (self.P - gain @ phi.T @ self.P) / self.lam
        return self.theta