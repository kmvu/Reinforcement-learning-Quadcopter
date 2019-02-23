import numpy as np
import copy

""" Ornstein-Uhlenbeck process, to encourage exploratory behaviour"""
class OUNoise:

    def __init__(self, size, mu, theta, sigma):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma

        self.reset()

    def reset(self):
        """Reset internal state (= noise) to mean (mu)"""
        self.state = copy.copy(self.mu)

    def noise_sample(self):
        """Update and return internal state as noise sample"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx

        return self.state
