import collections
import numpy as np

class MackeyGlass():

    def __init__(self,tau=17):
        self.delta_t = 10
        self.tau = tau
        self.history_len = tau * self.delta_t
        # Initial conditions for the history of the system
        self.timeseries = 1.2
        self.history = collections.deque(1.2 * np.ones(self.history_len) + 0.2 * \
                                    (np.random.rand(self.history_len) - 0.5))

    def get_sample(self):
        for _ in range(self.delta_t):
            xtau = self.history.popleft()
            self.history.append(self.timeseries)
            self.timeseries = self.history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) - \
                                        0.1 * self.history[-1]) / self.delta_t

        return np.tanh(self.timeseries - 1)