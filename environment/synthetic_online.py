import collections
import numpy as np

class MackeyGlass():

    def __init__(self,tau=17):
        self.delta_t = 10
        self.tau = tau
        self.history_len = tau * self.delta_t
        # Initial conditions for the history of the system
        self.timeseries = 1.2
        # self.history = collections.deque(1.2 * np.ones(self.history_len) + 0.2 * \
        #                           (np.random.rand(self.history_len) - 0.5))
        self.history = collections.deque(np.zeros(self.history_len))
        # TODO: maybe fill the history with 0

    def get_sample(self):
        for _ in range(self.delta_t):
            xtau = self.history.popleft()
            self.history.append(self.timeseries)
            self.timeseries = self.history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) - \
                                        0.1 * self.history[-1]) / self.delta_t
        # TODO: check why tanh
        #return np.tanh(self.timeseries - 1)
        return self.timeseries

class MSO():

    def __init__(self):
        self.t = 0
    def get_sample(self):
        x = np.sin(0.2 * self.t) + np.sin(0.311 * self.t) \
            + np.sin(0.42 * self.t) + np.sin(0.51 * self.t)
        self.t += 1
        return x
