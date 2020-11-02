import numpy as np
import torch

class RecurrentGeoffTask:

    def __init__(self, m, n, beta=0.5, bias=True, noise=False, mu_epsilon=0.0, sigma_epsilon=1.0,
                 geoff_mode='new'):
        """

        :param m: input size (in bits)
        :param n: number of hidden LTU units
        :param beta: the proportion of the bits that have to match the prototype
        :param bias: add bias bit 1 to representation or not
        :param noise: noise added to the final output or not
        :param mu_epsilon: mean of the noise distribution
        :param sigma_epsilon: sigma of the noise distribution
        :param geoff_mode: 'old' -> w_i ~ normal(0,1), 'new' -> w_i in {-1,0,+1}
        """
        self.m = m
        self.n = n
        self.beta = beta
        self.bias = bias
        self.noise = noise
        self.mu_epsilon = mu_epsilon
        self.sigma_epsilon = sigma_epsilon

        self.v = np.random.choice([-1, 1], (m+n, n), p=[0.5, 0.5])
        self.smin = np.count_nonzero(self.v == -1, axis=0) * -1
        self.theta = self.smin + (self.beta * (self.m+self.n))
        if geoff_mode == 'new':
            self.w = np.random.choice([-1, 0, 1], (n, 1))
        elif geoff_mode == 'old':
            self.w = np.random.normal(0.0, 1.0, (n, 1))
        self.s = np.zeros((1,n))

    def calculate_output(self, x):
        """

        :param x: input as a vector of bits
        :return: the output of ERR
        """
        assert x.shape == (self.m+self.n,) or x.shape == (self.m+self.n, 1)
        self.s = np.matmul(x.T, self.v)
        self.s = np.greater(self.s, self.theta).astype(int)
        if self.bias:
            self.s[:, -1] = 1

        y = np.dot(self.s, self.w)[0]
        return y

    def get_sample(self):
        """

            :return: a sample from the geoff task x as the input and y as the output
        """
        o = np.random.randint(2, size=(self.m, 1))#.astype("float")
        x = np.concatenate((self.s.T,o))
        if self.noise:
            y = self.calculate_output(x) + np.random.normal(self.mu_epsilon, self.sigma_epsilon, 1)[0]
        else:
            y = self.calculate_output(x)

        o = torch.from_numpy(o.T.astype('d'))
        y = torch.from_numpy(y.astype('d'))
        return o, y

    def get_sample_zero_input(self):
        """

            :return: a sample from the geoff task x as the input and y as the output
        """
        o = np.random.randint(2, size=(self.m, 1))#.astype("float")
        x = np.concatenate((self.s.T,o))
        if self.noise:
            y = self.calculate_output(x) + np.random.normal(self.mu_epsilon, self.sigma_epsilon, 1)[0]
        else:
            y = self.calculate_output(x)

        o = torch.from_numpy(o.T.astype('d'))
        y = torch.from_numpy(y.astype('d'))
        return o, y
    def state_restart(self):
        self.s = np.zeros((1,self.n))