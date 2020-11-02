import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import collections

from environment.synthetic_online import MackeyGlass
from agent.online_gru_backprop import GRUNet
from agent.online_rnn_backprop import RecurrentNet
def exp_mackyglass_rnn():
    # setting the random seeds
    np.random.seed(0)
    torch.manual_seed(0)

    error_sum = 0
    error_interval = 1000

    #net = GRUNet(1,128,8,step_size=0.01)
    net = RecurrentNet(1,128,5,step_size=0.01)
    net.double()

    env = MackeyGlass(tau=17)
    xs = collections.deque()
    xs.append(1.2)
    for i in range(12-1):
        xs.append(env.get_sample())
    yhats = []
    losses = []
    errors = []
    for i in range(1000000):
        x = env.get_sample()
        x_t = torch.tensor([[xs.popleft()]]).double()
        y_hat = net.forward(x_t)
        loss = net.save_targets(torch.tensor(x).double())
        y_hat_val = y_hat.data[0][0].numpy()
        error_sum += (y_hat_val - x) * (y_hat_val - x)
        if (i+1)%error_interval == 0:
            error_sum = np.power(error_sum,0.5)
            errors.append(error_sum/error_interval)
            error_sum = 0
        xs.append(x)
        yhats.append(y_hat)
        # if i > 95000:
        #     errors.append(error)
    plt.plot(errors)
    plt.show()
if __name__ == '__main__':
    exp_mackyglass_rnn()