import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from environment.synthetic_online import MackeyGlass
from agent.online_rnn_backprop import RecurrentNet

def exp_mackyglass_rnn():
    # setting the random seeds
    np.random.seed(0)
    torch.manual_seed(0)

    net = RecurrentNet(1,30,1,step_size=0.01)
    net.double()

    env = MackeyGlass(tau=17)
    xs = [1.2]
    yhats = []
    losses = []
    errors = []
    for i in range(100000):
        x = env.get_sample()
        x_t = torch.tensor([[xs[-1]]]).double()
        y_hat = net.forward(x_t)
        loss = net.save_targets(torch.tensor(x).double())
        y_hat_val = y_hat.data[0][0].numpy()
        error = (y_hat_val - x) * (y_hat_val - x)
        xs.append(x)
        yhats.append(y_hat)
        if i > 95000:
            errors.append(error)
    plt.plot(errors)
    plt.show()
if __name__ == '__main__':
    exp_mackyglass_rnn()