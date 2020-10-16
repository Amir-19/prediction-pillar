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

    net = RecurrentNet(1,5,1,step_size=0.1)
    net.double()

    env = MackeyGlass(tau=35)
    xs = [1.2]
    yhats = []
    losses = []
    for i in range(100000):
        x = env.get_sample()
        x_t = torch.tensor([[xs[-1]]]).double()
        y_hat = net.forward(x_t)
        loss = net.save_targets(torch.tensor(x).double())
        xs.append(x)
        yhats.append(y_hat)
        if loss is not None and i > 99000:
            losses.append(loss)
    print(loss)
    plt.plot(losses)
    plt.show()
if __name__ == '__main__':
    exp_mackyglass_rnn()