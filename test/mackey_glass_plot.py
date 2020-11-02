import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from environment.synthetic_online import MackeyGlass
from agent.online_rnn_backprop import RecurrentNet
import collections

def mackey_glass_plot():
    # setting the random seeds

    env = MackeyGlass(tau=17)
    xs = [1.2]
    yhats = []
    losses = []
    for i in range(1400):
        x = env.get_sample()
        xs.append(x)
    plt.plot(xs)
    plt.show()
if __name__ == '__main__':
    mackey_glass_plot()