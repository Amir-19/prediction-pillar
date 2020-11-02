import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import collections
from tqdm import tqdm
import seaborn as sns
import seaborn as sns
import pandas as pd

from environment.cycle_world import CycleWorld
from agent.online_rnn_trace_backprop import RecurrentNet
def cycle_world_rnn(time_steps):
    # set random seed to 0
    # np.random.seed(0)
    # torch.manual_seed(0)
    # data collection
    losses = []
    error_sum = 0
    error_interval = 1000
    errors = []
    # problem and solution details
    input_size = 2
    cycle_world_num_states = 6
    rnn_hidden_size = 5
    truncation = 1
    # build the model
    net = RecurrentNet(input_size, rnn_hidden_size, truncation, step_size=0.01, keep_hidden=True)
    net.double()

    net.make_hidden_state_trace(3,0,-1)
    net.make_hidden_state_trace(4,1,-1)
    env = CycleWorld(cycle_world_num_states)
    # begin to train
    x = env.get_obs()
    print("pre training hidden state weights: ")
    print(net.rnn_cell.weight_hh)
    for i in range(time_steps):
        x_p = env.step()
        y = env.get_binary_obs()
        y_hat = net.forward(x)
        loss = net.save_targets(y)
        if loss is not None:
            losses.append(loss)
        y_hat_val = y_hat.data[0][0].numpy()
        y_val = y.data[0].numpy()
        error_sum += ((y_val - y_hat_val) * (y_val - y_hat_val))
        if (i + 1) % error_interval == 0:
            errors.append(error_sum / error_interval)
            error_sum = 0
        x = x_p
    print("----")
    print("post training hidden state weights: ")
    print(net.rnn_cell.weight_hh)
    # print(net.rnn_cell.weight_ih)
    # print(net.rnn_cell.bias_hh)
    # print(net.rnn_cell.bias_ih)
    print("final_error:",errors[-1])
    return errors
if __name__ == '__main__':
    sns.set_style("darkgrid")
    errors = []
    for i in tqdm(range(10)):
        errors.append(np.array(cycle_world_rnn(time_steps=1000000)))
    df = pd.DataFrame(errors).melt()
    fig = sns.lineplot(x="variable", y="value", data=df)
    fig.set(xlabel='time step * 10^3', ylabel='NMSE')
    plt.show()