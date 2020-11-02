import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import collections

from environment.recurrent_geoff_task import RecurrentGeoffTask
from agent.online_rnn_backprop import RecurrentNet

if __name__ == '__main__':
    # set random seed to 0
    # np.random.seed(0)
    # torch.manual_seed(0)
    # data collection
    losses = []
    error_sum = 0
    error_interval = 1000
    errors = []
    # problem and solution details
    input_size = 5
    target_net_hidden_size = 5
    rnn_hidden_size = 30

    # build the model
    net = RecurrentNet(input_size,rnn_hidden_size,5,step_size=0.01)
    net.double()

    env = RecurrentGeoffTask(input_size,target_net_hidden_size,0.5)
    #begin to train
    for j in range(5):
        for i in range(1000000):
            #print('STEP: ', i)
            x,y =env.get_sample()
            y_hat = net.forward(x)
            loss = net.save_targets(y)
            if loss is not None:
                losses.append(loss)
            y_hat_val = y_hat.data[0][0].numpy()
            error_sum += ((y-y_hat_val)*(y-y_hat_val))
            if (i+1)%error_interval == 0:
                errors.append(error_sum/error_interval)
                error_sum = 0
        env = RecurrentGeoffTask(input_size, target_net_hidden_size, 0.5)
    plt.plot(errors)
    plt.show()