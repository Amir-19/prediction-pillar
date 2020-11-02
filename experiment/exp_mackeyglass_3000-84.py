import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


from environment.synthetic_online import MackeyGlass
from agent.online_rnn_backprop import RecurrentNet

def exp_mackyglass_rnn():
    # setting the random seeds
    truncation = 17
    net = RecurrentNet(1,30,truncation,step_size=0.01)
    net.double()

    env = MackeyGlass(tau=17)
    xs = [1.2]
    yhats = []
    losses = []
    for i in range(3000):
        x = env.get_sample()
        x_t = torch.tensor([[xs[-1]]]).double()
        y_hat = net.forward(x_t)
        loss = net.save_targets(torch.tensor(x).double())
        xs.append(x)
        yhats.append(y_hat)
        if loss is not None:
            losses.append(loss)

    x_p = torch.tensor([[xs[-1]]]).double() # the one for rnn
    y_hat = 0

    for i in range(84):
        x = env.get_sample()
        y_hat = net.forward(x_p)
        xs.append(x)
        x_p = y_hat
    # predicted and truth after 84 steps
    # print(y_hat)
    # print(xs[-1])
    y_hat_val = y_hat.data[0][0].numpy()
    return (y_hat_val-xs[-1])*(y_hat_val-xs[-1])

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    error_sum = 0
    num_exp = 15
    for i in range(num_exp):
        print("exp run: ",i+1)
        final_error = exp_mackyglass_rnn()
        error_sum += final_error
    exp_error = (np.power(error_sum/num_exp,0.5))
    print(exp_error)