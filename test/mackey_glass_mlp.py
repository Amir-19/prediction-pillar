import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from environment.synthetic_online import MackeyGlass
from agent.online_rnn_backprop import RecurrentNet

class SingleHiddenLayerNN(nn.Module):
    def __init__(self, input_size, feature_size,step_size=0.1):
        super(SingleHiddenLayerNN, self).__init__()

        self.hidden_layer = nn.Linear(input_size, feature_size,bias=True)
        self.output_layer = nn.Linear(feature_size, 1,bias=True)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(filter(lambda l: l.requires_grad, self.parameters()), lr=step_size)

    def forward(self, x):
        x = torch.sigmoid(self.hidden_layer(x))
        y = self.output_layer(x)
        return y

    def train_step(self,x,y):
        self.optimizer.zero_grad()  # zero the gradient buffers
        prediction = self.__call__(x)
        loss = self.criterion(prediction, y)
        loss.backward()
        self.optimizer.step()  # Does the update
        return prediction.data - y

def exp_mackyglass_rnn():
    # setting the random seeds
    np.random.seed(0)
    torch.manual_seed(0)

    net = SingleHiddenLayerNN(1,100,step_size=0.1)
    net.double()

    env = MackeyGlass(tau=17)
    xs = [1.2]
    yhats = []
    losses = []
    for i in range(1000):
        x = env.get_sample()
        x_t = torch.tensor([xs[-1]]).double()
        y_hat = net.forward(x_t)
        error = net.train_step(x_t,torch.tensor(x).double())
        loss = error*error
        xs.append(x)
        yhats.append(y_hat)
        if loss is not None and i > 100:
            losses.append(loss)
        if i>990:
            print(error)
    print(loss)
    plt.plot(losses)
    plt.show()
if __name__ == '__main__':
    exp_mackyglass_rnn()