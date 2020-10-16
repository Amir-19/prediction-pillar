import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class RNN_Network(nn.Module):
    def __init__(self, num_hidden = 100, truncation = 6, step_size = 0.1):
        super(RNN_Network, self).__init__()
        self.num_hidden = num_hidden
        self.truncation = truncation

        # network def
        self.rnn = nn.RNNCell(2, self.num_hidden)
        self.linear = nn.Linear(self.num_hidden, 1)

        # optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=step_size)
        #self.optimizer = optim.Adam(self.parameters(), lr=step_size)
        self.criterion = nn.MSELoss()

        # data
        self.outputs = []
        self.targets = []
        self.fwd_counter = 0

        # hidden state
        self.h_t = torch.zeros(1, self.num_hidden, dtype=torch.double)
        self.h_t.requires_grad = True
    def forward(self, input):
        self.h_t = self.rnn(input, self.h_t)
        output = self.linear(self.h_t)

        return output

    def step(self,input,target):
        self.fwd_counter += 1
        output = self.__call__(input)
        self.outputs += [output]
        self.targets += [target]
        if (self.fwd_counter == self.truncation):
            self.fwd_counter = 0
            outputs = torch.stack(self.outputs, 1).squeeze(2)
            targets = torch.tensor(self.targets).unsqueeze(0).double()
            print(outputs)
            print(targets)
            self.optimizer.zero_grad()
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.outputs = []
            self.targets = []

            self.h_t.detach_()
            #self.h_t = torch.zeros(1, self.num_hidden, dtype=torch.double)
            self.h_t.requires_grad = True
            return loss
if __name__ == '__main__':
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    x = torch.tensor([[0.0, 1.0],[0.0, 1.0],[0.0, 1.0],[0.0, 1.0],[0.0, 1.0],[1.0, 0.0]]).double()
    x = torch.unsqueeze(x,1)
    y = [0.0,0.0,0.0,0.0,1.0,0.0]
    # build the model
    net = RNN_Network()
    net.double()

    #begin to train
    for i in range(1000):
        print('STEP: ', i)
        for j in range(x.shape[0]):
            input = x[j]
            target = y[j]
            out = net.step(input,target)