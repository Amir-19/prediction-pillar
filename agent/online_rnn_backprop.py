import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RecurrentNet(nn.Module):
    def __init__(self, dim_input, num_hidden, truncation, step_size = 0.1):
        super(RecurrentNet, self).__init__()
        self.num_hidden = num_hidden
        self.rnn_cell = nn.RNNCell(dim_input, self.num_hidden)
        self.linear = nn.Linear(self.num_hidden, 1)
        self.truncation = truncation
        self.k = 0
        self.h_t = torch.zeros((1,self.num_hidden), dtype=torch.double)
        self.h_t.requires_grad = True
        self.outputs = []
        self.targets = []
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=step_size)

    def reset_hidden_state(self):
        self.h_t = torch.zeros((1, self.num_hidden), dtype=torch.double)
        #TODO: is this necessary
        self.h_t.requires_grad = True

    def forward(self, x):
        self.k += 1
        # one step forward
        self.h_t = self.rnn_cell(x, self.h_t)
        output = self.linear(self.h_t)
        self.outputs += [output] # save the output to a list of outputs for later bptt

        return output

    def save_targets(self, target):
        self.targets += [target]
        # check whether it is time to do bptt or not
        loss = None
        if self.k == self.truncation:
            self.k = 0
            loss = self.tbptt()
        return loss

    def tbptt(self):
        # get outputs and targets into shape for the bptt
        self.outputs = torch.stack(self.outputs, 1).squeeze(2)
        self.targets = torch.tensor(self.targets,dtype=torch.double).unsqueeze(0)
        # running bptt and updating the weights
        self.optimizer.zero_grad()
        loss = self.criterion(self.outputs, self.targets)
        loss.backward()
        self.optimizer.step()
        self.outputs = []
        self.targets = []
        # detaching the hidden state stop the gradient flow for the next time we run bptt
        self.h_t.detach_()
        self.h_t.requires_grad = True
        return loss