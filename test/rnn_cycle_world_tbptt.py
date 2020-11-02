import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
from torchviz import make_dot


class RnnTbptt(nn.Module):
    def __init__(self, num_input=2, num_hidden=100, k1=1, k2=6):
        # every k1 steps backprop taking k2 back steps
        super(RnnTbptt, self).__init__()
        self.num_hidden = num_hidden
        self.num_input = num_input
        self.rnn = nn.RNNCell(self.num_input, self.num_hidden)
        self.linear = nn.Linear(self.num_hidden, 1)

        self.k1 = k1
        self.k2 = k2
        self.k1_counter = 0
        self.retain_graph = k1 < k2

        self.outputs = []
        self.targets = []
        self.states = [(None, torch.zeros(1, self.num_hidden, dtype=torch.double))]

        self.optimizer = optim.SGD(self.parameters(), lr=0.1)
        self.criterion = nn.MSELoss()

    def forward(self, input, state):
        new_state = self.rnn(input, state)
        output = self.linear(new_state)
        return output, new_state

    def init_hidden_state(self):
        self.h_t = torch.zeros(1, self.num_hidden, dtype=torch.double)

    def train_step(self, input, target):
        self.k1_counter += 1
        state = self.states[-1][1].detach()
        state.requires_grad = True
        #print(state)
        output, new_state = self.__call__(input,state)
        self.states.append((state, new_state))
        self.outputs.append(output)
        self.targets.append(target)
        while len(self.outputs) > self.k1  :
            # Delete stuff that is too old
            del self.outputs[0]
            del self.targets[0]

        while len(self.states) > self.k2:
            # Delete stuff that is too old
            del self.states[0]
        if self.k1_counter == self.k1:
            print("haha")
            self.k1_counter = 0
            self.optimizer.zero_grad()
            loss = self.criterion(torch.squeeze(torch.squeeze(torch.stack(self.outputs),1),1), torch.tensor(self.targets))
            loss.backward(retain_graph=False)
            # for j in range(self.k2-1):
            #     if self.states[-j-2][0] is None:
            #         break
            #     curr_grad = self.states[-j-1][0].grad
            #     #print(curr_grad)
            #     self.states[-j-2][1].backward(curr_grad, retain_graph=self.retain_graph)
            self.optimizer.step()
        return output


if __name__ == '__main__':
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    torch.autograd.set_detect_anomaly(True)
    # build the model
    seq = RnnTbptt()
    seq.double()

    #begin to train
    for i in range(1000):
        print('STEP: ', i)
        out = seq.train_step(torch.tensor([[0.0, 1.0]]).double(), torch.tensor([[0.0]]).double())
        out = seq.train_step(torch.tensor([[0.0, 1.0]]).double(), torch.tensor([[0.0]]).double())
        # graph = make_dot(torch.squeeze(torch.squeeze(torch.stack(seq.outputs),1),1))
        # graph.format = 'png'
        # graph.render('rnn_computation_graph_tbptt')
        out = seq.train_step(torch.tensor([[0.0, 1.0]]).double(), torch.tensor([[0.0]]).double())
        out = seq.train_step(torch.tensor([[0.0, 1.0]]).double(), torch.tensor([[0.0]]).double())
        out = seq.train_step(torch.tensor([[0.0, 1.0]]).double(), torch.tensor([[1.0]]).double())
        out = seq.train_step(torch.tensor([[1.0, 0.0]]).double(), torch.tensor([[0.0]]).double())
        print(torch.squeeze(torch.squeeze(torch.stack(seq.outputs),1)))
