import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RecurrentNet(nn.Module):
    def __init__(self, dim_input, num_hidden, truncation, step_size = 0.1, keep_hidden = True):
        super(RecurrentNet, self).__init__()
        self.dim_hidden = num_hidden
        self.dim_input = dim_input
        self.rnn_cell = nn.RNNCell(dim_input, self.dim_hidden)
        self.linear = nn.Linear(self.dim_hidden, 1)
        self.truncation = truncation
        self.keep_hidden = keep_hidden
        self.k = 0
        self.h_t = torch.zeros((1,self.dim_hidden), dtype=torch.double)
        self.h_t.requires_grad = True
        self.outputs = []
        self.targets = []
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=step_size)
        #self.optimizer = optim.Adam(self.parameters(), lr=step_size)
        self.is_trace = np.zeros(self.dim_hidden)
    def reset_hidden_state(self):
        self.h_t = torch.zeros((1, self.dim_hidden), dtype=torch.double)
        #TODO: is this necessary
        self.h_t.requires_grad = True
    def make_hidden_state_trace(self,hidden_index,input_to_trace=-1,hidden_to_trace=-1):
        self.is_trace[hidden_index] = 1.0
        with torch.no_grad():
            new_input_tensor = torch.zeros(self.dim_input)
            new_input_tensor[input_to_trace] = 0.1
            self.rnn_cell.weight_ih[hidden_index] = new_input_tensor
            new_hidden_tensor = torch.zeros(self.dim_hidden)
            new_hidden_tensor[hidden_index] = 0.9
            self.rnn_cell.weight_hh[hidden_index] = new_hidden_tensor
            self.rnn_cell.bias_hh[hidden_index] = torch.tensor([0])
            self.rnn_cell.bias_ih[hidden_index] = torch.tensor([0])
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
        ####################################### added part for tracing features not to be changed!
        loss.backward()
        with torch.no_grad():
            for i in range(self.dim_hidden):
                if self.is_trace[i] == 1.0:
                    self.rnn_cell.weight_ih.grad[i] = torch.zeros(self.dim_input)
                    self.rnn_cell.weight_hh.grad[i] = torch.zeros(self.dim_hidden)
                    self.rnn_cell.bias_ih.grad[i] = torch.tensor([0])
                    self.rnn_cell.bias_hh.grad[i] = torch.tensor([0])
        #######################################
        self.optimizer.step()
        self.outputs = []
        self.targets = []
        # detaching the hidden state stop the gradient flow for the next time we run bptt
        if self.keep_hidden:
            self.h_t.detach_()
            self.h_t.requires_grad = True
        else:
            self.reset_hidden_state()
        return loss