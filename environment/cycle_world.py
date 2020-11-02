import torch


class CycleWorld:
    def __init__(self, num_states):
        self.cur_state = 0
        self.num_states = num_states

    def step(self):
        self.cur_state = (self.cur_state + 1) % self.num_states
        if self.cur_state == self.num_states - 1:
            return torch.tensor([[1.0, 0.0]]).double()  # when we observe 0
        else:
            return torch.tensor([[0.0, 1.0]]).double()  # when we observe 1

    def get_obs(self):
        if self.cur_state == self.num_states - 1:
            return torch.tensor([[1.0, 0.0]]).double()  # when we observe 0
        else:
            return torch.tensor([[0.0, 1.0]]).double()  # when we observe 1

    def get_binary_obs(self):
        if self.cur_state == self.num_states - 1:
            return torch.tensor([1.0]).double()  # when we observe 0
        else:
            return torch.tensor([0.0]).double()  # when we obs