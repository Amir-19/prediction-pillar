import torch.nn as nn


class LTU(nn.Module):

    def __init__(self, threshold):
        super(LTU, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        return (x > self.threshold).float()
