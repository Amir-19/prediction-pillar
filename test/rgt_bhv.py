import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import collections
from environment.recurrent_geoff_task import RecurrentGeoffTask


def rgt_output_with_no_data():
    # set random seed to 0
    #np.random.seed(0)
    #torch.manual_seed(0)

    #exp
    env = RecurrentGeoffTask(5,5,0.5)
    for i in range(100):
        print(env.get_sample())
    print("now we feed no input")
    for i in range(100):
        print(env.get_sample_zero_input())

def rgt_activations():
    env = RecurrentGeoffTask(5,5,0.5)
    x = np.zeros(5)
    for i in range(100):
        env.get_sample()
        print(env.s)
        x += env.s[0]
    print(x/100.0)
if __name__ == '__main__':
    rgt_activations()