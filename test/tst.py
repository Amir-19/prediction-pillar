import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from setuptools import setup, find_packages


def cmptAvg(alist):
    avg = []
    for i in range(len(alist)):
        average = sum(alist) / len(alist)
        avg.append(average)
        return avg


# main program
intNumbers = [8, 7, 5, 6, 9, 3, 2, 4]
realNumbers = [1.1, 5.4, 6.8, 9.3, 4.2, 3.8, 12.5]
print("The average =", cmptAvg(intNumbers))
print("The average =", cmptAvg(realNumbers))
