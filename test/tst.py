import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from setuptools import setup, find_packages

setup(name='pytorch-esn',
      version='1.2.4',
      packages=find_packages(),
      install_requires=[
          'torch',
          'torchvision',
          'numpy'
      ],
      description="Echo State Network module for PyTorch.",
      author='Stefano Nardo',
      author_email='stefano_nardo@msn.com',
      license='MIT',
      url="https://github.com/stefanonardo/pytorch-esn"
      )