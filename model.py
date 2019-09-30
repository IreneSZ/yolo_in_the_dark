#!/usr/bin/env python
# coding: utf-8

# dummy model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# input: tensor of 1 * 360 * 640 * 3
# output: scaler

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Linear(360*640*3, 1)
    
    def forward(self, x):
        x = self.output(x.flatten())
        return x
    
    
