import numpy as np
import torch
import torch.nn as nn
import pandas as pd

# multivariate logistic regression
class MultivariateLogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        '''
        input_dim: number of input features
        output_dim: number of conditions to predict
        '''
        super(MultivariateLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        logits = self.linear(x)
        return torch.sigmoid(logits)