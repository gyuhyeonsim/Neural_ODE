import numpy as np
import pickle
import os
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from torchdyn.models import DepthCat, GalLinear
from utils.model_utils.model_utils import *
from predefined_galerkin import *

"""
encoder : bidirectional RNN with last hidden state
decoder : predefined galerkin (controlled galerkin)
Note that it is not VAE, it is deterministic AE
"""

### not used at now
class Bidirectional_RNN(nn.Module):
    def __init__(self, input_dim = 1, hidden_dim = 6, output_dim = 3, num_layers = 1, bidirectional = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_di = 2 if bidirectional else 1
        self.RNN = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

        # decoder
        self.output_fc = nn.Linear(self.hidden_dim * self.num_di, output_dim)

    def forward(self, x):
        self.x = x
        self.batch_size = x.size(0)
        hidden = self.init_hidden() # initialize the hidden layers

        output, _ = self.RNN(x, hidden)
        forward_output = output[:, -1, :self.hidden_dim]
        backward_output = output[:, 0, self.hidden_dim:]
        output = torch.cat((forward_output, backward_output), dim=1)
        output = self.output_fc(output)
        return output

    def init_hidden(self):
        return torch.randn(self.num_layers * self.num_di, self.batch_size, self.hidden_dim).cuda()


class coeffs_decoder(nn.Module):
    def __init__(self, latent_dimension, coeffs_size):
        super().__init__()
        self.latent_dimension = latent_dimension
        self.fc1 = nn.Linear(latent_dimension, coeffs_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(coeffs_size, coeffs_size)

    def forward(self, x):
        # input latent vector
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out

class Galerkin_s(nn.Module):
    def __init__(self, in_features, out_features, latent_dimension, expfunc, n_harmonics, n_eig):
        super().__init__()
        self.depth_cat = DepthCat(1)
        self.gallinear = GalLinear(in_features = (in_features + latent_dimension), out_features=out_features,
                                   n_harmonics=n_harmonics, n_eig=n_eig, expfunc=expfunc)

        self.swish = get_nonlinearity('swish')
        self.linear1 = nn.Linear(out_features, out_features)
        self.linear2 = nn.Linear(out_features, in_features)
        self.z = None

    def forward(self, x):
        self.z = self.z.cuda()
        x = torch.cat((x, self.z), 1)
        x = self.depth_cat(x)

        out = self.swish(self.gallinear(x))
        out = self.swish(self.linear1(out))
        out = self.linear2(out)
        return out
