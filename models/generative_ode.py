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
"""
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
"""
def FourierExpansion(n_range, s):
    """Fourier eigenbasis expansion
    """
    s_n_range = s*n_range
    basis = [torch.cos(s_n_range), torch.sin(s_n_range)]
    return basis

def PolyExpansion(n_range, s):
    """Polynomial expansion
    """
    basis = [s**n_range]
    return basis

class CoeffDecoder(nn.Module):
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

class WeightAdaptiveGallinear(nn.Module):
    def __init__(self, in_features = 1, out_features = 16,
                 latent_dimension = 3, expfunc = FourierExpansion, n_harmonics = 5, n_eig = 2, dilation = True, shift = True):
        super(WeightAdaptiveGallinear).__init__()

        self.in_features, self.out_features, self.latent_dimension = in_features, out_features, latent_dimension
        self.dilation = torch.ones(1) if not dilation else nn.Parameter(data = torch.ones(1), requires_grad=True)
        self.shift = torch.zeros(1) if not shift else nn.Parameter(data=torch.zeros(1), requires_grad=True)
        self.expfunc = expfunc
        self.n_harmonics, self.n_eig = n_harmonics, n_eig

        coeffs_size = (in_features + 1) * out_features * n_harmonics * n_eig
        self.coeffs_generator = CoeffDecoder(latent_dimension, coeffs_size)
        self.depth_cat = DepthCat(1)


    def assign_weights(self, s, coeffs):
        n_range = torch.linspace(0, self.n_harmonics, self.n_harmonics).to(self.input.device)
        basis = self.expfunc(n_range, s * self.dilation.to(self.input.device) + self.shift.to(self.input.device))
        B = []
        for i in range(self.n_eig):
            Bin = torch.eye(self.n_harmonics).to(self.input.device)
            Bin[range(self.n_harmonics), range(self.n_harmonics)] = basis[i]
            B.append(Bin)
        B = torch.cat(B, 1).permute(1, 0).to(self.input.device)
        X = torch.matmul(coeffs, B)
        return X.sum(2)

    def forward(self, x):
        assert x.size(1) == (self.in_features + self.latent_dimension + 1) # x should be ordered in input_data, s, latent_variables
        self.batch_size = x.size(0)
        s = x[-1, self.in_features]
        self.input = torch.unsqueeze(x[:, :self.in_features], dim=-1)
        latent_variables = x[:, -self.latent_dimension:]

        coeffs = self.coeffs_generator(latent_variables).reshape(self.batch_size, (self.in_features + 1) * self.out_features, self.n_eig * self.n_harmonics)

        w = self.assign_weights(s, coeffs)
        self.weight = w[:, :(self.in_features * self.out_features)].reshape(self.batch_size, self.in_features, self.out_features)
        self.bias = w[:, (self.in_features * self.out_features):((self.in_features + 1) * self.out_features)].reshape(self.batch_size, self.out_features)

        self.weighted = torch.squeeze(torch.bmm(self.input, self.weight), dim=1)
        return torch.add(self.weighted, self.bias)

class AugmentedGalerkin(nn.Module):
    def __init__(self, in_features, out_features, latent_dimension, expfunc, n_harmonics, n_eig):
        super(AugmentedGalerkin).__init__()
        self.depth_cat = DepthCat(1)
        self.gallinear = WeightAdaptiveGallinear(in_features=in_features, out_features=out_features, latent_dimension=latent_dimension,
                                                  expfunc=expfunc, n_harmonics=n_harmonics, n_eig=n_eig)
        self.z = None

    def forward(self, x):
        # return dynamics
        x = self.depth_cat(x)
        self.z = self.z.cuda()
        x = torch.cat((x, self.z), 1)
        out = self.gallinear(x)
        return out

