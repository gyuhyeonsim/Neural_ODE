import numpy as np
import pickle
import os
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from torchdyn.models import DepthCat, GalLinear, NeuralDE
from utils.model_utils.model_utils import *

"""
encoder : bidirectional RNN with last hidden state
decoder : predefined galerkin (controlled galerkin)
Note that it is not VAE, it is deterministic AE
"""

def fourier_expansion(n_range, s):
    """Fourier eigenbasis expansion
    """
    s_n_range = s*n_range
    basis = [torch.cos(s_n_range), torch.sin(s_n_range)]
    return basis

def batch_fourier_expansion(n_range, s):
    # s is shape of (batch_size x n_eig * n_harmonics)
    cos_s = s[:, :n_range.size(0)] * n_range
    cos_s = torch.diag_embed(torch.cos(cos_s))
    sin_s = s[:, n_range.size(0):] * n_range
    sin_s = torch.diag_embed(torch.sin(sin_s))
    return torch.cat((cos_s, sin_s), dim=1)


def poly_expansion(n_range, s):
    """Polynomial expansion
    """
    basis = [s**n_range]
    return basis

class CoeffDecoder(nn.Module):
    def __init__(self, latent_dimension, hidden_dim,  coeffs_size):
        super().__init__()
        self.latent_dimension = latent_dimension
        self.fc1 = nn.Linear(latent_dimension, hidden_dim)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, coeffs_size)
        self.act2 = nn.Tanh()
        self.fc3 = nn.Linear(coeffs_size, coeffs_size)
        print('decoder output size: {}'.format(coeffs_size))

    def forward(self, x):
        # input latent vector
        out = self.act1(self.fc1(x))
        out = self.act2(self.fc2(out))
        return self.fc3(out)


# class WeightAdaptiveGallinear(nn.Module):
#     def __init__(self, hidden_dim, in_features = 1, out_features = 16,
#                  latent_dimension = 3, expfunc = fourier_expansion, n_harmonics = 5, n_eig = 2, dilation = False, shift = False):
#         super().__init__()
#
#         self.in_features, self.out_features, self.latent_dimension = in_features, out_features, latent_dimension
#         self.dilation = torch.ones(1) if not dilation else nn.Parameter(data = torch.ones(1), requires_grad=True)
#         self.shift = torch.zeros(1) if not shift else nn.Parameter(data=torch.zeros(1), requires_grad=True)
#         self.expfunc = expfunc
#         self.n_harmonics, self.n_eig = n_harmonics, n_eig
#
#         coeffs_size = (in_features + 1) * out_features * n_harmonics * n_eig
#
#         # latent_dimension 3 means amp, sign, phase
#         self.coeffs_generator = CoeffDecoder(latent_dimension, hidden_dim, coeffs_size)
#
#     def assign_weights(self, s, coeffs):
#         n_range = torch.linspace(0, self.n_harmonics, steps=self.n_harmonics).to(self.input.device)
#         basis = self.expfunc(n_range, s * self.dilation.to(self.input.device) + self.shift.to(self.input.device))
#         B = []
#         for i in range(self.n_eig):
#             Bin = torch.eye(self.n_harmonics).to(self.input.device)
#             Bin[range(self.n_harmonics), range(self.n_harmonics)] = basis[i]
#             B.append(Bin)
#         B = torch.cat(B, 1).permute(1, 0).to(self.input.device)
#         X = torch.matmul(coeffs, B)
#         return X.sum(2)
#
#     def forward(self, x):
#         assert x.size(1) == (self.in_features + self.latent_dimension + 1) # x should be ordered in input_data, s, latent_variables
#         self.batch_size = x.size(0)
#         s = x[-1, self.in_features]
#         self.input = torch.unsqueeze(x[:, :self.in_features], dim=-1)
#         latent_variables = x[:,-self.latent_dimension:]
#
#         coeffs = self.coeffs_generator(latent_variables).reshape(self.batch_size, (self.in_features + 1) * self.out_features, self.n_eig * self.n_harmonics)
#
#         w = self.assign_weights(s, coeffs)
#         self.weight = w[:, :(self.in_features * self.out_features)].reshape(self.batch_size, self.in_features, self.out_features)
#         self.bias = w[:, (self.in_features * self.out_features):((self.in_features + 1) * self.out_features)].reshape(self.batch_size, self.out_features)
#
#         self.weighted = torch.squeeze(torch.bmm(self.input, self.weight), dim=1)
#         return torch.add(self.weighted, self.bias)

'''
change shift and dilation
'''

class WeightAdaptiveGallinear(nn.Module):
    def __init__(self, hidden_dim, in_features = 1, out_features = 16,
                 latent_dimension = 3, expfunc = batch_fourier_expansion, n_harmonics = 5, n_eig = 2):
        super().__init__()

        self.in_features, self.out_features, self.latent_dimension = in_features, out_features, latent_dimension
        # self.dilation = torch.ones(1) if not dilation else nn.Parameter(data = torch.ones(1), requires_grad=True)
        # self.shift = torch.zeros(1) if not shift else nn.Parameter(data=torch.zeros(1), requires_grad=True)
        self.expfunc = expfunc
        self.n_harmonics, self.n_eig = n_harmonics, n_eig

        coeffs_size = ((in_features + 1) * out_features + 2)* n_harmonics * n_eig   # +1 more dilation and shift

        # latent_dimension 3 means amp, sign, phase
        self.coeffs_generator = CoeffDecoder(latent_dimension, hidden_dim, coeffs_size)

    def assign_weights(self, s, coeffs, dilation, shift):
        n_range = torch.Tensor([0., 1., 2.]).to(self.input.device)
        # n_range = torch.linspace(0, self.n_harmonics, steps=self.n_harmonics).to(self.input.device)
        s = s.new_full((self.batch_size, self.n_eig * self.n_harmonics), s)
        s = (s * dilation) + shift

        B = self.expfunc(n_range, s)   ## shape of (batch_size, n_eig * n_harmonics, n_harmonics)
        # B = []
        # for i in range(self.n_eig):
        #     Bin = torch.eye(self.n_harmonics).to(self.input.device)
        #     Bin[range(self.n_harmonics), range(self.n_harmonics)] = basis[i]
        #     B.append(Bin)
        # B = torch.cat(B, 1).permute(1, 0).to(self.input.device)   # shape of (n_harmonics*n_eig, n_harmonics)
        # B = torch.matmul(torch.unsqueeze(dilation, dim=1), B)
        X = torch.bmm(coeffs, B)
        return X.sum(2)

    def forward(self, x):
        assert x.size(1) == (self.in_features + self.latent_dimension + 1) # x should be ordered in input_data, s, latent_variables
        self.batch_size = x.size(0)
        s = x[-1, self.in_features]
        self.input = torch.unsqueeze(x[:, :self.in_features], dim=-1)   # shape of (batch_size x in_feature x 1)
        latent_variables = x[:,-self.latent_dimension:]  # shape of (batch_size x latent_dim)

        coeffs = self.coeffs_generator(latent_variables).reshape(self.batch_size, ((self.in_features + 1) * self.out_features + 2)* self.n_eig * self.n_harmonics)
        self.coeff = coeffs[:, :((self.in_features + 1) * self.out_features * self.n_eig * self.n_harmonics)].reshape(self.batch_size, (self.in_features + 1) * self.out_features, self.n_eig * self.n_harmonics)
        self.dilation = coeffs[:, ((self.in_features + 1) * self.out_features * self.n_eig * self.n_harmonics): (((self.in_features + 1) * self.out_features + 1)* self.n_eig * self.n_harmonics)]
        self.shift = coeffs[:, (((self.in_features + 1) * self.out_features + 1)* self.n_eig * self.n_harmonics):]

        w = self.assign_weights(s, self.coeff, self.dilation, self.shift)
        self.weight = w[:, :(self.in_features * self.out_features)].reshape(self.batch_size, self.in_features, self.out_features)
        self.bias = w[:, (self.in_features * self.out_features):((self.in_features + 1) * self.out_features)].reshape(self.batch_size, self.out_features)

        self.weighted = torch.squeeze(torch.bmm(self.input, self.weight), dim=1)
        return torch.add(self.weighted, self.bias)

class AugmentedGalerkin(nn.Module):
    def __init__(self, hidden_dim, in_features, out_features, latent_dim, expfunc, n_harmonics, n_eig):
        super().__init__()
        self.depth_cat = DepthCat(1)
        if expfunc=='fourier':
            expfunc = fourier_expansion
        self.gallinear = WeightAdaptiveGallinear(hidden_dim=hidden_dim, in_features=in_features, out_features=out_features, latent_dimension=latent_dim,
                                                  expfunc=expfunc, n_harmonics=n_harmonics, n_eig=n_eig)
        self.z = None

    def forward(self, x):
        # return dynamics
        # input: input vector (consists of [input, amp, sign, phase])
        # output: dynamics in the current time
        x = self.depth_cat(x)
        x = torch.cat((x, self.z), 1)
        out = self.gallinear(x)
        return out

class GalerkinDE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.func = AugmentedGalerkin(hidden_dim=args.model['in_feature'],
                                  in_features=args.model['in_feature'],
                                  out_features=args.model['out_feature'],
                                  latent_dim=args.model['latent_dim'],
                                  expfunc=args.model['expansion_type'],
                                  n_harmonics=args.model['n_harmonics'],
                                  n_eig=args.model['n_eig']).to(args.device)

        self.galerkine_ode = NeuralDE(self.func, solver='dopri5', order=2).to(args.device)

    def forward(self, t, x, latent_v):
        z = latent_v
        self.func.z = z
        pred = self.galerkine_ode.trajectory(x, t)
        return pred