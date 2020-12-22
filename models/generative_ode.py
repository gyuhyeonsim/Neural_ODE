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

def fourier_expansion(n_range, s, aew):
    """Fourier eigenbasis expansion
    """
    # print("s: {}".format(s))
    # print(s.size(), n_range.size())
    if aew:
        basis=[]
        for j in range(n_range.size(0)):
            for i in range(s.size(1)):  # n_range
                basis.append([torch.cos(s[0,i,0]*n_range[j]), torch.sin(s[0,i,1]*n_range[j])])
        basis = torch.Tensor(basis)
    else:
        s_n_range = s * n_range
        basis = [torch.cos(s_n_range), torch.sin(s_n_range)]

    return basis

def poly_expansion(n_range, s):
    """Polynomial expansion
    """
    basis = [s**n_range]
    return basis

class CoeffDecoder(nn.Module):
    def __init__(self, latent_dimension, hidden_dim,  coeffs_size, merger):
        super().__init__()
        self.latent_dimension = latent_dimension
        self.fc1 = nn.Linear(latent_dimension, hidden_dim)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, coeffs_size)
        # self.act2 = nn.Tanh()
        self.merger = merger
        if merger==True:
            self.coeff_distributer = nn.Linear(coeffs_size, coeffs_size)
        self.flag = True

        print('decoder output size: {}'.format(coeffs_size))

    def forward(self, x):
        # input latent vector
        out = self.act1(self.fc1(x))
        out = self.fc2(out)
        if self.merger:
            if self.flag:
                print(out[0])
            out = self.coeff_distributer(out)
            if self.flag:
                print(out[0])
            self.flag = False
        # out = self.act2(self.fc2(out))
        return out

class WeightAdaptiveGallinear(nn.Module):
    def __init__(self, hidden_dim, in_features = 1, out_features = 16,
                 latent_dimension = 3, expfunc = fourier_expansion, n_harmonics = 5,
                 n_eig = 2, dilation = False, shift = False, har_mode=4, merger=False, all_element_wise=True):
        super().__init__()

        self.in_features, self.out_features, self.latent_dimension = in_features, out_features, latent_dimension
        if all_element_wise==False:
            self.dilation = torch.ones(1) if not dilation else nn.Parameter(data=torch.ones(1), requires_grad=True)
            self.shift = torch.zeros(1) if not shift else nn.Parameter(data=torch.zeros(1), requires_grad=True)
        else:
            self.dilation1 = nn.Parameter(data=torch.ones(1), requires_grad=True)
            self.shift1 = nn.Parameter(data=torch.zeros(1), requires_grad=True)
            self.dilation2 = nn.Parameter(data=torch.ones(1), requires_grad=True)
            self.shift2 = nn.Parameter(data=torch.zeros(1), requires_grad=True)
            self.dilation3 = nn.Parameter(data=torch.ones(1), requires_grad=True)
            self.shift3 = nn.Parameter(data=torch.zeros(1), requires_grad=True)
            self.dilation4 = nn.Parameter(data=torch.ones(1), requires_grad=True)
            self.shift4 = nn.Parameter(data=torch.zeros(1), requires_grad=True)


        self.aew = all_element_wise
        self.expfunc = expfunc
        self.n_harmonics, self.n_eig = n_harmonics, n_eig
        self.har_mode = har_mode
        self.flag = True
        coeffs_size = (in_features + 1) * out_features * n_harmonics * n_eig

        # latent_dimension 3 means amp, sign, phase
        self.coeffs_generator = CoeffDecoder(latent_dimension, hidden_dim, coeffs_size, merger)
        self.depth_cat = DepthCat(1)

    def assign_weights(self, s, coeffs):
        if self.har_mode==1:
            n_range = torch.Tensor([1]).to(self.input.device)
        elif self.har_mode==2:
            n_range = torch.Tensor([0,0,1,1,2,2]).to(self.input.device)
            self.n_harmonics = 6
        elif self.har_mode==3:
            n_range = torch.Tensor([0,1,2]).to(self.input.device)
            self.n_harmonics = 3
        elif self.har_mode==4:
            n_range = torch.Tensor([0,1,1,1,1,2]).to(self.input.device)
            self.n_harmonics = 6
        elif self.har_mode == 5:
            n_range = torch.Tensor([1,2]).to(self.input.device)
            self.n_harmonics = 2
        else:
            n_range =  torch.linspace(0, self.n_harmonics, steps=self.n_harmonics).to(self.input.device)

        if self.aew:
            Bin = torch.zeros((2,4)).to(self.input.device)
            Bin[0][0] = torch.cos(s*self.dilation1+self.shift1)
            Bin[1][1] = torch.sin(s*self.dilation2+self.shift2)
            Bin[0][2] = torch.cos(s*self.dilation3+self.shift3)
            Bin[1][3] = torch.sin(s*self.dilation4+self.shift4)
            X = torch.matmul(coeffs, Bin)
            # print(self.dilation1)
        else:
            basis = self.expfunc(n_range,
                                 s * self.dilation.to(self.input.device) + self.shift.to(self.input.device),self.aew)
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
        latent_variables = x[:,-self.latent_dimension:]

        coeffs = self.coeffs_generator(latent_variables).reshape(self.batch_size, (self.in_features + 1) * self.out_features, self.n_eig * self.n_harmonics)
        if self.flag:
            print(coeffs[0], self.dilation1,self.dilation2,self.dilation3,self.dilation4,
                  self.shift1,self.shift2,self.shift3,self.shift4)
            self.flag=False

        w = self.assign_weights(s, coeffs)
        self.weight = w[:, :(self.in_features * self.out_features)].reshape(self.batch_size, self.in_features, self.out_features)
        self.bias = w[:, (self.in_features * self.out_features):((self.in_features + 1) * self.out_features)].reshape(self.batch_size, self.out_features)

        self.weighted = torch.squeeze(torch.bmm(self.input, self.weight), dim=1)
        return torch.add(self.weighted, self.bias)

class AugmentedGalerkin(nn.Module):
    def __init__(self, hidden_dim, in_features, out_features, latent_dim, expfunc, n_harmonics, n_eig, har_mode, merger=False, alw=False):
        super().__init__()
        self.depth_cat = DepthCat(1)
        if expfunc=='fourier':
            expfunc = fourier_expansion
        self.gallinear = WeightAdaptiveGallinear(hidden_dim=hidden_dim, in_features=in_features, out_features=out_features,
                                                 latent_dimension=latent_dim, expfunc=expfunc, n_harmonics=n_harmonics, n_eig=n_eig,
                                                 har_mode=har_mode, merger = merger, all_element_wise=alw)
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
                                  n_eig=args.model['n_eig'],
                                  har_mode=args.model['har_mode'],
                                  merger = args.model['merger'],
                                  alw =args.model['all_element_wise']).to(args.device)

        self.galerkine_ode = NeuralDE(self.func, solver='dopri5', order=2).to(args.device)

    def forward(self, t, x, latent_v):
        z = latent_v
        self.func.z = z
        pred = self.galerkine_ode.trajectory(x, t)
        return pred