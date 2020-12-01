
import numpy as np
import pickle
import os
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from torchdyn.models import DepthCat, NeuralDE

def batch_fourier_expansion(n_range, s):
    # s is shape of (batch_size x n_eig * n_harmonics)
    cos_s = s[:, :n_range.size(0)] * n_range
    cos_s = torch.diag_embed(torch.cos(cos_s))
    sin_s = s[:, n_range.size(0):] * n_range
    sin_s = torch.diag_embed(torch.sin(sin_s))
    return torch.cat((cos_s, sin_s), dim=1)


class CoeffDecoder(nn.Module):
    def __init__(self, latent_dimension, hidden_dim,  coeffs_size):
        super().__init__()
        self.latent_dimension = latent_dimension
        self.fc1 = nn.Linear(latent_dimension, coeffs_size)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(coeffs_size, coeffs_size)
        self.act2 = nn.Tanh()
        self.fc3 = nn.Linear(coeffs_size, coeffs_size)
        print('coeffs + dilation size: {}'.format(coeffs_size))

    def forward(self, x):
        # input latent vector
        out = self.act1(self.fc1(x))
        out = self.act2(self.fc2(out))
        return self.fc3(out)

class DilationDecoder(nn.Module):
    def __init__(self, latent_dimension, hidden_dim, dilation_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_dimension, hidden_dim)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, dilation_size)
        self.act2 = nn.Tanh()
        self.fc3 = nn.Linear(dilation_size, dilation_size)

    def forward(self, x):
        out = self.act1(self.fc1(x))
        return self.fc2(out)


class WeightAdaptiveGallinear(nn.Module):
    def __init__(self, hidden_dim, in_features = 1, out_features = 1,
                 latent_dimension = 3, expfunc = batch_fourier_expansion, n_harmonics = 3, n_eig = 2):
        super().__init__()

        self.in_features, self.out_features, self.latent_dimension = in_features, out_features, latent_dimension
        # self.dilation = torch.ones(1) if not dilation else nn.Parameter(data = torch.ones(1), requires_grad=True)
        # self.shift = torch.zeros(1) if not shift else nn.Parameter(data=torch.zeros(1), requires_grad=True)
        self.expfunc = expfunc
        self.n_harmonics, self.n_eig = n_harmonics, n_eig

        coeffs_size = ((in_features + 1) * out_features)* n_harmonics * n_eig
        dilation_size = n_harmonics * n_eig

        # latent_dimension 3 means amp, sign, phase
        self.coeffs_generator = CoeffDecoder(latent_dimension, hidden_dim, coeffs_size)
        self.dilation_generator = DilationDecoder(latent_dimension, hidden_dim, dilation_size)

    def assign_weights(self, s, coeffs, dilation):
        n_range = torch.Tensor([1.] * self.n_harmonics).to(self.input.device)
        # n_range = torch.linspace(0, self.n_harmonics, steps=self.n_harmonics).to(self.input.device)
        s = s.new_full((self.batch_size, self.n_eig * self.n_harmonics), s.item())
        s = s * dilation

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

        self.coeff = self.coeffs_generator(latent_variables).reshape(self.batch_size, (self.in_features + 1) * self.out_features , self.n_eig * self.n_harmonics)
        #self.coeff = coeffs[:, :((self.in_features + 1) * self.out_features * self.n_eig * self.n_harmonics)].reshape(self.batch_size, (self.in_features + 1) * self.out_features, self.n_eig * self.n_harmonics)
        self.dilation = self.dilation_generator(latent_variables).reshape(self.batch_size, self.n_eig * self.n_harmonics)
        #self.dilation = coeffs[:, ((self.in_features + 1) * self.out_features * self.n_eig * self.n_harmonics): (((self.in_features + 1) * self.out_features + 1)* self.n_eig * self.n_harmonics)]
        #self.shift = coeffs[:, (((self.in_features + 1) * self.out_features + 1)* self.n_eig * self.n_harmonics):]

        w = self.assign_weights(s, self.coeff, self.dilation)
        self.weight = w[:, :(self.in_features * self.out_features)].reshape(self.batch_size, self.in_features, self.out_features)
        self.bias = w[:, (self.in_features * self.out_features):((self.in_features + 1) * self.out_features)].reshape(self.batch_size, self.out_features)

        self.weighted = torch.squeeze(torch.bmm(self.input, self.weight), dim=1)
        return torch.add(self.weighted, self.bias)

class AugmentedGalerkin(nn.Module):
    def __init__(self, hidden_dim, in_features, out_features, latent_dim, expfunc, n_harmonics, n_eig):
        super().__init__()
        self.depth_cat = DepthCat(1)
        if expfunc=='fourier':
            expfunc = batch_fourier_expansion
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

        self.galerkine_ode = NeuralDE(self.func, solver='dopri5', sensitivity='autograd').to(args.device)
        self.dilation_penalty_lambda = args.model['dilation_penalty_lambda']
        self.dilation_penalty = args.model['dilation_penalty']
        self.n_harmonics = args.model['n_harmonics']

    def forward(self, t, x, latent_v):
        y0 = x[:, 0]
        t = torch.squeeze(t[0])
        z = latent_v
        self.func.z = z

        decoded_traj = self.galerkine_ode.trajectory(y0, t).transpose(0, 1)
        mse_loss = nn.MSELoss()(decoded_traj, x)

        cos_penalty = self.func.gallinear.dilation[:self.n_harmonics]
        cos_penalty = torch.einsum('bi,bj->bij', [cos_penalty, 1/cos_penalty]) - torch.eye(self.n_harmonics).to(x.device)
        sin_penalty = self.func.gallinear.dilation[self.n_harmonics:]
        sin_penalty = torch.einsum('bi,bj->bij', [sin_penalty, 1/sin_penalty]) - torch.eye(self.n_harmonics).to(x.device)

        dilation_penalty = torch.norm(cos_penalty) + torch.norm(sin_penalty)

        if self.dilation_penalty:
            loss = mse_loss - self.dilation_penalty_lambda * dilation_penalty
        else:
            loss = mse_loss
        return loss, self.dilation_penalty_lambda * dilation_penalty

    def predict(self, t, x, latent_v):
        y0 = x[:, 0]
        t = t[0]
        z = latent_v
        self.func.z = z
        decoded_traj = self.galerkine_ode.trajectory(y0, t).transpose(0, 1)
        return decoded_traj