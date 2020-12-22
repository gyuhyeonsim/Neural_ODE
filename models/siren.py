"""
reference: https://github.com/vsitzmann/siren
"""

import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        x = self.linear(input) # self.omega_0 * self.linear(input)
        return torch.pow(torch.sin(x),2)+x

###### self.net is used for dynamics function
class Siren(nn.Module):
    def __init__(self, args, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        in_features = args.model['in_feature'] + args.model['latent_dim']
        hidden_features = args.model['hidden_dim']
        hidden_layers = args.model['layer']
        out_features = args.model['out_feature']
        latent = 0 # pre-defined latent_vector

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, t, x):
        # x = x#coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(torch.cat([x, self.latent], dim=1))
        return output   #, coords

class SirenWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dynamics = Siren(args)

    def forward(self, t, x0, latent_v):
        self.dynamics.latent = latent_v
        output = odeint(self.dynamics, x0, t)
        return output