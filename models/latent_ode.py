"""
The reference is https://github.com/rtqichen/torchdiffeq
"""
import torch
import torch.nn as nn
from torchdiffeq import odeint


class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class RecognitionRNN(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)


class Decoder(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class LatentODE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.latent_dim = args.model['latent_dim']
        nhidden = args.model['nhidden']
        obs_dim = args.model['obs_dim']
        rnn_nhidden = args.model['rnn_nhidden']
        nspiral = args.dataset['batch']  ## for test or valid?, not used in training

        self.args = args
        self.func = LatentODEfunc(self.latent_dim, nhidden)
        self.rec = RecognitionRNN(self.latent_dim, obs_dim, rnn_nhidden, nspiral)
        self.dec = Decoder(self.latent_dim, obs_dim, nhidden)
        params = (list(self.func.parameters()) + list(self.dec.parameters()) +
                       list(self.rec.parameters()))
        self.optimizer = torch.optim.Adam(params, lr=args.lr)

    def forward(self, x, t):
        h = self.rec.initHidden().to(self.args.device)
        for t_ in reversed(range(self.args.dataset['interpolation'])):
            obs = x[:, t_, :]
            out, h = self.rec.forward(obs, h)
        qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(self.args.device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

        # forward in time and solve ode for reconstructions
        pred_z = odeint(self.func, z0, t).permute(1, 0, 2)
        pred_x = self.dec(pred_z)
        return pred_x

    def infer(self):
        pass