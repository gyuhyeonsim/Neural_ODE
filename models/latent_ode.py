"""
The reference is https://github.com/rtqichen/torchdiffeq
"""
import torch
import torch.nn as nn
import numpy as np
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

    def initHidden(self,batch_size):
        return torch.zeros(batch_size, self.nhidden)


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
        h = self.rec.initHidden(x.size(0)).to(self.args.device)

        for t_ in reversed(range(self.args.dataset['interpolation']//2)):
            obs = x[:, t_, :]
            out, h = self.rec.forward(obs, h)
        qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(self.args.device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

        # forward in time and solve ode for reconstructions
        pred_z = odeint(self.func, z0, t).permute(1, 0, 2)
        pred_x = self.dec(pred_z)

        # compute loss
        noise_std_ = torch.zeros(pred_x.size()).to(self.args.device) + .3
        noise_logvar = 2. * torch.log(noise_std_).to(self.args.device)
        logpx = self.log_normal_pdf(
            x, pred_x, noise_logvar).sum(-1).sum(-1)
        pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(self.args.device)
        analytic_kl = self.normal_kl(qz0_mean, qz0_logvar,
                                pz0_mean, pz0_logvar).sum(-1)
        loss = torch.mean(-logpx + analytic_kl, dim=0)

        return pred_x, loss

    def infer(self, x, t):
        h = self.rec.initHidden(x.size(0)).to(self.args.device)

        for t_ in reversed(range(self.args.dataset['interpolation'] // 2)):
            obs = x[:, t_, :]
            out, h = self.rec.forward(obs, h)

        qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(self.args.device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

        # forward in time and solve ode for reconstructions
        pred_z = odeint(self.func, z0, t).permute(1, 0, 2)
        pred_x = self.dec(pred_z)
        return pred_x

    def log_normal_pdf(self, x, mean, logvar):
        const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
        const = torch.log(const)
        return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


    def normal_kl(self, mu1, lv1, mu2, lv2):
        v1 = torch.exp(lv1)
        v2 = torch.exp(lv2)
        lstd1 = lv1 / 2.
        lstd2 = lv2 / 2.

        kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
        return kl
