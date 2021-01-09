import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
# from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

class StabilizeNet(nn.Module):
    def __init__(self, args, x_dim, h_dim, z_dim, n_layers, outdim=None, bias=False):
        super(StabilizeNet, self).__init__()

        self.args = args
        self.x_dim = x_dim # indim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.obs_noise_std=1e-2

        # recurrence
        # self.enc = nn.Sequential(
        #     nn.Linear(x_dim, h_dim),
        #     nn.ReLU(),
        #     nn.Linear(h_dim, h_dim))
        self.rnn = nn.GRU(x_dim, h_dim, n_layers, bias)
        self.dec = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, outdim))
        self.mse_loss = nn.MSELoss(reduction='sum')

    def forward(self, x, gt=None):
        all_dec_mean = []
        mse_loss = 0
        h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(self.args.device)
        if gt is not None:
            dlen = x.size(0)
            for t in range(dlen):
                # recurrence
                _, h = self.rnn(x[t].unsqueeze(0), h)
                pred = self.dec(h)
                all_dec_mean.append(pred)
                mse_loss += self.mse_loss(pred.squeeze(0), gt[t])
        else:
            dlen = x.size(0)-1
            for t in range(dlen):
                # recurrence
                _, h = self.rnn(x[t].unsqueeze(0), h)
                pred = self.dec(h)
                all_dec_mean.append(pred)
                mse_loss += self.mse_loss(pred.squeeze(0), x[t+1])


        return mse_loss, all_dec_mean

    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim)

        h = Variable(torch.zeros(self.n_layers, 1, self.h_dim))
        for t in range(seq_len):
            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)
            phi_x_t = self.phi_x(dec_mean_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data

        return sample

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""
        kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                       std_2.pow(2) - 1)
        return 0.5 * torch.sum(kld_element)

    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x * torch.log(theta) + (1 - x) * torch.log(1 - theta))

    def _nll_gauss(self, mean, std, x):
        return