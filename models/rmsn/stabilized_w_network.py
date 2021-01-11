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
            nn.Linear(h_dim, outdim*2))  # mean, std
        self.mse_loss = nn.MSELoss(reduction='sum')

        obs_noise_std = 1e-2
        self.obs_noise_std = torch.tensor(obs_noise_std)

    def forward(self, x, gt=None):
        all_dec_mean = []
        all_dec_std = []
        mse_loss = 0
        log_like = 0
        h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(self.args.device)
        if gt is not None:
            dlen = x.size(0)
            for t in range(dlen):
                # recurrence
                _, h = self.rnn(x[t].unsqueeze(0), h)
                pred = self.dec(h).squeeze(0)
                # print(pred.size())  -> [1, 32, 5]
                mean, var = torch.chunk(pred, 2, dim=1)
                all_dec_mean.append(mean)
                all_dec_std.append(var)
                std = torch.pow(torch.abs(var) + 1e-5, 0.5)

                log_like += gaussian_KL(mean, gt[t], std, self.obs_noise_std).sum()#self.mse_loss(pred.squeeze(0), gt[t])
                mse_loss += self.mse_loss(mean, gt[t])
        else:
            dlen = x.size(0)-1
            for t in range(dlen):
                # recurrence
                _, h = self.rnn(x[t].unsqueeze(0), h)
                pred = self.dec(h).squeeze(0)
                mean, var = torch.chunk(pred, 2, dim=1)
                all_dec_mean.append(mean)
                all_dec_std.append(var)
                std = torch.pow(torch.abs(var) + 1e-5, 0.5)

                log_like += gaussian_KL(mean, x[t+1], std, self.obs_noise_std).sum()#self.mse_loss(pred.squeeze(0), x[t+1])
                mse_loss += self.mse_loss(mean, x[t+1])

        return log_like, mse_loss, (all_dec_mean, all_dec_std)

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

def gaussian_KL(mu_1, mu_2, sigma_1, sigma_2):
    """ 여기서 sigma_2로 나누는거 때문에 0으로 casting되어 에러가 발생할 수도 있다."""
    return (torch.log(sigma_2) - torch.log(sigma_1) + (torch.pow(sigma_1, 2) + torch.pow((mu_1 - mu_2), 2)) / (
                2 * sigma_2 ** 2) - 0.5)
