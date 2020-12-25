import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class CensorNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # self.encoder = nn.Linear(input_size, hidden_size)
        self.model = nn.GRUCell(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, 1)
        self.epsilon = 0.0000001
        self.act = nn.Sigmoid()
        self.n_layers = 1
        self.h_dim = hidden_size
        self.BCE_loss = nn.BCELoss(reduction='sum')

    def forward(self, x, gt=None):
        h = Variable(torch.zeros(x.size(1), self.h_dim))
        # print(h.size(), x.size())
        # >> torch.Size([64, 64]) torch.Size([50, 64, 2])
        nll_loss = 0

        for t in range(x.size(0)-1):
            # h = self.encoder(x[t])
            h = self.model(x[t], h) # update hidden
            C = self.decoder(h)
            C = self.act(C)
            # nll_loss += self._nll_bernoulli(C, gt[t+1])
            # print(C.size(), gt[t+1].size())
            nll_loss += self.BCE_loss(C, gt[t+1])
        # nll_loss/=torch.ones_like(gt).sum()
        return nll_loss

    def infer(self, x):
        h = Variable(torch.zeros(x.size(1), self.h_dim))
        C_list = []

        for t in range(x.size(0)-1):
            h = self.model(x[t], h) # update hidden
            C = self.decoder(h)
            C = self.act(C)
            C_list.append(C)
        # nll_loss/=torch.ones_like(gt).sum()
        return 1-torch.stack(C_list)  # C=0

    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x * torch.log(theta+self.epsilon) +
                           (1 - x) * torch.log(1 - theta+self.epsilon))
