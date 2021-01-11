import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class EncoderNet(nn.Module):
    def __init__(self, args, input_size, hidden_size, obs_dim, propensity_net):
        super().__init__()
        self.model = nn.GRUCell(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, obs_dim)
        self.epsilon = 0.000001
        self.loss = nn.MSELoss(reduction='sum')
        self.n_layers = 1
        self.h_dim = hidden_size

        print('[Encoder Net] Encoder is initialized')
        self.propensity = propensity_net
        self.obs_dim = obs_dim
        self.args = args

    def forward(self, x, gt=None, valid=False, jupyter=False):
        """
        :param x: [L_{t-1}, A_{t-1}]
        :param gt: Y_{t}
        :return:
        """
        h = Variable(torch.zeros(x.size(1), self.h_dim)).to(self.args.device)
        mse_loss = 0
        pred_list = []
        hidden_list = []

        # predict the state
        for t in range(x.size(0)-1):
            h = self.model(x[t], h) # update hidden
            pred = self.decoder(h)
            pred_list.append(pred)
            if jupyter:
                hidden_list.append(h)

        mse_loss = torch.pow(torch.stack(pred_list)-gt[1:,:,:], 2)

        if not valid:
            mse_loss = torch.sum(mse_loss, dim=2).unsqueeze(2)
            # print(mse_loss.size())
            # >> [49, 64, 1]
            with torch.no_grad():
                # predict Censoring Weight
                censor_numerator = self.propensity.censor_numer.infer(x[:,:,self.obs_dim:]) # update hidden
                censor_denominator = self.propensity.censor_denom.infer(x)
                CW = (censor_numerator/(censor_denominator+self.epsilon)) # Censoring Weight
                mse_loss *= CW/(CW.sum()/torch.ones_like(gt[1:,:,:]).sum())

                # predict Stabilzed Weight
                _,_, sw_numerator = self.propensity.st_numer(x[:,:,self.obs_dim:])
                A_mean = torch.stack(sw_numerator[0])
                A_std = torch.stack(sw_numerator[1])
                sw_numerator = self.gaussian_dist(x[1:,:,self.obs_dim:], A_mean, A_std)

                H=torch.cat([x[1:,:,:self.obs_dim], x[:x.size(0)-1,:,self.obs_dim:]],dim=2)
                _,_, sw_demoninator = self.propensity.st_denom(x=H, gt=x[:,:,self.obs_dim:])
                A_mean = torch.stack(sw_demoninator[0])
                A_std = torch.stack(sw_demoninator[1])
                sw_demoninator = self.gaussian_dist(x[1:,:,self.obs_dim:], A_mean, A_std)

                SW=1
                for i in range(sw_numerator.size(2)):
                    SW*=sw_numerator[:,:,i]/(sw_demoninator[:,:,i]+self.epsilon)
                SW = SW.unsqueeze(2)
                SW = SW/(SW.sum()/torch.ones_like(gt[1:,:,:]).sum())

            mse_loss *= SW

        mse_loss = mse_loss.sum()
        if jupyter:
            hidden=torch.stack(hidden_list)
        else:
            hidden =None
        return mse_loss, torch.stack(pred_list), hidden

    def gaussian_dist(self, obs, mean, var):
        """
        :param obs:
        :param mean:
        :param var:
        :return: p1 * p2
        """
        return torch.exp((-0.5)*torch.pow((obs-mean)/var,2))/(var+self.epsilon)