import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class DecoderNet(nn.Module):
    def __init__(self, args, input_hidden_size, hidden_size, itv_dim, obs_dim):
        super().__init__()
        self.args = args
        self.obs_dim = obs_dim

        # ELU non-linear adaptor
        self.memory_adaptor = nn.Sequential(
                                nn.Linear(input_hidden_size, hidden_size),
                                nn.ELU())
        self.model = nn.GRUCell(itv_dim, hidden_size)
        self.decoder = nn.Linear(hidden_size, obs_dim)

        self.epsilon = 0.1
        self.loss = nn.MSELoss(reduction='sum')
        self.n_layers = 1
        self.h_dim = hidden_size

        print('[Decoder Net] Decoder is initialized')

    def forward(self, init_hidden, itv, mask, weight, x, valid=False):
        # with truncated datasets (weight and init_hidden are given.)
        h = self.memory_adaptor(init_hidden)
        itv = itv.permute(1,0,2)
        pred_list = []
        for t in range(itv.size(0)):
            h = self.model(itv[t], h) # update hidden
            pred = self.decoder(h)
            pred_list.append(pred)

        pred_list = torch.stack(pred_list).permute(1,0,2)
        SW = weight[:,:,0].unsqueeze(2)
        SW = SW/(SW.sum()/mask.sum())

        CW = weight[:, :,1].unsqueeze(2)
        CW = CW / (CW.sum() / mask.sum())

        mse_loss = torch.pow((pred_list-x), 2)
        mse_loss = torch.sum(mse_loss*mask, dim=2).unsqueeze(2)

        if not valid:
            mse_loss *= SW
            mse_loss *= CW
        mse_loss = mse_loss.sum()

        assert mse_loss.item()>=0
        return mse_loss

    def forward_nontruncated(self, x, itv, encoder, valid=False):
        x = x.permute(1,0,2).to(self.args.device)
        itv = itv.permute(1,0,2)[:,:,:self.args.model['intervention_dim']].to(self.args.device)
        time_to_encode = int(x.size(0)*self.args.tte)

        # for t in range(time_to_encode):
        with torch.no_grad():
            # hidden state in time 0 is defined in the encoder network
            H = torch.cat([x[:time_to_encode],
                           itv[:time_to_encode,:,:]],
                           dim=2)
            _,_,h = encoder(H, gt=x[1:time_to_encode+1], valid=True, jupyter=True)
            decoder_inital_state = h[h.size(0)-1,:,:]
        h = self.memory_adaptor(decoder_inital_state)

        pred_list = []
        for t in range(time_to_encode, itv.size(0)-1):
            h = self.model(itv[t], h) # update hidden
            pred = self.decoder(h)
            pred_list.append(pred)
        pred_list = torch.stack(pred_list)
        # >> torch.Size([44, 64, 2])

        # calculate propensity_weight
        with torch.no_grad():
            # predict Censoring Weight
            censor_numerator = encoder.propensity.censor_numer.infer(itv)  # update hidden
            censor_denominator = encoder.propensity.censor_denom.infer(torch.cat([x,itv],dim=2))
            CW = (censor_numerator+ self.epsilon / (censor_denominator + self.epsilon))  # Censoring Weight
            # print(CW.size())
            # >> torch.Size([49, 64, 1])

            CW = CW[time_to_encode-2:-1,:,:] # t=4~ t=48까지 필요하다
            for i in range(1, CW.size(0)):
                CW[i] = CW[i - 1] * CW[i]
            proc_CW = CW[1:]
            CW = proc_CW
            CW = CW/((CW.sum()/torch.ones_like(pred_list).sum()+self.epsilon))
            # print(CW.size())
            # exit()
            # >> torch.Size([44, 64, 1])

            # predict Stabilzed Weight
            _, _, sw_numerator = encoder.propensity.st_numer(itv)
            A_mean = torch.stack(sw_numerator[0])
            A_std = torch.stack(sw_numerator[1])
            sw_numerator = self.gaussian_dist(itv[1:], A_mean, A_std)
            H = torch.cat([x[1:], itv[:x.size(0) - 1]], dim=2)
            # print(H.size())
            # >> torch.Size([49, 64, 2])

            _, _, sw_demoninator = encoder.propensity.st_denom(x=H, gt=itv)
            A_mean = torch.stack(sw_demoninator[0])
            A_std = torch.stack(sw_demoninator[1])
            sw_demoninator = self.gaussian_dist(itv[1:], A_mean, A_std)
            # print(sw_demoninator.size(), sw_numerator.size())
            # torch.Size([49, 64, 2]) torch.Size([49, 64, 2])

            SW = 1
            for i in range(sw_numerator.size(2)):
                SW *= (sw_numerator[:, :, i]+self.epsilon) / (sw_demoninator[:, :, i] + self.epsilon)
            SW = SW.unsqueeze(2)
            SW = SW[time_to_encode-2:-1,:,:] # t=4~ t=48까지 필요하다

            for i in range(1, SW.size(0)):
                SW[i] = SW[i - 1] * SW[i]

            proc_CW = SW[1:]
            SW = proc_CW
            SW = SW / ((SW.sum() / (torch.ones_like(pred_list[1:, :, :]).sum())+self.epsilon)+self.epsilon)

        mse_loss = torch.pow((pred_list-x[time_to_encode+1:]), 2)
        mse_loss = torch.sum(mse_loss, dim=2).unsqueeze(2)

        if not valid:
            mse_loss *= SW
            mse_loss *= CW
        mse_loss = mse_loss.sum()
        return mse_loss

    def gaussian_dist(self, obs, mean, var):
        """
        :param obs:
        :param mean:
        :param var:
        :return: p1 * p2
        """
        return torch.exp((-0.5)*torch.pow((obs-mean)/(var+self.epsilon),2))/(var+self.epsilon)