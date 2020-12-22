import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
from models.rmsn.variational_rnn import VRNN
from models.rmsn.censor_network import CensorNet

class PropensityNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        # def __init__(self, x_dim, h_dim, z_dim, n_layers, bias=False):
        A_dim = args.model['intervention_dim']
        H_dim = args.model['intervention_dim'] + args.model['observation_dim']
        out_dim = args.model['observation_dim']

        st_h_dim = args.model['st_hidden_dim']
        cs_h_dim = args.model['cs_hidden_dim']

        z_dim = args.model['latent_dim']
        n_layer = args.model['layer']

        # stabilized_weights
        self.st_numer = VRNN(x_dim=A_dim, h_dim=st_h_dim, z_dim=z_dim, n_layers=n_layer)
        self.st_n_optim = get_optimizer(args, self.st_numer)

        self.st_denor = VRNN(x_dim=H_dim, h_dim=st_h_dim, z_dim=z_dim,
                             outdim=out_dim, n_layers=n_layer)
        self.st_d_optim = get_optimizer(args, self.st_denor)

        # probability mass function; Is cencoring false?
        self.censor_numer = CensorNet(A_dim, cs_h_dim)#nn.GRU(A_dim, cs_h_dim, n_layer)
        self.cs_n_optim = get_optimizer(args, self.censor_numer)

        self.censor_denor = CensorNet(H_dim, cs_h_dim)#nn.GRU(H_dim+1, cs_h_dim, n_layer) # time_dependent
        self.cs_d_optim = get_optimizer(args, self.censor_denor)

    def forward(self, t, x):
        pass

def get_optimizer(args, model):
    if args.model['optimizer']=='ADAM':
        print('optimizer: {}'.format(args.model['optimizer']))
        optim = torch.optim.Adam(model.parameters(), args.lr, amsgrad=False)
    return optim
