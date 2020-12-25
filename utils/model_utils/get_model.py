from models.vanilla_node import NeuralODE
from models.generative_ode import GalerkinDE
from models.latent_ode import LatentODE
from models.fourier_series import FourierSeries
from models.siren import SirenWrapper
from models.rmsn.propensity_net import PropensityNet

import torch

def get_model(args):
    if args.model['name']=='fourier_series':
        model = FourierSeries(args)
        return model, 'dummy'
    if args.model['name'] == 'vanilla':
        model = NeuralODE(args)
        optim = get_optimizer(args, model)
    elif args.model['name'] == 'generative' or args.model['name'] == 'irregular':
        model = GalerkinDE(args)
        optim = get_optimizer(args, model)
    elif args.model['name'] =='latentode':
        model = LatentODE(args)
        optim = model.optimizer
    elif args.model['name'] == 'siren':
        """
        def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True,
             first_omega_0=30, hidden_omega_0=30.):
        """
        model = SirenWrapper(args)
        optim = get_optimizer(args, model)
    elif args.model['name']== 'rmsn':
        if args.model['phase']==1:
            # learning propensity network
            args.torch_device = torch.device("cuda")
            model = PropensityNet(args)
            optim = None
        elif args.model['phase']==2:
            model = PropensityNet(args)
            optim = None
        elif args.model['phase']==3:
            model = PropensityNet(args)
            optim = None

    print("model: {}, number of params: {}".format(args.model['name'], count_parameters(model)))
    return model, optim

def get_optimizer(args, model):
    if args.model['optimizer']=='ADAM':
        print('optimizer: {}'.format(args.model['optimizer']))
        optim = torch.optim.Adam(model.parameters(), args.lr, amsgrad=False)
    return optim

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_pretrained_dict(args):
    return torch.load(args.pretrained['path'])