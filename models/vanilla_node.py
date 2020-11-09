import torch
import torch.nn as nn

# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint

class NN(nn.Module):
    def __init__(self, args):
        super(NN, self).__init__()
        self.args = args
        self.function = nn.Sequential(
            nn.Linear(args.model['input_size'], args.model['hidden_size']),
            nn.Tanh(),
            nn.Linear(args.model['hidden_size'], args.model['input_size']),
        )

        for m in self.function.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, x):
        # t-invariant
        return self.function(x)
class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        self.encoder = nn.Linear(args.model['input_size'], args.model['hidden_size'])
        self.decoder = nn.Linear(args.model['hidden_size'], args.model['input_size'])
        self.gru = nn.GRUCell(args.model['input_size'], args.model['hidden_size'])

    def forward(self, t, x):
        return self.gru(x)

class NeuralODE(nn.Module):
    def __init__(self, args):
        super(NeuralODE, self).__init__()
        self.args = args
        self.device = args.device

        if args.model['ode_func'] == 'nn':
            self.func = NN(args)
            self.func.to(self.device)
            self.func.double()
        elif args.model['ode_func'] == 'rnn':
            self.func = RNN(args)
            self.encoder = self.func.encoder # linear encoder
            self.decoder = self.func.decoder # linear decoder

    # train
    def forward(self, x0, t):
        pred_y = odeint(self.func, x0, t, method='dopri5').permute(1, 0, 2)
        return pred_y