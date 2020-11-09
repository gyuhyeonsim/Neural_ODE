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

"""
https://github.com/edebrouwer/gru_ode_bayes
"""
class FullGRUODECell_Autonomous(torch.nn.Module):
    def __init__(self, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()

        #self.lin_xh = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xr = torch.nn.Linear(input_size, hidden_size, bias=bias)

        #self.lin_x = torch.nn.Linear(input_size, hidden_size * 3, bias=bias)

        self.lin_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, t, h):
        """
        Executes one step with autonomous GRU-ODE for all h.
        The step size is given by delta_t.

        Args:
            t        time of evaluation
            h        hidden state (current)

        Returns:
            Updated h
        """
        #xr, xz, xh = torch.chunk(self.lin_x(x), 3, dim=1)
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))   # reset gate
        z = torch.sigmoid(x + self.lin_hz(h))   # zero gate
        u = torch.tanh(x + self.lin_hh(r * h))  # update gate

        dh = (1 - z) * (u - h)
        return dh

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
            self.func = FullGRUODECell_Autonomous(args.model['hidden_size'])
            self.encoder = self.func.encoder # linear encoder
            self.decoder = self.func.decoder # linear decoder

    # train
    def forward(self, x0, t):
        if self.args.model['ode_func'] == 'rnn':
            pred_y = odeint(self.func, x0, t, method='dopri5').permute(1, 0, 2)
        return pred_y