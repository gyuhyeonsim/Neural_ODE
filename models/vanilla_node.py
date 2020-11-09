import torch
import torch.nn as nn

# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint

class NN(nn.Module):
    patient_type_embed = 0  # applied to all of class instances
    def __init__(self, args):
        super(NN, self).__init__()
        self.args = args

        if self.args.data=='ICLR2020':
            self.function = nn.Sequential(
                nn.Linear(args.input+args.latent, args.hidden),
                nn.Tanh(),
                nn.Linear(args.hidden, args.input),
            )
        elif self.args.data == 'tumor_growth':  # no-intervention
            self.function = nn.Sequential(
                nn.Linear(args.input, args.hidden),
                nn.Tanh(),
                nn.Linear(args.hidden, args.input),
            )
        for m in self.function.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

        # self.patient_type_embed = initial_patient_type.double()

    def forward(self, t, x):
        # t-invariant
        if self.args.intervention==True:
            return self.function(torch.cat([x,self.patient_type_embed],axis=1))
        elif self.args.intervention==False:
            return self.function(x)

class NeuralODE(nn.Module):
    def __init__(self, args, device):
        super(NeuralODE, self).__init__()
        self.args = args
        self.device = device
        """
        possible_patient_types = [1,2,3]  > latent space
        """
        self.patient_type_embed = nn.Embedding(args.ptype, args.ptype_latent).to(self.device)   # index to embedding vector
        if args.func_type == 'nn':
            self.func = NN(args)
            self.func.to(self.device)
            self.func.double()

        elif args.func_type == 'rnn':
            # linear encoding, non-linear encoding
            self.args.latent_space = True
            self.encoder = nn.Linear(args.input, args.latent) # linear encoder
            self.decoder = nn.Linear(args.latent, args.output) # linear decoder
            self.func = nn.GRUCell(args.latent, args.hidden)

    # train
    def forward(self, x, t, itv, itv_mask):
        if self.args.func_type =='nn' and self.args.intervention==True:   # intervention awaring
            x0 = x[:,0,:1]  # [batch x features]
            x1 = x[:,0,1:]  # patient type [batch x features]
            # embeds the patient type[batch x feartures]
            self.func.patient_type_embed = self.patient_type_embed(x1.long()).squeeze(1).double()
            pred_y = odeint(self.func, x0, t, method='dopri5').permute(1, 0, 2)

        elif self.args.func_type =='nn' and self.args.intervention==False:      # no intervention
            x0 = x[:,0,:]
            pred_y = odeint(self.func, x0, t[0].float(), method='dopri5').permute(1, 0, 2)
        return pred_y