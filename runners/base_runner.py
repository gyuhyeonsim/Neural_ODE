import torch
import torch.nn as nn
import random
from torch.utils.tensorboard import SummaryWriter

class Runner():
    def __init__(self, args, data_loader, model, optimizer):
        self.args = args
        self.dl = data_loader
        self.torch_device = torch.device("cuda")
        self.model = model.to(self.torch_device)
        self.optimizer = optimizer
        self.seed = random.randint(0, 10000)
        self.writer = SummaryWriter("./runs/{}".format(self.args.exid))
        self.metric = dict()

        # for debug
        self.debug = args.debug  # True if debug mode
        self.cal_loss = nn.MSELoss()
    def train(self):
        # train
        iter = 0
        self.model.train()
        for i in range(self.args.niter):
            for j, b in enumerate(self.dl['train']):
                t = b['t'][0].to(self.torch_device).float()  # all data entries share the same time point
                point = b['point'].to(self.torch_device).float()
                freq = b['freq'].unsqueeze(1).float()
                amp = b['amp'].unsqueeze(1).float()
                sign = b['sign'].unsqueeze(1).float()
                latent_v = torch.cat([freq, amp, sign], dim=1).to(self.torch_device)
                #print(t.size(), point.size(), freq.size(), amp.size())
                # >> torch.Size([64, 100]) torch.Size([64, 100, 1]) torch.Size([64]) torch.Size([64])
                # print(torch.cat([freq, amp, sign], dim=1).size())
                # >> torch.Size([64, 3])
                pred = self.model(t, point[:,0,:], latent_v=latent_v).permute(1,0,2)
                loss = self.cal_loss(pred, point)
                loss.backward()
                self.optimizer.step()

                print("epoch:{}/{}, iter:{}/{}, loss: {}".format(i,self.args.niter,
                                                                 j, len(self.dl['train']),
                                                                 loss.item()))
                iter+=1

    def valid(self):
        self.model.eval()
        with torch.no_grad:
            pass