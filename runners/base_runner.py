import os

import torch
import torch.nn as nn
import random

from torch.utils.tensorboard import SummaryWriter
from utils.run_utils.tensorboard_utils import *

class Runner():
    def __init__(self, args, data_loader, model, optimizer):
        self.args = args
        self.dl = data_loader
        self.torch_device = torch.device("cuda")
        self.model = model.to(self.torch_device)
        self.optimizer = optimizer
        self.seed = random.randint(0, 10000)
        # self.writer = SummaryWriter("./runs/{}".format(self.args.exid))
        self.metric = dict()

        # for debug
        self.debug = args.debug  # True if debug mode
        self.cal_loss = nn.MSELoss()

    def train(self):
        # train
        iter = 0
        bset_mse = float('inf')
        self.model.train()
        for i in range(self.args.niter):
            for j, b in enumerate(self.dl['train']):
                t = b['t'][0].to(self.torch_device).float()  # all data entries share the same time point
                point = b['point'].to(self.torch_device).float()
                # freq = b['freq'].unsqueeze(1).float()
                amp = b['amp'].float()
                print()
                # sign = b['sign'].unsqueeze(1).float()
                # latent_v = torch.cat([freq, amp, sign], dim=1).to(self.torch_device)
                #print(t.size(), point.size(), freq.size(), amp.size())
                # >> torch.Size([64, 100]) torch.Size([64, 100, 1]) torch.Size([64]) torch.Size([64])
                # print(torch.cat([freq, amp, sign], dim=1).size())
                # >> torch.Size([64, 3])
                pred = self.model(t, point[:,0,:], latent_v=amp).permute(1,0,2)
                loss = self.cal_loss(pred, point)
                loss.backward()
                self.optimizer.step()

                print("[Train] epoch:{}/{}, iter:{}/{}, loss: {}".format(i,self.args.niter,
                                                                 j, len(self.dl['train']),
                                                                 loss.item()))
                iter+=1
            # valid_loss = self.valid(self.dl['valid'])
            # print("[Validation] epoch:{}/{}, iter:{}/{}, loss: {}".format(i,self.args.niter,
            #                                                               iter,len(self.dl['train'])*self.args.niter,
            #                                                               valid_loss.item()))
            # if bset_mse>valid_loss:
            #     self.model_save(epoch=i, loss=valid_loss)
            if i % 20 == 0:
                self.model_save(epoch=i, loss=loss)

    def valid(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            valid_loss = 0
            total_data_size = 0
            for i, b in enumerate(dataloader):
                t = b['t'][0].to(self.torch_device).float()  # all data entries share the same time point
                point = b['point'].to(self.torch_device).float()
                freq = b['freq'].unsqueeze(1).float()
                amp = b['amp'].unsqueeze(1).float()
                sign = b['sign'].unsqueeze(1).float()
                latent_v = torch.cat([freq, amp, sign], dim=1).to(self.torch_device)
                pred = self.model(t, point[:,0,:], latent_v=latent_v).permute(1,0,2)

                loss = torch.pow(pred-point,2).sum()
                total_data_size += torch.ones_like(point).sum().item()
                valid_loss+=loss

        return valid_loss/total_data_size

    def model_save(self, epoch, loss):
        print("saved epoch {}".format(epoch))
        path = "/data/private/generativeODE/galerkin_pretest/{}".format(self.args.exid)
        self.check_direcory('/data/private/generativeODE/galerkin_pretest')
        self.check_direcory(path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        },path+"/best_model_"+str(self.seed))

    def check_direcory(self, directory):
        if not os.path.isdir(directory):
            os.mkdir(directory)  # {}".format(args.exp_idx)
